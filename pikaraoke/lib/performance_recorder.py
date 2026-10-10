"""Records mic+backing-track audio for each performance, for scoring.

Phase 1 only: capture a clean WAV per performance and prove the pipeline is
reliable. Nothing here scores anything yet -- that reads these files back in
a later phase.

Two backends, chosen by platform, same split as SoundManager's mic
passthrough: sounddevice (PortAudio) on Windows/macOS, parec (PulseAudio/
PipeWire) on Linux. Deliberately not routed through SoundManager itself,
which enumerates passthrough devices filtered to whatever shares a host API
with the default output; a scoring capture device is a fixed, explicitly
chosen input (the USB interface tapping the mixer) with no such constraint.
"""

import logging
import os
import re
import shutil
import subprocess
import wave
from datetime import datetime

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.get_platform import get_data_directory, is_linux
from pikaraoke.lib.playback_controller import PlaybackController
from pikaraoke.lib.preference_manager import PreferenceManager
from pikaraoke.lib.sound_manager import _pactl_list_sources

_SAMPLE_RATE = 48000
_CHANNELS = 2
_SAMPLE_WIDTH_BYTES = 2  # int16

_SOUNDDEVICE_AVAILABLE = False
sd = None
if not is_linux():
    try:
        import sounddevice as sd

        _SOUNDDEVICE_AVAILABLE = True
    except (ImportError, OSError):
        pass

_HAS_PAREC = is_linux() and shutil.which("parec") is not None

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def _recordings_directory() -> str:
    path = os.path.join(get_data_directory(), "recordings")
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_for_filename(text: str) -> str:
    """Strip characters that are illegal in a filename on Windows or POSIX."""
    return _UNSAFE_FILENAME_CHARS.sub("_", text).strip() or "unknown"


def _performance_filename(song: str, performer: str) -> str:
    return "{}_{}_{}.wav".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        _sanitize_for_filename(performer),
        _sanitize_for_filename(song),
    )


def list_input_devices() -> list[dict]:
    """Return every audio input device this platform's backend can see.

    Unfiltered, unlike SoundManager's enumeration: the scoring device is
    whatever the host picks explicitly (a USB interface, not necessarily on
    the same host API as the default output), so there is nothing to narrow
    the list by. On Linux, the returned "deviceId" is the PulseAudio/PipeWire
    source name -- what the scoring_input_device preference should hold,
    same value parec's --device wants.
    """
    if _HAS_PAREC:
        return [{"deviceId": d["paSource"], "label": d["label"]} for d in _pactl_list_sources()]
    if _SOUNDDEVICE_AVAILABLE:
        try:
            devices = sd.query_devices()
        except sd.PortAudioError as e:
            logging.error(f"Failed to query audio devices: {e}")
            return []
        return [
            {"deviceId": str(i), "label": dev["name"]}
            for i, dev in enumerate(devices)
            if dev["max_input_channels"] > 0
        ]
    return []


class PerformanceRecorder:
    """Captures one WAV file per performance from a chosen input device.

    Starts on "playback_started", stops and finalizes on "song_ended". Reads
    the song/performer directly off PlaybackController rather than off the
    event, which carries no payload -- other listeners of the same events
    (e.g. Karaoke.update_now_playing_socket) already read state this way
    rather than threading it through emit().

    Silently does nothing when no device is configured or neither backend is
    available, so this is always safe to construct.
    """

    def __init__(
        self,
        preferences: PreferenceManager,
        events: EventSystem,
        playback_controller: PlaybackController,
    ) -> None:
        self._preferences = preferences
        self._events = events
        self._playback_controller = playback_controller
        self._stream = None  # sounddevice backend
        self._wave_file: wave.Wave_write | None = None  # sounddevice backend
        self._process: subprocess.Popen | None = None  # parec backend
        self._current_file_path: str | None = None

        events.on("playback_started", self._on_playback_started)
        events.on("song_ended", self._on_song_ended)

    @property
    def available(self) -> bool:
        return _SOUNDDEVICE_AVAILABLE or _HAS_PAREC

    def _configured_device_id(self) -> str | None:
        device_id = self._preferences.get_or_default("scoring_input_device")
        return device_id or None

    def _is_recording(self) -> bool:
        return self._stream is not None or self._process is not None

    def _on_playback_started(self) -> None:
        device_id = self._configured_device_id()
        if device_id is None:
            return
        if not self.available:
            logging.warning("Scoring input device configured but no capture backend is available")
            return
        if self._is_recording():
            # A transpose restart ends and immediately restarts the stream
            # without a song_ended in between finishing first; keep recording
            # into the same file rather than leaking the old stream/process.
            return

        song = self._playback_controller.now_playing or "unknown"
        performer = self._playback_controller.now_playing_user or "unknown"
        file_path = os.path.join(_recordings_directory(), _performance_filename(song, performer))

        if _HAS_PAREC:
            self._start_parec(device_id, file_path)
        else:
            self._start_sounddevice(device_id, file_path)

    def _start_sounddevice(self, device_id: str, file_path: str) -> None:
        try:
            self._wave_file = wave.open(file_path, "wb")
            self._current_file_path = file_path
            self._wave_file.setnchannels(_CHANNELS)
            self._wave_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
            self._wave_file.setframerate(_SAMPLE_RATE)

            # RawInputStream, not InputStream: the latter wraps each callback's
            # data in a numpy array for convenience, which pulls in numpy as an
            # undeclared runtime dependency the moment a stream actually opens.
            # The callback below only ever does bytes(indata), so the raw CFFI
            # buffer a RawInputStream hands back is all it needs -- same
            # reasoning SoundManager's mic passthrough already uses RawStream for.
            self._stream = sd.RawInputStream(
                device=int(device_id),
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="int16",
                callback=self._on_audio_block,
            )
            self._stream.start()
            logging.info(f"Recording performance to {file_path}")
        except (sd.PortAudioError, OSError) as e:
            logging.error(f"Failed to start performance recording: {e}")
            self._cleanup(discard=True)

    def _start_parec(self, device_id: str, file_path: str) -> None:
        try:
            # parec writes a complete WAV itself (via --file-format=wav) and
            # finalizes the header on a graceful stop -- see _cleanup -- so
            # there is no equivalent of the sounddevice backend's manual
            # wave.open()/writeframes() bookkeeping to do here.
            self._process = subprocess.Popen(
                [
                    "parec",
                    f"--device={device_id}",
                    f"--rate={_SAMPLE_RATE}",
                    f"--channels={_CHANNELS}",
                    "--format=s16le",
                    "--file-format=wav",
                    file_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._current_file_path = file_path
            logging.info(f"Recording performance to {file_path}")
        except OSError as e:
            logging.error(f"Failed to start performance recording: {e}")
            self._cleanup(discard=True)

    def _on_audio_block(self, indata, frames, time_info, status) -> None:
        if status:
            logging.debug(f"Performance recording stream status: {status}")
        if self._wave_file is not None:
            self._wave_file.writeframes(bytes(indata))

    def _on_song_ended(self, reason: str | None = None) -> None:
        if reason == "transpose":
            # Same performance continuing in a new key; PlaybackController
            # restarts the stream right after this, and _on_playback_started
            # already treats an in-progress recording as "keep going".
            return
        self._cleanup(discard=False)

    def _cleanup(self, discard: bool) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except (sd.PortAudioError, OSError) as e:
                logging.warning(f"Error closing performance recording stream: {e}")
            self._stream = None

        if self._wave_file is not None:
            self._wave_file.close()
            self._wave_file = None

        if self._process is not None:
            # A graceful stop (not kill) is what makes parec fix up the WAV
            # header's size fields before exiting -- the same reason a
            # terminal recording is normally stopped with ctrl-c rather than
            # killed. Falls back to a kill only if it won't stop, in which
            # case the file below is corrupt and discard-worthy regardless.
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("parec did not exit after terminate(); killing it")
                self._process.kill()
                self._process.wait()
                discard = True
            self._process = None

        if self._current_file_path is not None:
            if discard:
                try:
                    os.remove(self._current_file_path)
                except OSError:
                    pass
            else:
                self._events.emit("performance_recorded", self._current_file_path)
            self._current_file_path = None
