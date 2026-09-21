"""Records mic+backing-track audio for each performance, for scoring.

Phase 1 only: capture a clean WAV per performance and prove the pipeline is
reliable. Nothing here scores anything yet -- that reads these files back in
a later phase. Uses sounddevice (PortAudio) directly rather than going through
SoundManager, which enumerates devices for passthrough (filtered to whatever
shares a host API with the default output); a scoring capture device is a
fixed, explicitly chosen input (the USB interface tapping the mixer) with no
such constraint.
"""

import logging
import os
import re
import wave
from datetime import datetime

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.get_platform import get_data_directory
from pikaraoke.lib.playback_controller import PlaybackController
from pikaraoke.lib.preference_manager import PreferenceManager

_SAMPLE_RATE = 48000
_CHANNELS = 2
_SAMPLE_WIDTH_BYTES = 2  # int16

_SOUNDDEVICE_AVAILABLE = False
sd = None
try:
    import sounddevice as sd

    _SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    pass

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def _recordings_directory() -> str:
    path = os.path.join(get_data_directory(), "recordings")
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_for_filename(text: str) -> str:
    """Strip characters that are illegal in a filename on Windows or POSIX."""
    return _UNSAFE_FILENAME_CHARS.sub("_", text).strip() or "unknown"


def list_input_devices() -> list[dict]:
    """Return every audio input device sounddevice can see.

    Unfiltered, unlike SoundManager's enumeration: the scoring device is
    whatever the host picks explicitly (a USB interface, not necessarily on
    the same host API as the default output), so there is nothing to narrow
    the list by.
    """
    if not _SOUNDDEVICE_AVAILABLE:
        return []
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


class PerformanceRecorder:
    """Captures one WAV file per performance from a chosen input device.

    Starts on "playback_started", stops and finalizes on "song_ended". Reads
    the song/performer directly off PlaybackController rather than off the
    event, which carries no payload -- other listeners of the same events
    (e.g. Karaoke.update_now_playing_socket) already read state this way
    rather than threading it through emit().

    Silently does nothing when no device is configured or sounddevice is
    unavailable (Linux does not install it -- see pyproject.toml), so this is
    always safe to construct.
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
        self._stream = None
        self._wave_file: wave.Wave_write | None = None
        self._current_file_path: str | None = None

        events.on("playback_started", self._on_playback_started)
        events.on("song_ended", self._on_song_ended)

    @property
    def available(self) -> bool:
        return _SOUNDDEVICE_AVAILABLE

    def _configured_device_id(self) -> str | None:
        device_id = self._preferences.get_or_default("scoring_input_device")
        return device_id or None

    def _on_playback_started(self) -> None:
        device_id = self._configured_device_id()
        if device_id is None:
            return
        if not self.available:
            logging.warning("Scoring input device configured but sounddevice is unavailable")
            return
        if self._stream is not None:
            # A transpose restart ends and immediately restarts the stream
            # without a song_ended in between finishing first; keep recording
            # into the same file rather than leaking the old stream.
            return

        song = self._playback_controller.now_playing or "unknown"
        performer = self._playback_controller.now_playing_user or "unknown"
        filename = "{}_{}_{}.wav".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            _sanitize_for_filename(performer),
            _sanitize_for_filename(song),
        )
        file_path = os.path.join(_recordings_directory(), filename)

        try:
            self._wave_file = wave.open(file_path, "wb")
            self._current_file_path = file_path
            self._wave_file.setnchannels(_CHANNELS)
            self._wave_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
            self._wave_file.setframerate(_SAMPLE_RATE)

            self._stream = sd.InputStream(
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

    def _on_audio_block(self, indata, frames, time_info, status) -> None:
        if status:
            logging.debug(f"Performance recording stream status: {status}")
        if self._wave_file is not None:
            self._wave_file.writeframes(bytes(indata))

    def _on_song_ended(self, reason: str | None = None) -> None:
        if reason == "transpose":
            # Same performance continuing in a new key; PlaybackController
            # restarts the stream right after this, and _on_playback_started
            # already treats an in-progress stream as "keep going".
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
            if discard and self._current_file_path is not None:
                try:
                    os.remove(self._current_file_path)
                except OSError:
                    pass
            elif not discard and self._current_file_path is not None:
                self._events.emit("performance_recorded", self._current_file_path)
            self._current_file_path = None
