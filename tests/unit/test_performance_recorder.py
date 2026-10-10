"""Unit tests for performance_recorder module."""

import subprocess
import wave
from unittest.mock import MagicMock

import pytest

import pikaraoke.lib.performance_recorder as performance_recorder
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.performance_recorder import PerformanceRecorder, list_input_devices
from pikaraoke.lib.playback_controller import PlaybackController
from pikaraoke.lib.preference_manager import PreferenceManager


class _FakePortAudioError(Exception):
    pass


class _FakeInputStream:
    """Stands in for sd.RawInputStream, capturing the callback so a test can drive it."""

    instances: list["_FakeInputStream"] = []

    def __init__(self, device, samplerate, channels, dtype, callback):
        self.device = device
        self.callback = callback
        self.started = False
        self.closed = False
        _FakeInputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clear_fake_stream_instances():
    _FakeInputStream.instances.clear()
    yield
    _FakeInputStream.instances.clear()


@pytest.fixture
def fake_sd(monkeypatch):
    """A minimal fake sounddevice module, wired in as available."""
    fake = MagicMock()
    fake.RawInputStream = _FakeInputStream
    fake.PortAudioError = _FakePortAudioError
    fake.query_devices.return_value = [
        {"name": "Speakers", "max_input_channels": 0},
        {"name": "UF0202 Line In", "max_input_channels": 2},
    ]
    monkeypatch.setattr(performance_recorder, "sd", fake)
    monkeypatch.setattr(performance_recorder, "_SOUNDDEVICE_AVAILABLE", True)
    return fake


class _FakeParecProcess:
    """Stands in for the subprocess.Popen handle of a running parec.

    Touches the target file on construction, same as parec does almost
    immediately after starting -- tests can assert on the recordings
    directory without a real parec binary.
    """

    instances: list["_FakeParecProcess"] = []

    def __init__(self, args, stdout=None, stderr=None):
        self.args = args
        self.file_path = args[-1]
        self.terminated = False
        self.killed = False
        self.hang_on_terminate = False
        self._waits = 0
        with open(self.file_path, "wb"):
            pass
        _FakeParecProcess.instances.append(self)

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self._waits += 1
        if self.hang_on_terminate and self._waits == 1:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout)

    def kill(self):
        self.killed = True


@pytest.fixture(autouse=True)
def clear_fake_parec_instances():
    _FakeParecProcess.instances.clear()
    yield
    _FakeParecProcess.instances.clear()


@pytest.fixture
def fake_parec(monkeypatch):
    """Wires in the parec (Linux) backend as available, with a fake Popen."""
    monkeypatch.setattr(performance_recorder, "_HAS_PAREC", True)
    monkeypatch.setattr(performance_recorder.subprocess, "Popen", _FakeParecProcess)
    return _FakeParecProcess


@pytest.fixture
def prefs(tmp_path):
    return PreferenceManager(str(tmp_path / "config.ini"))


@pytest.fixture
def events():
    return EventSystem()


@pytest.fixture
def playback_controller(prefs, events):
    return PlaybackController(prefs, events, filename_from_path=lambda p, remove_youtube_id=True: p)


@pytest.fixture
def recordings_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(performance_recorder, "get_data_directory", lambda: str(tmp_path))
    return tmp_path / "recordings"


def _start_a_song(playback_controller, song="Never Gonna Give You Up", performer="Rick"):
    playback_controller.now_playing = song
    playback_controller.now_playing_user = performer


class TestAvailability:
    def test_unavailable_without_sounddevice(self, prefs, events, playback_controller, monkeypatch):
        monkeypatch.setattr(performance_recorder, "_SOUNDDEVICE_AVAILABLE", False)
        recorder = PerformanceRecorder(prefs, events, playback_controller)
        assert recorder.available is False

    def test_available_with_sounddevice(self, prefs, events, playback_controller, fake_sd):
        recorder = PerformanceRecorder(prefs, events, playback_controller)
        assert recorder.available is True


class TestNoDeviceConfigured:
    def test_does_nothing_when_no_device_configured(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")

        assert _FakeInputStream.instances == []
        assert not recordings_dir.exists()


class TestRecordingLifecycle:
    def test_records_a_wav_for_the_performance(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        prefs.set("scoring_input_device", "1")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller, song="Africa", performer="Toto")

        events.emit("playback_started")
        stream = _FakeInputStream.instances[0]
        assert stream.started is True

        stream.callback(b"\x01\x02" * 10, 10, None, None)
        events.emit("song_ended", "complete")

        assert stream.closed is True
        files = list(recordings_dir.glob("*.wav"))
        assert len(files) == 1
        assert "Toto" in files[0].name
        assert "Africa" in files[0].name

        with wave.open(str(files[0]), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 48000
            assert wf.getnframes() == 5  # 20 bytes / (2 channels * 2 bytes)

    def test_emits_performance_recorded_with_the_finished_file(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        prefs.set("scoring_input_device", "1")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)
        recorded_paths = []
        events.on("performance_recorded", lambda path: recorded_paths.append(path))

        events.emit("playback_started")
        events.emit("song_ended", "complete")

        assert recorded_paths == [str(list(recordings_dir.glob("*.wav"))[0])]

    def test_does_not_emit_performance_recorded_when_discarded(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        def _raise(*args, **kwargs):
            raise _FakePortAudioError("no such device")

        fake_sd.RawInputStream = _raise
        prefs.set("scoring_input_device", "1")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)
        recorded_paths = []
        events.on("performance_recorded", lambda path: recorded_paths.append(path))

        events.emit("playback_started")

        assert recorded_paths == []

    def test_transpose_keeps_recording_into_the_same_file(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        prefs.set("scoring_input_device", "1")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")
        events.emit("song_ended", "transpose")
        # PlaybackController restarts the stream right after a transpose ends it.
        events.emit("playback_started")

        # Only the original stream was ever created; the second start was a no-op.
        assert len(_FakeInputStream.instances) == 1

    def test_skip_still_finalizes_the_recording(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        prefs.set("scoring_input_device", "1")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")
        events.emit("song_ended", "skip")

        assert len(list(recordings_dir.glob("*.wav"))) == 1

    def test_failed_stream_start_discards_the_file(
        self, prefs, events, playback_controller, fake_sd, recordings_dir
    ):
        def _raise(*args, **kwargs):
            raise _FakePortAudioError("no such device")

        fake_sd.RawInputStream = _raise
        prefs.set("scoring_input_device", "1")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")

        assert list(recordings_dir.glob("*.wav")) == []


class TestListInputDevices:
    def test_filters_out_devices_with_no_input_channels(self, fake_sd):
        devices = list_input_devices()
        assert devices == [{"deviceId": "1", "label": "UF0202 Line In"}]

    def test_empty_when_sounddevice_unavailable(self, monkeypatch):
        monkeypatch.setattr(performance_recorder, "_SOUNDDEVICE_AVAILABLE", False)
        assert list_input_devices() == []

    def test_prefers_parec_devices_over_sounddevice_when_both_available(self, fake_sd, monkeypatch):
        monkeypatch.setattr(performance_recorder, "_HAS_PAREC", True)
        monkeypatch.setattr(
            performance_recorder,
            "_pactl_list_sources",
            lambda: [{"deviceId": "0", "label": "ignored", "paSource": "alsa_input.usb-UF0202"}],
        )

        devices = list_input_devices()

        assert devices == [{"deviceId": "alsa_input.usb-UF0202", "label": "ignored"}]


class TestParecBackend:
    """The Linux capture backend (parec), parallel to TestRecordingLifecycle."""

    def test_records_a_wav_for_the_performance(
        self, prefs, events, playback_controller, fake_parec, recordings_dir
    ):
        prefs.set("scoring_input_device", "alsa_input.usb-UF0202")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller, song="Africa", performer="Toto")

        events.emit("playback_started")
        process = fake_parec.instances[0]
        assert "--device=alsa_input.usb-UF0202" in process.args
        assert process.file_path == str(list(recordings_dir.glob("*.wav"))[0])

        events.emit("song_ended", "complete")

        assert process.terminated is True

    def test_emits_performance_recorded_with_the_finished_file(
        self, prefs, events, playback_controller, fake_parec, recordings_dir
    ):
        prefs.set("scoring_input_device", "alsa_input.usb-UF0202")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)
        recorded_paths = []
        events.on("performance_recorded", lambda path: recorded_paths.append(path))

        events.emit("playback_started")
        events.emit("song_ended", "complete")

        assert recorded_paths == [str(list(recordings_dir.glob("*.wav"))[0])]

    def test_kills_and_discards_when_parec_wont_stop_gracefully(
        self, prefs, events, playback_controller, fake_parec, recordings_dir
    ):
        prefs.set("scoring_input_device", "alsa_input.usb-UF0202")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)
        recorded_paths = []
        events.on("performance_recorded", lambda path: recorded_paths.append(path))

        events.emit("playback_started")
        fake_parec.instances[0].hang_on_terminate = True
        events.emit("song_ended", "complete")

        assert fake_parec.instances[0].killed is True
        assert recorded_paths == []
        assert list(recordings_dir.glob("*.wav")) == []

    def test_transpose_keeps_recording_into_the_same_file(
        self, prefs, events, playback_controller, fake_parec, recordings_dir
    ):
        prefs.set("scoring_input_device", "alsa_input.usb-UF0202")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")
        events.emit("song_ended", "transpose")
        events.emit("playback_started")

        assert len(fake_parec.instances) == 1

    def test_skip_still_finalizes_the_recording(
        self, prefs, events, playback_controller, fake_parec, recordings_dir
    ):
        prefs.set("scoring_input_device", "alsa_input.usb-UF0202")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")
        events.emit("song_ended", "skip")

        assert len(list(recordings_dir.glob("*.wav"))) == 1

    def test_failed_start_discards_the_file(
        self, prefs, events, playback_controller, fake_parec, recordings_dir, monkeypatch
    ):
        def _raise(*args, **kwargs):
            raise OSError("parec not found")

        monkeypatch.setattr(performance_recorder.subprocess, "Popen", _raise)
        prefs.set("scoring_input_device", "alsa_input.usb-UF0202")
        PerformanceRecorder(prefs, events, playback_controller)
        _start_a_song(playback_controller)

        events.emit("playback_started")

        assert list(recordings_dir.glob("*.wav")) == []
