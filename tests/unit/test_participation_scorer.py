"""Unit tests for participation_scorer module."""

import math
import wave

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.participation_scorer import ParticipationScorer, score_participation

_SAMPLE_RATE = 48000
_CHANNELS = 2


def _write_wav(path, seconds_of_silence, seconds_of_tone, amplitude=20000, frequency=440):
    """Write a WAV alternating a block of silence then a block of a sine tone.

    A sine tone stands in for "singing": sustained energy well above silence,
    without needing a real recording fixture.
    """
    wf: wave.Wave_write = wave.open(str(path), "wb")
    try:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)

        silence_frames = int(_SAMPLE_RATE * seconds_of_silence)
        wf.writeframes(b"\x00\x00" * _CHANNELS * silence_frames)

        tone_frames = int(_SAMPLE_RATE * seconds_of_tone)
        samples = bytearray()
        for i in range(tone_frames):
            value = int(amplitude * math.sin(2 * math.pi * frequency * i / _SAMPLE_RATE))
            packed = value.to_bytes(2, byteorder="little", signed=True)
            samples += packed * _CHANNELS
        wf.writeframes(bytes(samples))
    finally:
        wf.close()


class TestScoreParticipation:
    def test_mostly_silent_recording_scores_low(self, tmp_path):
        path = tmp_path / "silent.wav"
        _write_wav(path, seconds_of_silence=9, seconds_of_tone=1)

        score = score_participation(str(path))

        assert score is not None
        assert score < 20

    def test_mostly_singing_recording_scores_high(self, tmp_path):
        path = tmp_path / "singing.wav"
        _write_wav(path, seconds_of_silence=1, seconds_of_tone=9)

        score = score_participation(str(path))

        assert score is not None
        assert score > 80

    def test_score_is_within_the_0_to_99_scale(self, tmp_path):
        path = tmp_path / "singing.wav"
        _write_wav(path, seconds_of_silence=1, seconds_of_tone=4)

        score = score_participation(str(path))

        assert score is not None
        assert 0 <= score <= 99

    def test_returns_none_for_a_constant_recording_with_no_contrast(self, tmp_path):
        """A recording with no quiet moments at all has nothing to score against."""
        path = tmp_path / "constant_tone.wav"
        _write_wav(path, seconds_of_silence=0, seconds_of_tone=5)

        assert score_participation(str(path)) is None

    def test_returns_none_for_totally_silent_recording(self, tmp_path):
        path = tmp_path / "totally_silent.wav"
        _write_wav(path, seconds_of_silence=5, seconds_of_tone=0)

        assert score_participation(str(path)) is None

    def test_returns_none_for_too_short_a_recording(self, tmp_path):
        path = tmp_path / "tiny.wav"
        _write_wav(path, seconds_of_silence=0, seconds_of_tone=0.05)

        assert score_participation(str(path)) is None

    def test_returns_none_for_a_missing_file(self, tmp_path):
        assert score_participation(str(tmp_path / "does_not_exist.wav")) is None

    def test_returns_none_for_a_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.wav"
        path.write_bytes(b"not actually a wav file")

        assert score_participation(str(path)) is None


class TestParticipationScorer:
    def test_emits_performance_scored_when_scorable(self, tmp_path):
        events = EventSystem()
        ParticipationScorer(events)
        scores = []
        events.on("performance_scored", lambda score: scores.append(score))

        path = tmp_path / "singing.wav"
        _write_wav(path, seconds_of_silence=1, seconds_of_tone=9)
        events.emit("performance_recorded", str(path))

        assert len(scores) == 1
        assert scores[0] > 80

    def test_stays_silent_when_recording_is_not_scorable(self, tmp_path):
        events = EventSystem()
        ParticipationScorer(events)
        scores = []
        events.on("performance_scored", lambda score: scores.append(score))

        path = tmp_path / "totally_silent.wav"
        _write_wav(path, seconds_of_silence=5, seconds_of_tone=0)
        events.emit("performance_recorded", str(path))

        assert scores == []
