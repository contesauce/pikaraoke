"""Unit tests for participation_scorer module."""

import random
import subprocess
from array import array
from unittest.mock import patch

import pytest

from pikaraoke.lib import participation_scorer as ps
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.participation_scorer import ParticipationScorer, score_participation

_SECONDS = 120
_SLICES = _SECONDS * 50  # 20ms slices


def _music_envelope(seed=1) -> list[float]:
    """A backing track's per-slice power: loud sections, quiet bridges, jitter."""
    rng = random.Random(seed)
    envelope = []
    for i in range(_SLICES):
        section_level = 4e6 if (i // 500) % 3 else 1e6
        envelope.append(section_level * rng.uniform(0.5, 1.5))
    return envelope


def _recording(music, gain=0.5, delay=0, sung_fraction=0.0, vocal_power=4e6, seed=2):
    """Mix the music envelope with vocals on the first sung_fraction of each 2s phrase.

    gain may be a function of the slice index, for a gain that changes mid-song.
    """
    rng = random.Random(seed)
    gain_at = gain if callable(gain) else lambda _i: gain
    delayed = [0.0] * delay + music[: len(music) - delay]
    phrase = 100
    recording = []
    for i, m in enumerate(delayed):
        vocal = vocal_power * rng.uniform(0.7, 1.3) if i % phrase < sung_fraction * phrase else 0
        recording.append(gain_at(i) * m + 1e3 + vocal)
    return recording


def _score(recording, reference):
    def fake_slice_powers(path):
        return {"rec.wav": recording, "song.mp4": reference}[path]

    with patch.object(ps, "_slice_powers", side_effect=fake_slice_powers):
        return score_participation("rec.wav", "song.mp4")


class TestScoreParticipation:
    def test_music_alone_scores_low_even_though_it_is_loud(self):
        music = _music_envelope()

        score = _score(_recording(music, sung_fraction=0.0), music)

        assert score is not None
        assert score < 15

    def test_singing_throughout_scores_high(self):
        music = _music_envelope()

        score = _score(_recording(music, sung_fraction=0.6), music)

        assert score is not None
        assert score > 85

    def test_half_hearted_singing_lands_in_between(self):
        music = _music_envelope()

        low = _score(_recording(music, sung_fraction=0.0), music)
        mid = _score(_recording(music, sung_fraction=0.2), music)
        high = _score(_recording(music, sung_fraction=0.6), music)

        assert low < mid < high

    def test_tolerates_the_recording_starting_late(self):
        music = _music_envelope()

        score = _score(_recording(music, delay=37, sung_fraction=0.0), music)

        assert score is not None
        assert score < 15

    def test_returns_none_when_a_decode_fails(self):
        music = _music_envelope()

        assert _score(None, music) is None
        assert _score(_recording(music), None) is None

    def test_returns_none_for_too_short_a_recording(self):
        music = _music_envelope()[:100]

        assert _score(_recording(music), music) is None

    def test_returns_none_for_a_silent_reference(self):
        music = _music_envelope()

        assert _score(_recording(music), [0.0] * _SLICES) is None

    def test_follows_the_music_fader_moving_mid_song(self):
        """Someone turns the music up halfway through; that alone isn't singing.

        An instant jump reads as singing for a few seconds around it, until
        the local gain catches up -- a real fader move is gradual.
        """
        music = _music_envelope()
        half = _SLICES // 2

        score = _score(_recording(music, gain=lambda i: 0.3 if i < half else 0.6), music)

        assert score is not None
        assert score < 25

    def test_follows_a_slow_gain_drift_like_loudnorm(self):
        music = _music_envelope()

        score = _score(_recording(music, gain=lambda i: 0.3 + 0.4 * i / _SLICES), music)

        assert score is not None
        assert score < 15

    def test_singing_nearly_end_to_end_still_scores_high(self):
        """Only breaths between lines are unsung; the music gain must not absorb the voice."""
        music = _music_envelope()

        score = _score(_recording(music, sung_fraction=0.9), music)

        assert score is not None
        assert score > 85

    def test_a_voice_mixed_under_the_music_still_counts(self):
        music = _music_envelope()

        score = _score(_recording(music, sung_fraction=0.6, vocal_power=1e6), music)

        assert score is not None
        assert score > 40


class TestFindOffset:
    @pytest.mark.parametrize("delay", [0, 12, 200, -40])
    def test_recovers_a_known_offset(self, delay):
        music = _music_envelope()
        if delay >= 0:
            recording = [0.0] * delay + music
        else:
            recording = music[-delay:]

        assert ps._find_offset(recording, music) == delay


class TestSlicePowers:
    def test_returns_mean_power_per_slice(self):
        samples = array("h", [100] * 160 + [-200] * 160 + [7] * 10)
        completed = subprocess.CompletedProcess([], 0, stdout=samples.tobytes(), stderr=b"")

        with patch.object(ps.subprocess, "run", return_value=completed) as run:
            powers = ps._slice_powers("song.mp4")

        assert powers == [10000.0, 40000.0]
        assert "song.mp4" in run.call_args.args[0]

    def test_returns_none_when_ffmpeg_fails(self):
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Invalid data")

        with patch.object(ps.subprocess, "run", side_effect=error):
            assert ps._slice_powers("song.zip") is None


class TestParticipationScorer:
    @pytest.fixture
    def scorer(self):
        events = EventSystem()
        return ParticipationScorer(events), events

    def test_scores_in_a_greenlet_off_the_event_handler(self, scorer):
        scorer_obj, events = scorer

        with patch.object(ps, "spawn") as spawn:
            events.emit("performance_recorded", "rec.wav", "song.mp4")

        spawn.assert_called_once_with(scorer_obj.score_recording, "rec.wav", "song.mp4")

    def test_emits_the_score(self, scorer):
        scorer_obj, events = scorer
        scores = []
        events.on("performance_scored", scores.append)

        with patch.object(ps, "score_participation", return_value=72):
            scorer_obj.score_recording("rec.wav", "song.mp4")

        assert scores == [72]

    def test_stays_silent_and_logs_when_unscorable(self, scorer, caplog):
        scorer_obj, events = scorer
        scores = []
        events.on("performance_scored", scores.append)

        with patch.object(ps, "score_participation", return_value=None):
            with caplog.at_level("INFO"):
                scorer_obj.score_recording("rec.wav", "song.mp4")

        assert scores == []
        assert "Could not compute a participation score" in caplog.text

    def test_skips_when_the_song_file_is_unknown(self, scorer):
        scorer_obj, events = scorer
        scores = []
        events.on("performance_scored", scores.append)

        with patch.object(ps, "score_participation") as score:
            scorer_obj.score_recording("rec.wav", None)

        score.assert_not_called()
        assert scores == []
