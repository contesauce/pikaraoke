"""Unit tests for participation_scorer module."""

import random
import subprocess
from array import array
from unittest.mock import patch

import gevent
import pytest

from pikaraoke.lib import participation_scorer as ps
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.participation_scorer import ParticipationScorer, score_performance
from pikaraoke.lib.preference_manager import PreferenceManager

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
    return score_performance(recording, {"both": reference}, "rec.wav")


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

        assert ps._find_offset(recording, music)[0] == delay


class TestSlicePowers:
    def test_returns_mean_power_per_slice_for_each_channel(self):
        left = [100] * 160 + [-200] * 160 + [7] * 10
        right = [10] * 330
        interleaved = array("h", (v for pair in zip(left, right) for v in pair))
        completed = subprocess.CompletedProcess([], 0, stdout=interleaved.tobytes(), stderr=b"")

        with patch.object(ps.subprocess, "run", return_value=completed) as run:
            powers = ps._slice_powers("song.mp4", "highpass=f=200", channels=2)

        assert powers == [[10000.0, 40000.0], [100.0, 100.0]]
        args = run.call_args.args[0]
        assert "song.mp4" in args
        assert args[args.index("-af") + 1] == "highpass=f=200"

    def test_returns_none_when_ffmpeg_fails(self):
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Invalid data")

        with patch.object(ps.subprocess, "run", side_effect=error):
            assert ps._slice_powers("song.zip", "highpass=f=200", channels=1) is None


class TestChannelMatching:
    def test_picks_the_channel_the_mixer_received(self):
        """A stereo plug in a mono input: only the left channel reaches the mixer.

        A beat panned hard left is then louder in the recording than in a
        both-channel reference on every hit, which must not read as singing.
        """
        music = _music_envelope()
        beat = [8e6 if i % 112 < 10 else 0.0 for i in range(_SLICES)]  # once a bar
        references = {
            "both": [m + b / 4 for m, b in zip(music, beat)],
            "left": [m + b for m, b in zip(music, beat)],
            "right": music,
        }
        recording = _recording(references["left"])

        channel, _offset, sung = ps.analyze_performance(recording, references)
        both_only = _score(recording, references["both"])

        assert channel == "left"
        assert sum(sung) / len(sung) < 0.05
        assert both_only > 20


class TestReferencePowers:
    def test_applies_loudnorm_only_when_playback_did(self):
        with patch.object(ps, "_slice_powers", return_value=[[1.0]] * 4) as slice_powers:
            ps.reference_powers("song.mp4", normalize_audio=True)
            ps.reference_powers("song.mp4", normalize_audio=False)

        normalized, plain = (c.args[1] for c in slice_powers.call_args_list)
        assert normalized.startswith("loudnorm=i=-16:tp=-1.5:lra=11,")
        assert "loudnorm" not in plain

    def test_returns_each_channel_by_name(self):
        channels = [[1.0], [2.0], [3.0], [4.0]]
        with patch.object(ps, "_slice_powers", return_value=channels):
            references = ps.reference_powers("song.mp4", normalize_audio=False)

        assert references == {"both": [1.0], "left": [2.0], "right": [3.0], "difference": [4.0]}


class TestParticipationScorer:
    @pytest.fixture
    def preferences(self, tmp_path):
        return PreferenceManager(str(tmp_path / "config.ini"))

    @pytest.fixture
    def events(self):
        return EventSystem()

    @pytest.fixture
    def scorer(self, events, preferences):
        return ParticipationScorer(events, preferences)

    def test_scores_in_a_greenlet_off_the_event_handler(self, scorer, events):
        with patch.object(ps, "spawn") as spawn:
            events.emit("performance_recorded", "rec.wav", "song.mp4")

        spawn.assert_called_once_with(scorer.score_recording, "rec.wav", "song.mp4")

    def test_prepares_the_reference_when_the_song_starts(self, scorer, events, preferences):
        preferences.set("normalize_audio", True)

        with patch.object(ps, "spawn") as spawn:
            events.emit("performance_started", "song.mp4")

        spawn.assert_called_once_with(ps.reference_powers, "song.mp4", True)

    def test_scores_against_the_prepared_reference(self, scorer, events):
        with patch.object(ps, "reference_powers", return_value={"both": [2.0]}) as reference:
            events.emit("performance_started", "song.mp4")
            gevent.sleep(0)
        with (
            patch.object(ps, "reference_powers") as late_reference,
            patch.object(ps, "recording_powers", return_value=[1.0]),
            patch.object(ps, "score_performance", return_value=72) as score,
        ):
            scorer.score_recording("rec.wav", "song.mp4")

        reference.assert_called_once()
        late_reference.assert_not_called()
        assert score.call_args.args[:2] == ([1.0], {"both": [2.0]})

    def test_computes_the_reference_late_for_a_different_song(self, scorer, events, preferences):
        with patch.object(ps, "reference_powers", return_value={"both": [2.0]}):
            events.emit("performance_started", "other.mp4")
            gevent.sleep(0)
        with (
            patch.object(ps, "reference_powers", return_value={"both": [3.0]}) as late_reference,
            patch.object(ps, "recording_powers", return_value=[1.0]),
            patch.object(ps, "score_performance", return_value=72),
        ):
            scorer.score_recording("rec.wav", "song.mp4")

        late_reference.assert_called_once_with("song.mp4", False)

    def test_emits_the_score(self, scorer, events):
        scores = []
        events.on("performance_scored", scores.append)

        with (
            patch.object(ps, "reference_powers", return_value={"both": [2.0]}),
            patch.object(ps, "recording_powers", return_value=[1.0]),
            patch.object(ps, "score_performance", return_value=72),
        ):
            scorer.score_recording("rec.wav", "song.mp4")

        assert scores == [72]

    def test_stays_silent_and_logs_when_a_decode_fails(self, scorer, events, caplog):
        scores = []
        events.on("performance_scored", scores.append)

        with (
            patch.object(ps, "reference_powers", return_value=None),
            patch.object(ps, "recording_powers", return_value=[1.0]),
            caplog.at_level("INFO"),
        ):
            scorer.score_recording("rec.wav", "song.mp4")

        assert scores == []
        assert "Could not compute a participation score" in caplog.text

    def test_skips_when_the_song_file_is_unknown(self, scorer, events):
        scores = []
        events.on("performance_scored", scores.append)

        with patch.object(ps, "recording_powers") as recording:
            scorer.score_recording("rec.wav", None)

        recording.assert_not_called()
        assert scores == []
