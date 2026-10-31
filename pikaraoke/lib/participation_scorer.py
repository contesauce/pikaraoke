"""Beginner-tier scoring: did the singer make sound while the song played.

Not pitch, not rhythm -- just how much of the performance had vocal energy
on top of the backing track. The capture is the mixer's post-mix output, so
it carries mic and backing track summed together; scoring it alone reads as
"was the music loud". pikaraoke knows which file was playing, so this
compares the recording's loudness envelope against that file's, and counts
the moments where the recording is clearly louder than the music alone
explains.

Envelopes, not waveforms: subtracting the reference sample-by-sample needs
sample-accurate alignment, which a USB interface's clock drifting against
the playback device's never gives over a whole song. Loudness in 20ms
slices only needs alignment to within a slice, and uncorrelated signals add
in power, so "recording power minus scaled reference power" is the vocal
power without any phase matching.
"""

import logging
import math
import subprocess
from array import array
from operator import mul

from gevent import Greenlet, spawn

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.ffmpeg import LOUDNORM_OPTIONS
from pikaraoke.lib.preference_manager import PreferenceManager

_DECODE_RATE = 8000
# Both signals go through the same band-pass, centered on the voice. Bass
# and cymbals carry most of a backing track's power and none of a singer's,
# so cutting them makes vocals a larger share of what is compared. Downmixed
# and resampled first, since filtering at 8kHz is several times cheaper than
# at the capture's 48kHz.
_VOICE_BAND_FILTER = (
    f"aformat=channel_layouts=mono,aresample={_DECODE_RATE},highpass=f=200,lowpass=f=3500"
)
_SLICE_SAMPLES = 160  # 20ms at _DECODE_RATE
_SLICE_MS = 20
_SLICES_PER_WINDOW = 5  # scored in 100ms windows
WINDOW_SECONDS = _SLICE_MS * _SLICES_PER_WINDOW / 1000
# Stream startup and an avsync setting both offset the recording from the
# file; the search covers that much either way.
_MAX_OFFSET_SLICES = 250  # 5s
_ALIGNMENT_SLICES = 3000  # first 60s is plenty to lock the offset
# The music-to-recording gain is the ratio seen when nobody is singing. Not
# knowing when that is, take a low percentile over windows where the music
# is clearly playing: this assumes at least that share of them (intro,
# breaks, breaths between lines) has no voice on it. Vocals only ever add
# power, so the low tail is where the music-alone ratio lives.
_GAIN_PERCENTILE = 0.05
# The gain is re-estimated per block from the windows around it rather than
# once per song: playback's loudnorm option and a hand on the mixer's music
# fader both change it mid-song. Applying loudnorm to the reference instead
# would match playback exactly but takes longer than the score screen waits.
_GAIN_BLOCK_WINDOWS = 25  # 2.5s
_GAIN_CONTEXT_WINDOWS = 75  # 7.5s either side
_MIN_GAIN_SAMPLES = 20
_MUSIC_PRESENT_FRACTION = 0.01  # 20dB under the song's median level
# A window counts as sung when it is this much louder than music plus room
# noise alone would make it. Music alone stays within about 0.7dB of that in
# testing; 2dB credits a voice mixed a little under the music.
_ACTIVE_MARGIN_DB = 2.0
# Share of windows that earns full marks. Songs have intros, solos, and
# breaths between lines, so nobody sings through every 100ms of a song.
_FULL_CREDIT_FRACTION = 0.4
_MIN_WINDOWS = 50


def recording_powers(path: str) -> list[float] | None:
    """Per-20ms band-passed power of a performance recording."""
    return _slice_powers(path, _VOICE_BAND_FILTER)


def reference_powers(path: str, normalize_audio: bool) -> list[float] | None:
    """Per-20ms band-passed power of the song file, as playback rendered it.

    normalize_audio must match playback's setting: loudnorm boosts quiet
    passages, and without it on the reference those read as singing. That
    filter is slow (several seconds a song), which is why ParticipationScorer
    starts this when the song starts rather than when it ends.
    """
    audio_filter = _VOICE_BAND_FILTER
    if normalize_audio:
        options = ":".join(f"{key}={value}" for key, value in LOUDNORM_OPTIONS.items())
        audio_filter = f"loudnorm={options},{audio_filter}"
    return _slice_powers(path, audio_filter)


def score_performance(recording: list[float], reference: list[float], label: str) -> int | None:
    """Score a recording 0-99 by how much of it had vocals over the music.

    Returns None when there is nothing meaningful to measure, so the score
    screen falls back to its random score rather than a confident-looking 0.
    """
    analysis = analyze_performance(recording, reference)
    if analysis is None:
        logging.info(f"Nothing to score in {label} (too short, or silent song)")
        return None
    offset_ms, sung = analysis
    fraction = sum(sung) / len(sung)
    logging.info(
        f"Scoring {label}: offset {offset_ms}ms, sung in {fraction:.0%} of {len(sung)} windows"
    )
    return round(min(fraction / _FULL_CREDIT_FRACTION, 1.0) * 99)


def analyze_performance(
    recording: list[float], reference: list[float]
) -> tuple[int, list[bool]] | None:
    """Return the detected offset in ms and, per 100ms window, whether it was sung.

    None when there is nothing meaningful to measure. Public so the
    explain_score script can show where the score came from.
    """
    offset = _find_offset(recording, reference)
    reference = _shift(reference, offset, len(recording))
    rec_windows = _to_windows(recording)
    ref_windows = _to_windows(reference)
    if len(rec_windows) < _MIN_WINDOWS:
        return None

    gains = _local_gains(rec_windows, ref_windows)
    if gains is None:
        return None
    # Room noise and mic hiss: what's left of the quietest windows once the
    # music is taken out. Taken from the recording alone, a performance sung
    # nearly end to end would put vocals into the "noise" too.
    residuals = [r - g * m for r, m, g in zip(rec_windows, ref_windows, gains)]
    noise_floor = max(_percentile(residuals, 0.05), 0.0)

    margin = 10 ** (_ACTIVE_MARGIN_DB / 10)
    sung = [r > margin * (g * m + noise_floor) for r, m, g in zip(rec_windows, ref_windows, gains)]
    return offset * _SLICE_MS, sung


def _local_gains(rec_windows: list[float], ref_windows: list[float]) -> list[float] | None:
    """Return the music-to-recording gain in effect at each window.

    None when the reference never clearly plays, so there is no music to
    measure a gain against.
    """
    music_level = _percentile(ref_windows, 0.5)
    if music_level <= 0:
        return None
    # Quiet passages count too: playback's loudnorm boosts them relative to
    # the loud ones, so a gain learned only from loud passages reads every
    # quiet one as a voice. Only near-silence is left out, where room noise
    # rather than music sets the ratio.
    music_present = music_level * _MUSIC_PRESENT_FRACTION
    ratios = [r / m if m > music_present else None for r, m in zip(rec_windows, ref_windows)]
    song_ratios = [x for x in ratios if x is not None]
    if not song_ratios:
        return None
    song_gain = _percentile(song_ratios, _GAIN_PERCENTILE)

    gains = []
    for block_start in range(0, len(rec_windows), _GAIN_BLOCK_WINDOWS):
        context = ratios[
            max(0, block_start - _GAIN_CONTEXT_WINDOWS) : block_start
            + _GAIN_BLOCK_WINDOWS
            + _GAIN_CONTEXT_WINDOWS
        ]
        nearby = [x for x in context if x is not None]
        gain = (
            _percentile(nearby, _GAIN_PERCENTILE) if len(nearby) >= _MIN_GAIN_SAMPLES else song_gain
        )
        gains += [gain] * _GAIN_BLOCK_WINDOWS
    return gains[: len(rec_windows)]


def _slice_powers(path: str, audio_filter: str) -> list[float] | None:
    """Decode path through audio_filter and return mean power per 20ms slice."""
    args = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn"]
    args += ["-af", audio_filter, "-f", "s16le", "-"]
    try:
        result = subprocess.run(args, capture_output=True, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        stderr = getattr(e, "stderr", b"") or b""
        logging.warning(f"Could not decode {path} for scoring: {e} {stderr.decode()[-300:]}")
        return None

    samples = array("h")
    samples.frombytes(result.stdout[: len(result.stdout) - len(result.stdout) % 2])
    return [
        sum(map(mul, chunk, chunk)) / _SLICE_SAMPLES
        for chunk in (
            samples[i : i + _SLICE_SAMPLES]
            for i in range(0, len(samples) - _SLICE_SAMPLES + 1, _SLICE_SAMPLES)
        )
    ]


def _find_offset(recording: list[float], reference: list[float]) -> int:
    """Return how many slices later the music sits in the recording than in the file.

    Correlates log-loudness envelopes, which follow the song's structure
    (verses, drops, stops) regardless of the gain between the two signals.
    Pearson rather than a raw dot product, so a silent stretch at either
    start can't outweigh the music itself.
    """
    span = _ALIGNMENT_SLICES + _MAX_OFFSET_SLICES
    rec = [math.log10(p + 1.0) for p in recording[:span]]
    ref = [math.log10(p + 1.0) for p in reference[:span]]
    best_offset, best_score = 0, -math.inf
    for offset in range(-_MAX_OFFSET_SLICES, _MAX_OFFSET_SLICES + 1):
        rec_part = rec[max(0, offset) :]
        ref_part = ref[max(0, -offset) :]
        length = min(len(rec_part), len(ref_part), _ALIGNMENT_SLICES)
        if length < _MIN_WINDOWS:
            continue
        score = _pearson(rec_part[:length], ref_part[:length])
        if score > best_score:
            best_offset, best_score = offset, score
    return best_offset


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    sum_a, sum_b = sum(a), sum(b)
    covariance = sum(map(mul, a, b)) - sum_a * sum_b / n
    variance_a = sum(map(mul, a, a)) - sum_a * sum_a / n
    variance_b = sum(map(mul, b, b)) - sum_b * sum_b / n
    if variance_a <= 0 or variance_b <= 0:
        return -math.inf
    return covariance / math.sqrt(variance_a * variance_b)


def _shift(reference: list[float], offset: int, length: int) -> list[float]:
    """Line reference up with the recording and pad/trim it to length."""
    shifted = [0.0] * offset + reference if offset >= 0 else reference[-offset:]
    return (shifted + [0.0] * length)[:length]


def _to_windows(slices: list[float]) -> list[float]:
    n = _SLICES_PER_WINDOW
    return [sum(slices[i : i + n]) / n for i in range(0, len(slices) - n + 1, n)]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


class ParticipationScorer:
    """Scores a performance as soon as PerformanceRecorder finalizes it.

    Listens for "performance_started" (source_path) to prepare the song's
    reference while it plays, and "performance_recorded" (file_path,
    source_path) to score. Emits "performance_scored" (score: int) for
    Karaoke to relay to the score screen, or nothing when no score can be
    measured, so the screen falls back to its random score.

    All work runs in greenlets: the events arrive inside socket handlers,
    and the server is a single gevent loop, so scoring inline would stall
    every page and the score's own delivery until it finished.
    """

    def __init__(self, events: EventSystem, preferences: PreferenceManager) -> None:
        self._events = events
        self._preferences = preferences
        self._pending_reference: tuple[str, Greenlet] | None = None
        events.on("performance_started", self._on_performance_started)
        events.on("performance_recorded", self._on_performance_recorded)

    def _normalize_audio(self) -> bool:
        return bool(self._preferences.get_or_default("normalize_audio"))

    def _on_performance_started(self, source_path: str) -> None:
        greenlet = spawn(reference_powers, source_path, self._normalize_audio())
        self._pending_reference = (source_path, greenlet)

    def _on_performance_recorded(self, file_path: str, source_path: str | None) -> None:
        spawn(self.score_recording, file_path, source_path)

    def score_recording(self, file_path: str, source_path: str | None) -> None:
        if source_path is None:
            logging.info(f"No song file known for {file_path}; skipping scoring")
            return
        reference = self._take_reference(source_path)
        recording = recording_powers(file_path)
        score = None
        if reference is not None and recording is not None:
            score = score_performance(recording, reference, file_path)
        if score is None:
            logging.info(
                f"Could not compute a participation score for {file_path} -- "
                "score screen will fall back to the random score"
            )
            return
        logging.info(f"Participation score {score} for {file_path}")
        self._events.emit("performance_scored", score)

    def _take_reference(self, source_path: str) -> list[float] | None:
        """The reference prepared during the song, or one computed now if none was."""
        pending, self._pending_reference = self._pending_reference, None
        if pending is not None and pending[0] == source_path:
            return pending[1].get()
        return reference_powers(source_path, self._normalize_audio())
