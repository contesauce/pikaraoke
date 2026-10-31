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
_VOICE_BAND = f"aresample={_DECODE_RATE},highpass=f=200,lowpass=f=3500"
_RECORDING_FILTER = f"aformat=channel_layouts=mono,{_VOICE_BAND}"
# Which channels of the song reach the mixer depends on the cable: a stereo
# plug in a mono input carries only the left channel, and in a balanced input
# left minus right. Sounds panned to one side are then louder or quieter in
# the recording than in a both-channel reference, on every hit -- reading as
# singing once a bar. The reference is decoded all four ways in one pass and
# the one whose envelope best matches the recording is used. Mono files are
# upmixed first, leaving their "difference" channel silent and never chosen.
REFERENCE_CHANNELS = ("both", "left", "right", "difference")
_REFERENCE_FILTER = (
    "aformat=channel_layouts=stereo,"
    "pan=4c|c0=0.5*c0+0.5*c1|c1=c0|c2=c1|c3=0.5*c0-0.5*c1,"
    f"{_VOICE_BAND}"
)
_SLICE_SAMPLES = 160  # 20ms at _DECODE_RATE
_SLICE_MS = 20
_SLICES_PER_WINDOW = 5  # scored in 100ms windows
WINDOW_SECONDS = _SLICE_MS * _SLICES_PER_WINDOW / 1000
# Stream startup and an avsync setting both offset the recording from the
# file; the search covers that much either way.
_MAX_OFFSET_SLICES = 250  # 5s
_ALIGNMENT_SLICES = 3000  # first 60s is plenty to lock the offset
# The music-to-recording gain is the ratio seen when nobody is singing. On
# real hardware that ratio wobbles about +-1.5dB from window to window, but
# it clusters tightly around the true gain, and singing only ever moves
# ratios above it. So the gain is the lowest substantial peak of the ratio
# histogram: the cluster itself, not a low percentile (which sat below the
# cluster and read normal wobble as singing -- a song with nobody singing
# scored 99), and not the tallest peak (which is the singing, for someone
# singing most of the song).
_MODE_BIN_DB = 0.5
_MIN_PEAK_SHARE = 0.2
# The gain is re-estimated per block from the windows around it rather than
# once per song: a hand on the mixer's music fader, or a volume step when the
# browser starts playing, changes it mid-song.
_GAIN_BLOCK_WINDOWS = 25  # 2.5s
_GAIN_CONTEXT_WINDOWS = 75  # 7.5s either side
_MIN_GAIN_SAMPLES = 20
_MUSIC_PRESENT_FRACTION = 0.01  # 20dB under the song's median level
# A window counts as sung when it is this far above what music plus room
# noise alone would make it: at least _MIN_MARGIN_DB, wider where the music
# itself wobbles more. The wobble is measured from ratios below the mode,
# which singing never reaches.
_MIN_MARGIN_DB = 2.0
_MARGIN_SPREADS = 2.0
# Share of windows that earns full marks. Songs have intros, solos, and
# breaths between lines, so nobody sings through every 100ms of a song.
_FULL_CREDIT_FRACTION = 0.4
_MIN_WINDOWS = 50
# The start of each recording isn't scored. The splash screen's background
# music is still fading out for several seconds after a song starts, and that
# extra sound reads as singing; songs almost never have vocals this early.
_SKIPPED_START_WINDOWS = 80  # 8s
SKIPPED_START_SECONDS = _SKIPPED_START_WINDOWS * _SLICE_MS * _SLICES_PER_WINDOW / 1000


def recording_powers(path: str) -> list[float] | None:
    """Per-20ms band-passed power of a performance recording."""
    channels = _slice_powers(path, _RECORDING_FILTER, 1)
    return channels[0] if channels else None


def reference_powers(path: str, normalize_audio: bool) -> dict[str, list[float]] | None:
    """Per-20ms band-passed power of the song file as playback rendered it,
    for each of REFERENCE_CHANNELS.

    normalize_audio must match playback's setting: loudnorm boosts quiet
    passages, and without it on the reference those read as singing. That
    filter is slow (several seconds a song), which is why ParticipationScorer
    starts this when the song starts rather than when it ends.
    """
    audio_filter = _REFERENCE_FILTER
    if normalize_audio:
        options = ":".join(f"{key}={value}" for key, value in LOUDNORM_OPTIONS.items())
        audio_filter = f"loudnorm={options},{audio_filter}"
    channels = _slice_powers(path, audio_filter, len(REFERENCE_CHANNELS))
    return dict(zip(REFERENCE_CHANNELS, channels)) if channels else None


def score_performance(
    recording: list[float], references: dict[str, list[float]], label: str
) -> int | None:
    """Score a recording 0-99 by how much of it had vocals over the music.

    Returns None when there is nothing meaningful to measure, so the score
    screen falls back to its random score rather than a confident-looking 0.
    """
    analysis = analyze_performance(recording, references)
    if analysis is None:
        logging.info(f"Nothing to score in {label} (too short, or silent song)")
        return None
    channel, offset_ms, sung = analysis
    fraction = sum(sung) / len(sung)
    logging.info(
        f"Scoring {label}: {channel} channel(s), offset {offset_ms}ms, "
        f"sung in {fraction:.0%} of {len(sung)} windows"
    )
    return round(min(fraction / _FULL_CREDIT_FRACTION, 1.0) * 99)


def analyze_performance(
    recording: list[float], references: dict[str, list[float]]
) -> tuple[str, int, list[bool]] | None:
    """Return the matched reference channel, the offset in ms, and per 100ms
    window from SKIPPED_START_SECONDS on, whether it was sung.

    None when there is nothing meaningful to measure. Public so the
    explain_score script can show where the score came from.
    """
    matches = {name: _find_offset(recording, ref) for name, ref in references.items()}
    channel = max(matches, key=lambda name: matches[name][1])
    offset, correlation = matches[channel]
    if correlation == -math.inf:
        return None
    reference = _shift(references[channel], offset, len(recording))
    rec_windows = _to_windows(recording)
    ref_windows = _to_windows(reference)
    if len(rec_windows) < _SKIPPED_START_WINDOWS + _MIN_WINDOWS:
        return None

    model = _local_music_model(rec_windows, ref_windows)
    if model is None:
        return None
    # Room noise and mic hiss: what's left of the quietest windows once the
    # music is taken out. Taken from the recording alone, a performance sung
    # nearly end to end would put vocals into the "noise" too.
    residuals = [r - g * m for r, m, (g, _) in zip(rec_windows, ref_windows, model)]
    noise_floor = max(_percentile(residuals, 0.05), 0.0)

    sung = [
        r > margin * (g * m + noise_floor)
        for r, m, (g, margin) in zip(rec_windows, ref_windows, model)
    ]
    return channel, offset * _SLICE_MS, sung[_SKIPPED_START_WINDOWS:]


def _local_music_model(
    rec_windows: list[float], ref_windows: list[float]
) -> list[tuple[float, float]] | None:
    """Return, per window, the music-to-recording gain in effect and the
    factor above music-alone power that counts as singing.

    None when the reference never clearly plays, so there is no music to
    measure a gain against.
    """
    music_level = _percentile(ref_windows, 0.5)
    if music_level <= 0:
        return None
    # Quiet passages count too; only near-silence is left out, where room
    # noise rather than music sets the ratio.
    music_present = music_level * _MUSIC_PRESENT_FRACTION
    ratios_db = [
        10 * math.log10(r / m) if m > music_present and r > 0 else None
        for r, m in zip(rec_windows, ref_windows)
    ]
    song_ratios = [x for x in ratios_db if x is not None]
    if len(song_ratios) < _MIN_GAIN_SAMPLES:
        return None
    song_model = _music_alone(song_ratios)

    model = []
    for block_start in range(0, len(rec_windows), _GAIN_BLOCK_WINDOWS):
        context = ratios_db[
            max(0, block_start - _GAIN_CONTEXT_WINDOWS) : block_start
            + _GAIN_BLOCK_WINDOWS
            + _GAIN_CONTEXT_WINDOWS
        ]
        nearby = [x for x in context if x is not None]
        block_model = _music_alone(nearby) if len(nearby) >= _MIN_GAIN_SAMPLES else song_model
        model += [block_model] * _GAIN_BLOCK_WINDOWS
    return model[: len(rec_windows)]


def _music_alone(ratios_db: list[float]) -> tuple[float, float]:
    """Gain and singing-threshold factor from a set of recording/song ratios in dB."""
    bins: dict[int, int] = {}
    for value in ratios_db:
        key = round(value / _MODE_BIN_DB)
        bins[key] = bins.get(key, 0) + 1
    smoothed = {
        k: sum(w * bins.get(k + d, 0) for d, w in ((-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1)))
        for k in range(min(bins) - 2, max(bins) + 3)
    }
    tallest = max(smoothed.values())
    peak = min(
        k
        for k, height in smoothed.items()
        if height >= _MIN_PEAK_SHARE * tallest
        and height >= smoothed.get(k - 1, 0)
        and height >= smoothed.get(k + 1, 0)
    )
    center = peak * _MODE_BIN_DB
    near = [v for v in ratios_db if abs(v - center) <= 1.5 * _MODE_BIN_DB]
    mode = sum(near) / len(near)

    # Median deviation below the mode, scaled to a standard deviation.
    below = sorted(mode - v for v in ratios_db if v < mode)
    spread = below[len(below) // 2] / 0.674 if below else 0.0
    margin_db = max(_MIN_MARGIN_DB, _MARGIN_SPREADS * spread)
    return 10 ** (mode / 10), 10 ** (margin_db / 10)


def _slice_powers(path: str, audio_filter: str, channels: int) -> list[list[float]] | None:
    """Decode path through audio_filter and return, per output channel, the
    mean power of each 20ms slice.
    """
    args = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn"]
    args += ["-af", audio_filter, "-f", "s16le", "-"]
    try:
        result = subprocess.run(args, capture_output=True, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        stderr = getattr(e, "stderr", b"") or b""
        logging.warning(f"Could not decode {path} for scoring: {e} {stderr.decode()[-300:]}")
        return None

    frame_bytes = 2 * channels
    samples = array("h")
    samples.frombytes(result.stdout[: len(result.stdout) - len(result.stdout) % frame_bytes])
    return [_powers(samples[channel::channels]) for channel in range(channels)]


def _powers(samples: array) -> list[float]:
    return [
        sum(map(mul, chunk, chunk)) / _SLICE_SAMPLES
        for chunk in (
            samples[i : i + _SLICE_SAMPLES]
            for i in range(0, len(samples) - _SLICE_SAMPLES + 1, _SLICE_SAMPLES)
        )
    ]


def _find_offset(recording: list[float], reference: list[float]) -> tuple[int, float]:
    """Return how many slices later the music sits in the recording than in
    the file, and how well the envelopes correlate there.

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
    return best_offset, best_score


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

    def _take_reference(self, source_path: str) -> dict[str, list[float]] | None:
        """The reference prepared during the song, or one computed now if none was."""
        pending, self._pending_reference = self._pending_reference, None
        if pending is not None and pending[0] == source_path:
            return pending[1].get()
        return reference_powers(source_path, self._normalize_audio())
