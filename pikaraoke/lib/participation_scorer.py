"""Beginner-tier scoring: did the singer make sound while the song played.

Not pitch, not rhythm -- just whether the mic picked up sustained vocal
energy through the performance, as a fraction of a 0-99 score to match the
scale the (currently random) score screen already uses. Deliberately crude:
this is the tier that works on every performance with no reference data,
and later tiers are expected to replace it as the primary number once they
exist.
"""

import logging
import math
import wave
from array import array

from pikaraoke.lib.events import EventSystem

_WINDOW_MS = 100
# A window counts as "singing" once it's this many dB above the recording's
# own quiet-window level, so the threshold adapts to mic gain and room noise
# instead of a fixed absolute level that would misread a hot or quiet input.
_ACTIVE_MARGIN_DB = 12.0
# What "quiet" means for that baseline: the window level below which this
# fraction of the recording's windows fall. Low, deliberately -- a real
# performance is mostly singing, so the quiet baseline has to stay inside
# whatever sliver of true silence/instrumental exists rather than assuming
# any particular split between the two. Above the 0th percentile only so one
# outlier window (mic hiss on an otherwise dead channel) can't set it alone.
_QUIET_PERCENTILE = 0.05
_MIN_DBFS = -96.0  # floor for a window with zero signal, where dB is undefined
# Every Nth sample is plenty for a coarse energy reading and keeps a 4-5
# minute recording's pure-Python RMS pass to a few hundred ms rather than
# several seconds -- this is voice-activity detection, not a waveform
# analysis that needs every sample.
_RMS_SAMPLE_STRIDE = 4


def score_participation(wav_path: str) -> int | None:
    """Score a recording 0-99 by how much of it has sustained vocal energy.

    Returns None for a recording too short or too quiet throughout to say
    anything meaningful about -- callers fall back to the random score
    rather than reporting a confident-looking 0.
    """
    levels = _window_levels_dbfs(wav_path)
    if len(levels) < 10:
        return None

    sorted_levels = sorted(levels)
    quiet_level = _percentile(sorted_levels, _QUIET_PERCENTILE)
    peak_level = sorted_levels[-1]
    if peak_level - quiet_level < _ACTIVE_MARGIN_DB:
        # Nothing in the recording stands out from its own quiet baseline --
        # uniformly silent, or (much less likely) uniformly loud throughout.
        # Either way there is no contrast to call "active" versus not.
        return None

    threshold = quiet_level + _ACTIVE_MARGIN_DB
    active_fraction = sum(1 for level in levels if level >= threshold) / len(levels)
    return round(min(active_fraction, 1.0) * 99)


def _window_levels_dbfs(wav_path: str) -> list[float]:
    """Return one RMS dBFS reading per _WINDOW_MS window of the recording."""
    levels = []
    try:
        with wave.open(wav_path, "rb") as wf:
            frame_rate = wf.getframerate()
            window_frames = max(1, int(frame_rate * _WINDOW_MS / 1000))
            while True:
                raw = wf.readframes(window_frames)
                if not raw:
                    break
                samples = array("h")
                samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
                strided = samples[::_RMS_SAMPLE_STRIDE]
                if strided:
                    levels.append(_rms_dbfs(strided))
    except (wave.Error, OSError) as e:
        logging.warning(f"Failed to read recording for scoring: {wav_path}: {e}")
        return []
    return levels


def _rms_dbfs(samples: array) -> float:
    """RMS level of int16 samples, in dB relative to full scale."""
    mean_square = sum(s * s for s in samples) / len(samples)
    if mean_square <= 0:
        return _MIN_DBFS
    rms = mean_square**0.5
    full_scale = 32768.0
    return max(_MIN_DBFS, 20 * math.log10(rms / full_scale))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


class ParticipationScorer:
    """Scores a performance's recording as soon as PerformanceRecorder finalizes it.

    Listens for "performance_recorded" (file_path) and emits
    "performance_scored" (score: int) so Karaoke can relay it to the score
    screen. Emits nothing when scoring can't say anything meaningful, so the
    screen falls back to its usual random score rather than showing a
    misleadingly confident number.
    """

    def __init__(self, events: EventSystem) -> None:
        self._events = events
        events.on("performance_recorded", self._on_performance_recorded)

    def _on_performance_recorded(self, file_path: str) -> None:
        score = score_participation(file_path)
        if score is not None:
            self._events.emit("performance_scored", score)
