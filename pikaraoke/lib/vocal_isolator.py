"""Subtracts the known backing track from a mixed performance recording.

PerformanceRecorder's capture comes off the mixer's post-mix output, which
carries the backing track at full volume whether or not anyone is singing --
scoring that recording directly can't tell "instrumental playing" from
"instrumental playing while someone sings over it" (both read as loud,
sustained energy). Since pikaraoke already knows exactly which file was
playing, this subtracts that known reference from the mixed recording
instead of trying to detect vocals blindly, leaving a residual that's
mostly voice.

Deliberately simple: one best-fit gain scalar, not an adaptive multi-tap
filter, and a bounded, heavily-downsampled alignment search rather than a
full cross-correlation -- both chosen to stay fast in pure Python without a
numpy/scipy dependency. Good enough to break "always scores high because
the music is loud"; not studio-grade cancellation. A transposed performance
gets its reference pitch-shifted to match what was actually played, so
cancellation degrades gracefully rather than failing outright, but won't be
as clean as an untransposed one -- ffmpeg's rubberband shift is an
approximation of the same shift the stream itself applied, not a replay of
the identical audio.
"""

import logging
import os
import subprocess
import tempfile
import wave
from array import array

_SAMPLE_RATE = 48000
_SAMPLE_WIDTH_BYTES = 2  # int16

# How far off the recording's start can plausibly be from the reference's,
# given stream buffering before PerformanceRecorder's capture begins.
# Widening this costs alignment-search time quadratically in the window
# below, so both stay conservative rather than chasing unlikely drift.
_ALIGNMENT_SEARCH_RANGE_SECONDS = 2.0
# How much audio the alignment search looks at to find that offset. Short on
# purpose, and almost always the song's instrumental intro/lead-in -- there
# is usually little or no vocal there yet, which is fine: this is aligning
# the recording's instrumental content against the reference's, not vocals.
_ALIGNMENT_WINDOW_SECONDS = 3.0
# Alignment is a coarse search, not the final subtraction, so it works on a
# heavily downsampled copy of both signals to keep the O(range * window)
# search fast in pure Python. 500Hz still resolves offsets to 2ms, plenty
# for subtracting a mixer tap's worth of drift.
_ALIGNMENT_SAMPLE_RATE = 500
# Every Nth sample for the final gain fit and subtraction pass, same
# reasoning and value as participation_scorer's RMS stride: a full-resolution
# pass over several minutes of audio is pure-Python-slow for no real benefit
# here. The residual is written at the correspondingly reduced rate rather
# than reconstructed to full rate -- score_participation() reads whatever
# rate a WAV declares, so there is nothing to gain from inflating it back.
_SUBTRACTION_SAMPLE_STRIDE = 4


def isolate_vocals(
    recording_path: str, reference_source_path: str, transpose_semitones: int = 0
) -> str | None:
    """Produce a vocal-dominant residual WAV alongside the recording.

    Returns the residual's path, or None if isolation failed for any reason
    (missing/corrupt source, ffmpeg unavailable, no usable alignment) --
    callers should fall back to scoring the raw recording rather than
    treating this as fatal.
    """
    reference_wav_path = None
    try:
        reference_wav_path = _extract_reference_wav(reference_source_path, transpose_semitones)
        recording_samples = _read_mono_downmix(recording_path)
        reference_samples = _read_mono_downmix(reference_wav_path)
        if recording_samples is None or reference_samples is None:
            return None

        offset = _find_alignment_offset(recording_samples, reference_samples)
        aligned_reference = _apply_offset(reference_samples, offset, len(recording_samples))

        gain = _best_fit_gain(recording_samples, aligned_reference)
        residual = _subtract(recording_samples, aligned_reference, gain)

        residual_path = _residual_path_for(recording_path)
        residual_rate = _SAMPLE_RATE // _SUBTRACTION_SAMPLE_STRIDE
        _write_mono_wav(residual_path, residual, residual_rate)
        return residual_path
    except (subprocess.CalledProcessError, OSError, wave.Error) as e:
        logging.warning(f"Vocal isolation failed for {recording_path}: {e}")
        return None
    finally:
        if reference_wav_path is not None:
            try:
                os.remove(reference_wav_path)
            except OSError:
                pass


def _residual_path_for(recording_path: str) -> str:
    base, _ext = os.path.splitext(recording_path)
    return base + "_vocals.wav"


def _extract_reference_wav(source_path: str, transpose_semitones: int) -> str:
    """Extract source_path's audio as mono 48kHz WAV, pitch-shifted to match
    what was actually played, if it was transposed.
    """
    fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="pikaraoke_reference_")
    os.close(fd)

    args = ["ffmpeg", "-y", "-i", source_path]
    if transpose_semitones != 0:
        pitch = 2 ** (transpose_semitones / 12)
        args += ["-af", f"rubberband=pitch={pitch}"]
    args += ["-ac", "1", "-ar", str(_SAMPLE_RATE), output_path]

    subprocess.run(args, capture_output=True, check=True)
    return output_path


def _read_mono_downmix(wav_path: str) -> array | None:
    """Read a WAV file's samples as mono int16, downmixing stereo if needed."""
    with wave.open(wav_path, "rb") as wf:
        if wf.getframerate() != _SAMPLE_RATE or wf.getsampwidth() != _SAMPLE_WIDTH_BYTES:
            logging.warning(f"Unexpected format reading {wav_path} for vocal isolation")
            return None
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % (2 * channels))])
    if channels == 1:
        return samples
    if channels == 2:
        return array("h", ((samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)))
    logging.warning(f"Unsupported channel count {channels} reading {wav_path}")
    return None


def _find_alignment_offset(recording: array, reference: array) -> int:
    """Return reference's best-fit start offset into recording, in samples.

    Positive means the reference should be read starting that many samples
    in (it started before the recording did); negative means that many
    silent samples should be prepended to it (the recording started first).
    """
    stride = max(1, _SAMPLE_RATE // _ALIGNMENT_SAMPLE_RATE)
    window_samples = int(_ALIGNMENT_WINDOW_SECONDS * _SAMPLE_RATE) // stride
    search_range = int(_ALIGNMENT_SEARCH_RANGE_SECONDS * _SAMPLE_RATE) // stride

    rec_strided = recording[::stride][:window_samples]
    ref_strided = reference[::stride]

    best_offset_strided = 0
    best_score = None
    for candidate in range(-search_range, search_range + 1):
        ref_start = max(0, candidate)
        rec_start = max(0, -candidate)
        length = min(len(rec_strided) - rec_start, len(ref_strided) - ref_start)
        if length <= 0:
            continue
        score = sum(rec_strided[rec_start + i] * ref_strided[ref_start + i] for i in range(length))
        if best_score is None or score > best_score:
            best_score = score
            best_offset_strided = candidate

    return best_offset_strided * stride


def _apply_offset(reference: array, offset: int, target_length: int) -> array:
    """Shift reference by offset and pad/trim it to target_length."""
    if offset >= 0:
        shifted = reference[offset:]
    else:
        shifted = array("h", [0] * -offset) + reference

    if len(shifted) < target_length:
        shifted = shifted + array("h", [0] * (target_length - len(shifted)))
    return shifted[:target_length]


def _best_fit_gain(recording: array, reference: array) -> float:
    """Least-squares scalar gain mapping reference onto its copy within recording."""
    stride = _SUBTRACTION_SAMPLE_STRIDE
    numerator = 0
    denominator = 0
    for i in range(0, len(recording), stride):
        r = reference[i]
        numerator += recording[i] * r
        denominator += r * r
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _subtract(recording: array, reference: array, gain: float) -> array:
    """Subtract gain*reference from recording, every _SUBTRACTION_SAMPLE_STRIDE'th
    sample only -- see that constant's comment for why the residual comes out
    at a reduced rate instead of full resolution.
    """
    stride = _SUBTRACTION_SAMPLE_STRIDE
    residual = array("h")
    for i in range(0, len(recording), stride):
        value = int(recording[i] - gain * reference[i])
        residual.append(max(-32768, min(32767, value)))
    return residual


def _write_mono_wav(path: str, samples: array, sample_rate: int) -> None:
    wf: wave.Wave_write = wave.open(path, "wb")
    try:
        wf.setnchannels(1)
        wf.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    finally:
        wf.close()
