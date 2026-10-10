"""Unit tests for vocal_isolator module."""

import math
import os
import subprocess
import wave
from array import array
from unittest.mock import patch

from pikaraoke.lib.vocal_isolator import (
    _apply_offset,
    _best_fit_gain,
    _find_alignment_offset,
    _read_mono_downmix,
    _subtract,
    _write_mono_wav,
    isolate_vocals,
)

_SAMPLE_RATE = 48000


def _tone(seconds, frequency=440, amplitude=10000, sample_rate=_SAMPLE_RATE):
    n = int(seconds * sample_rate)
    return array(
        "h",
        (int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(n)),
    )


def _write_stereo_wav(path, samples: array, sample_rate=_SAMPLE_RATE):
    wf: wave.Wave_write = wave.open(str(path), "wb")
    try:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        interleaved = array("h")
        for s in samples:
            interleaved.append(s)
            interleaved.append(s)
        wf.writeframes(interleaved.tobytes())
    finally:
        wf.close()


class TestAlignmentAndSubtraction:
    def test_find_alignment_offset_recovers_a_known_shift(self):
        reference = _tone(5)
        shift = 4800  # 100ms at 48kHz
        recording = array("h", [0] * shift) + reference

        offset = _find_alignment_offset(recording, reference)

        # The search is coarse (strided), so exact-to-the-sample isn't the bar.
        assert abs(offset - (-shift)) < 500

    def test_apply_offset_pads_a_negative_offset(self):
        reference = array("h", [1, 2, 3, 4])

        result = _apply_offset(reference, -2, target_length=6)

        assert list(result) == [0, 0, 1, 2, 3, 4]

    def test_apply_offset_trims_a_positive_offset(self):
        reference = array("h", [1, 2, 3, 4, 5])

        result = _apply_offset(reference, 2, target_length=3)

        assert list(result) == [3, 4, 5]

    def test_best_fit_gain_recovers_a_known_scale(self):
        reference = _tone(1, amplitude=10000)
        recording = array("h", [int(s * 0.5) for s in reference])

        gain = _best_fit_gain(recording, reference)

        assert abs(gain - 0.5) < 0.05

    def test_subtract_cancels_a_pure_copy(self):
        reference = _tone(1, amplitude=10000)
        recording = reference

        residual = _subtract(recording, reference, gain=1.0)

        assert max(abs(s) for s in residual) < 50

    def test_subtract_leaves_an_added_signal_behind(self):
        reference = _tone(1, frequency=440, amplitude=10000)
        vocal = _tone(1, frequency=880, amplitude=8000)
        recording = array("h", [r + v for r, v in zip(reference, vocal)])

        residual = _subtract(recording, reference, gain=1.0)

        # What's left should look like the vocal tone, not silence.
        assert max(abs(s) for s in residual) > 4000


class TestReadWriteMonoWav:
    def test_read_mono_downmix_handles_stereo(self, tmp_path):
        path = tmp_path / "stereo.wav"
        _write_stereo_wav(path, array("h", [100, -100, 200]))

        samples = _read_mono_downmix(str(path))

        assert list(samples) == [100, -100, 200]

    def test_write_then_read_roundtrips(self, tmp_path):
        """_read_mono_downmix only ever reads 48kHz inputs in this module (the
        recording and the ffmpeg-extracted reference) -- a reduced-rate
        residual is score_participation()'s concern, which reads rate
        generically rather than assuming 48kHz.
        """
        path = tmp_path / "mono.wav"
        original = array("h", [1, -1, 1000, -1000])

        _write_mono_wav(str(path), original, sample_rate=_SAMPLE_RATE)

        with wave.open(str(path), "rb") as wf:
            assert wf.getframerate() == _SAMPLE_RATE
            assert wf.getnchannels() == 1
        assert list(_read_mono_downmix(str(path))) == list(original)


class TestIsolateVocals:
    def test_returns_none_when_ffmpeg_fails(self, tmp_path):
        recording_path = tmp_path / "recording.wav"
        _write_stereo_wav(recording_path, _tone(2))

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            result = isolate_vocals(str(recording_path), "/songs/whatever.mp4")

        assert result is None

    def test_produces_a_residual_file_when_ffmpeg_succeeds(self, tmp_path):
        reference_tone = _tone(5, amplitude=10000)
        recording = array("h", [int(s * 0.7) for s in reference_tone])
        recording_path = tmp_path / "recording.wav"
        _write_stereo_wav(recording_path, recording)

        def fake_ffmpeg_run(args, **kwargs):
            output_path = args[-1]
            _write_mono_wav(output_path, reference_tone, _SAMPLE_RATE)
            return subprocess.CompletedProcess(args, 0)

        with patch("subprocess.run", side_effect=fake_ffmpeg_run):
            result = isolate_vocals(str(recording_path), "/songs/whatever.mp4")

        assert result is not None
        assert result.endswith("_vocals.wav")
        with wave.open(result, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0

    def test_cleans_up_the_extracted_reference_file(self, tmp_path):
        recording_path = tmp_path / "recording.wav"
        _write_stereo_wav(recording_path, _tone(2))
        extracted_paths = []

        def fake_ffmpeg_run(args, **kwargs):
            output_path = args[-1]
            extracted_paths.append(output_path)
            _write_mono_wav(output_path, _tone(2), _SAMPLE_RATE)
            return subprocess.CompletedProcess(args, 0)

        with patch("subprocess.run", side_effect=fake_ffmpeg_run):
            isolate_vocals(str(recording_path), "/songs/whatever.mp4")

        assert extracted_paths
        assert not os.path.exists(extracted_paths[0])
