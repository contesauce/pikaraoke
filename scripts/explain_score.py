"""Show when in a performance the participation scorer heard singing.

For tuning scoring on real hardware: lists the stretches of a recording
that counted as sung, so you can listen to the WAV at those times and hear
whether it was a singer, the room, or music leaking through the mics.
Scores against the song both with and without playback's loudness
normalization, since a mismatch there is the likeliest false-singing cause.

Usage:
    uv run python scripts/explain_score.py [RECORDING.wav] [SONG_FILE]

With no arguments, takes the newest recording and finds its song in
~/pikaraoke-songs by title.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import pikaraoke modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pikaraoke.lib.participation_scorer import (
    WINDOW_SECONDS,
    analyze_performance,
    recording_powers,
    reference_powers,
    score_performance,
)
from pikaraoke.lib.performance_recorder import _sanitize_for_filename

RECORDINGS_DIR = Path("~/.pikaraoke/recordings").expanduser()
SONGS_DIR = Path("~/pikaraoke-songs").expanduser()
# Stretches shorter than this are listed only in the totals, not one by one.
_MIN_LISTED_SECONDS = 0.5


def newest_recording() -> Path | None:
    recordings = sorted(RECORDINGS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    return recordings[-1] if recordings else None


def song_title(song_file: Path) -> str:
    """The display title PiKaraoke derives from a song filename."""
    stem = song_file.stem
    for marker in ("---", " ["):
        stem = stem.split(marker)[0]
    return stem


def find_song(recording: Path, songs_dir: Path) -> Path | None:
    """Find the song a recording was made over, by the title in its filename."""
    matches = [
        song
        for song in songs_dir.rglob("*")
        if song.is_file()
        and recording.stem.endswith("_" + _sanitize_for_filename(song_title(song)))
    ]
    return max(matches, key=lambda p: len(p.name)) if matches else None


def sung_spans(sung: list[bool]) -> list[tuple[float, float]]:
    """Merge consecutive sung windows into (start, end) times in seconds."""
    spans = []
    start = None
    for i, is_sung in enumerate(sung + [False]):
        if is_sung and start is None:
            start = i
        elif not is_sung and start is not None:
            spans.append((start * WINDOW_SECONDS, i * WINDOW_SECONDS))
            start = None
    return spans


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def explain(recording: list[float], reference: list[float], label: str) -> None:
    analysis = analyze_performance(recording, reference)
    if analysis is None:
        print(f"{label}: nothing to analyze")
        return
    offset_ms, sung = analysis
    score = score_performance(recording, reference, label)
    spans = sung_spans(sung)
    listed = [(a, b) for a, b in spans if b - a >= _MIN_LISTED_SECONDS]
    print(f"\n{label}")
    print(f"  Offset {offset_ms}ms, sung {sum(sung) / len(sung):.0%}, score {score}")
    print(f"  {len(spans)} sung stretches, {len(listed)} of them {_MIN_LISTED_SECONDS}s or longer:")
    for start, end in listed[:15]:
        print(f"    {_clock(start)} - {_clock(end)}  ({end - start:.1f}s)")
    if len(listed) > 15:
        print(f"    ...and {len(listed) - 15} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", nargs="?", help="default: newest recording")
    parser.add_argument("song", nargs="?", help="default: found by the recording's title")
    args = parser.parse_args()

    recording_path = Path(args.recording) if args.recording else newest_recording()
    if recording_path is None:
        sys.exit(f"No recordings in {RECORDINGS_DIR}")
    song_path = Path(args.song) if args.song else find_song(recording_path, SONGS_DIR)
    if song_path is None:
        sys.exit(f"Could not find the song for {recording_path.name} in {SONGS_DIR}")
    print(f"Recording: {recording_path.name}")
    print(f"Song:      {song_path.name}")

    recording = recording_powers(str(recording_path))
    if recording is None:
        sys.exit("Could not decode the recording")
    for normalized in (True, False):
        reference = reference_powers(str(song_path), normalized)
        if reference is None:
            sys.exit("Could not decode the song")
        label = "With normalization" if normalized else "Without normalization"
        explain(recording, reference, label)


if __name__ == "__main__":
    main()
