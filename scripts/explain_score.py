"""Show when in a performance the participation scorer heard singing.

For tuning scoring on real hardware: lists the stretches of a recording
that counted as sung, so you can listen to the WAV at those times and hear
whether it was a singer, the room, or music leaking through the mics.

Usage:
    uv run python scripts/explain_score.py RECORDING.wav SONG_FILE
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import pikaraoke modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pikaraoke.lib.participation_scorer import (
    WINDOW_SECONDS,
    analyze_performance,
    score_participation,
)

# Stretches shorter than this are listed only in the totals, not one by one.
_MIN_LISTED_SECONDS = 0.5


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", help="WAV from ~/.pikaraoke/recordings")
    parser.add_argument("song", help="the song file that was played")
    args = parser.parse_args()

    analysis = analyze_performance(args.recording, args.song)
    if analysis is None:
        print("Could not analyze (see warnings above).")
        return
    offset_ms, sung = analysis
    spans = sung_spans(sung)
    listed = [(a, b) for a, b in spans if b - a >= _MIN_LISTED_SECONDS]

    print(f"Offset: {offset_ms}ms")
    print(f"Sung: {sum(sung) / len(sung):.0%} of {len(sung) * WINDOW_SECONDS:.0f}s")
    print(f"Score: {score_participation(args.recording, args.song)}")
    print(
        f"{len(spans)} sung stretches, {len(spans) - len(listed)} shorter than "
        f"{_MIN_LISTED_SECONDS}s. Longer ones:"
    )
    for start, end in listed:
        print(f"  {_clock(start)} - {_clock(end)}  ({end - start:.1f}s)")


if __name__ == "__main__":
    main()
