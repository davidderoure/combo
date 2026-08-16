"""Manual sanity check for the ensemble skeleton (DESIGN.md §2/§4) — no MIDI needed.

    python -m ensemble.demo                          # blues_in_f.chart, machine speed
    python -m ensemble.demo --chart songs/foo.chart
    python -m ensemble.demo --mode real_time          # paces to tempo, prints as it goes
"""

import argparse
import time
from pathlib import Path

from ensemble import MACHINE_SPEED, REAL_TIME, Session, Voice, chord_tone_generator
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent.parent / "songs" / "blues_in_f.chart"
SAX_REGISTER = (55, 79)


def note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--mode", choices=[MACHINE_SPEED, REAL_TIME], default=MACHINE_SPEED)
    args = parser.parse_args()

    song = parse_chart(args.chart.read_text())
    print(f"{song.title}  ({song.tempo_bpm:.0f} bpm, {song.feel}, {song.total_beats:.0f} beats)")

    sax = Voice(
        id="sax",
        instrument="sax",
        register=SAX_REGISTER,
        source="ai",
        generator=chord_tone_generator(SAX_REGISTER),
    )
    session = Session(song=song, voices=[sax])

    started = time.time()
    timeline = session.generate(mode=args.mode)
    elapsed = time.time() - started

    for event in timeline:
        chord = song.chord_at(event.start_beat)
        print(
            f"beat {event.start_beat:6.1f}  {chord!s:>6}  "
            f"{event.voice_id:>4}  {note_name(event.pitch):>4}  vel={event.velocity}"
        )

    print(f"\n{len(timeline)} events, {elapsed:.2f}s wall-clock ({args.mode})")


if __name__ == "__main__":
    main()
