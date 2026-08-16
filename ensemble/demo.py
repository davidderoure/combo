"""Manual sanity check for the ensemble skeleton (DESIGN.md §2/§4) — no MIDI needed.

    python -m ensemble.demo                          # blues_in_f.chart, machine speed
    python -m ensemble.demo --chart songs/foo.chart
    python -m ensemble.demo --mode real_time          # paces to tempo, prints as it goes
"""

import argparse
import time
from pathlib import Path

from ensemble import MACHINE_SPEED, REAL_TIME, Session, Voice, chord_tone_generator, drum_generator
from ensemble.comping import BUSY_THRESHOLD, SPARSE_THRESHOLD, comping_generator
from ensemble.drums import ACOUSTIC_SNARE, CLOSED_HI_HAT, RIDE_CYMBAL_1
from ensemble.listening import density as listening_density
from ensemble.listening import synthetic_varying_density_generator
from ensemble.timeline import BEATS_PER_BAR
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent.parent / "songs" / "blues_in_f.chart"
SAX_REGISTER = (55, 79)
DRUM_REGISTER = (35, 59)  # not musically meaningful for percussion, kept for Voice's shape
KEYS_REGISTER = (48, 72)
COMPING_LOOKBACK_BARS = 2

GM_PERCUSSION_NAMES = {
    ACOUSTIC_SNARE: "Snare",
    CLOSED_HI_HAT: "HiHat",
    RIDE_CYMBAL_1: "Ride",
}


def note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def event_label(event) -> str:
    if event.voice_id == "drums":
        return GM_PERCUSSION_NAMES.get(event.pitch, str(event.pitch))
    return note_name(event.pitch)


def demo_sax_and_drums(chart_path: Path, mode: str) -> None:
    song = parse_chart(chart_path.read_text())
    print(f"{song.title}  ({song.tempo_bpm:.0f} bpm, {song.feel}, {song.total_beats:.0f} beats)")

    sax = Voice(
        id="sax",
        instrument="sax",
        register=SAX_REGISTER,
        source="ai",
        generator=chord_tone_generator(SAX_REGISTER),
    )
    drums = Voice(
        id="drums",
        instrument="drums",
        register=DRUM_REGISTER,
        source="ai",
        generator=drum_generator(seed=42),
    )
    session = Session(song=song, voices=[sax, drums])

    started = time.time()
    timeline = session.generate(mode=mode)
    elapsed = time.time() - started

    for event in timeline:
        chord = song.chord_at(event.start_beat)
        print(
            f"beat {event.start_beat:6.1f}  {chord!s:>6}  "
            f"{event.voice_id:>5}  {event_label(event):>5}  vel={event.velocity}"
        )

    print(f"\n{len(timeline)} events, {elapsed:.2f}s wall-clock ({mode})")


def demo_comping(chart_path: Path) -> None:
    print("\n--- Accompaniment-listening demo (DESIGN.md §5) ---")
    print("Using a synthetic varying-density fixture as a stand-in soloist, not the")
    print("sax stub above: chord_tone_generator plays a constant 4 notes every bar,")
    print("so there'd be nothing for comping to react to. Comping's response for bar")
    print(f"N is based on the soloist's density over bars [N-{COMPING_LOOKBACK_BARS}, N)")
    print("— the prior bars, not bar N itself — so that's what's printed below.\n")

    song = parse_chart(chart_path.read_text())
    soloist = Voice(
        id="soloist",
        instrument="test-fixture",
        register=SAX_REGISTER,
        source="ai",
        generator=synthetic_varying_density_generator(seed=1),
    )
    comper = Voice(
        id="keys",
        instrument="keys",
        register=KEYS_REGISTER,
        source="ai",
        generator=comping_generator(
            KEYS_REGISTER, target_voice_id="soloist", lookback_bars=COMPING_LOOKBACK_BARS, seed=2
        ),
    )
    session = Session(song=song, voices=[soloist, comper])
    timeline = session.generate(mode=MACHINE_SPEED)

    # Classify using the *same* listening.density() call comping_generator makes
    # internally, over the identical window — not by re-counting raw events with a
    # bar-boundary cutoff, which a humanised note can jitter across and make the
    # printed classification disagree with what comping actually decided.
    for bar_index in range(12):
        since = max(0, bar_index - COMPING_LOOKBACK_BARS) * BEATS_PER_BAR
        until = bar_index * BEATS_PER_BAR
        d = listening_density(timeline, "soloist", since, until)
        if d >= BUSY_THRESHOLD:
            response = "duck"
        elif d <= SPARSE_THRESHOLD:
            response = "fill"
        else:
            response = "moderate"
        comping_notes = sum(1 for e in timeline if e.voice_id == "keys" and until <= e.start_beat < until + BEATS_PER_BAR)
        print(f"  bar {bar_index:2d}: soloist density {d:.2f} notes/beat -> comping {response} ({comping_notes} notes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--mode", choices=[MACHINE_SPEED, REAL_TIME], default=MACHINE_SPEED)
    args = parser.parse_args()

    demo_sax_and_drums(args.chart, args.mode)
    demo_comping(args.chart)


if __name__ == "__main__":
    main()
