"""Manual sanity check for the ensemble skeleton (DESIGN.md §2/§4) — no MIDI needed.

    python -m ensemble.demo                          # blues_in_f.chart, machine speed
    python -m ensemble.demo --chart songs/foo.chart
    python -m ensemble.demo --mode real_time          # paces to tempo, prints as it goes
"""

import argparse
import time
from pathlib import Path

from ensemble import MACHINE_SPEED, REAL_TIME, Session, Voice, chord_tone_generator, drum_generator
from ensemble.comping import BUSY_THRESHOLD, INTENSITY_SPREAD, SPARSE_THRESHOLD, comping_generator
from ensemble.director import Director, constant_director_source, ensemble_intensity_critic
from ensemble.drums import ACOUSTIC_SNARE, CLOSED_HI_HAT, RIDE_CYMBAL_1
from ensemble.listening import density as listening_density
from ensemble.listening import synthetic_varying_density_generator
from ensemble.sax import sax_generator
from ensemble.timeline import BEATS_PER_BAR, Timeline
from ensemble.transitions import TransitionController, scripted_gesture_source
from gesture.vocabulary import Gesture
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent.parent / "songs" / "blues_in_f.chart"
SAX_REGISTER = (55, 79)
DRUM_REGISTER = (35, 59)  # not musically meaningful for percussion, kept for Voice's shape
KEYS_REGISTER = (48, 72)
BASS_REGISTER = (28, 52)
COMPING_LOOKBACK_BARS = 2
SAX_WEIGHTS_PATH = Path(__file__).resolve().parent / "wolfson" / "models" / "sax_best.pt"

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


def _comping_response(target_density: float, intensity: float) -> str:
    """Mirrors ensemble/comping.py's threshold logic exactly (same formula), so this
    display classification can never drift from what comping actually decided —
    same reasoning as demo_comping()'s density-based classification above."""
    shift = (intensity - 0.5) * INTENSITY_SPREAD
    if target_density >= BUSY_THRESHOLD + shift:
        return "duck"
    if target_density <= SPARSE_THRESHOLD + shift:
        return "fill"
    return "moderate"


def demo_director(chart_path: Path) -> None:
    print("\n--- Musical director demo (DESIGN.md §11) ---")
    print("Dial channel only: the gesture channel (DirectorSignal.gesture,")
    print("aggregate_director_signals) has a tested data model but isn't wired to")
    print("anything that acts on it -- §8's handover triggers now exist as code (see")
    print("the transitions demo below), but they consume gestures from a Session's")
    print("gesture_source, not from DirectorSignal.gesture, so a director-emitted")
    print("handover() still has nowhere to act (see ensemble/director.py's docstring).\n")

    song = parse_chart(chart_path.read_text())

    def run_with_intensity(intensity: float) -> Timeline:
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
        director = Director(id="d", source="ai", signal_source=constant_director_source(intensity))
        session = Session(song=song, voices=[soloist, comper], directors=[director])
        return session.generate(mode=MACHINE_SPEED)

    # Same soloist fixture/seed in both runs, so the density column is identical --
    # only the director's intensity differs, isolating exactly what it changes.
    low_timeline = run_with_intensity(0.0)
    high_timeline = run_with_intensity(1.0)

    print("  same soloist fixture, low (0.0) vs. high (1.0) director intensity:")
    for bar_index in range(12):
        since = max(0, bar_index - COMPING_LOOKBACK_BARS) * BEATS_PER_BAR
        until = bar_index * BEATS_PER_BAR
        d = listening_density(low_timeline, "soloist", since, until)
        low_response = _comping_response(d, 0.0)
        high_response = _comping_response(d, 1.0)
        print(f"    bar {bar_index:2d}: density {d:.2f}  ->  low={low_response:<8}  high={high_response}")

    print("\n  ensemble_intensity_critic (a real 'AI critic', DESIGN.md §11) reading")
    print("  the sax+drums session from the first demo above:")
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=chord_tone_generator(SAX_REGISTER))
    drums = Voice(id="drums", instrument="drums", register=DRUM_REGISTER, source="ai", generator=drum_generator(seed=42))
    critic_timeline = Session(song=song, voices=[sax, drums]).generate(mode=MACHINE_SPEED)
    critic = ensemble_intensity_critic(voice_ids=["sax", "drums"], lookback_bars=COMPING_LOOKBACK_BARS)
    for bar_index in (0, 5, 11):
        prior = Timeline([e for e in critic_timeline if e.start_beat < bar_index * BEATS_PER_BAR])
        signal = critic(song, bar_index, prior)
        print(f"    bar {bar_index:2d}: critic-derived intensity = {signal.intensity:.2f}")


def demo_transitions(chart_path: Path) -> None:
    print("\n--- Handover/transition triggers demo (DESIGN.md §8) ---")
    print("A scripted handover() at bar 24 (partway through blues_in_f.chart's Solos")
    print("x3) should shorten Solos to end after its current chorus -- moving later")
    print("section boundaries 12 bars earlier than the nominal bar-count scaffold.\n")

    song = parse_chart(chart_path.read_text())
    transitions = TransitionController()
    transitions.on_gesture(Gesture("handover"), song, 24 * BEATS_PER_BAR)
    effective = transitions.effective_song(song)

    print("  bar  nominal section          effective section (after handover)")
    for bar_index in (0, 11, 12, 23, 24, 35, 36, 47, 48, 59):
        beat = bar_index * BEATS_PER_BAR
        nominal_section, nominal_chorus = song.section_at(beat)
        effective_section, effective_chorus = effective.section_at(beat)
        print(
            f"  {bar_index:3d}  {nominal_section.name} (chorus {nominal_chorus})".ljust(35)
            + f"{effective_section.name} (chorus {effective_chorus})"
        )

    print("\n  Same shift reaching a real consumer -- drums' section-aware density")
    print("  (§7) at bar 36, with vs. without the scripted handover:")
    drums_voice = Voice(id="drums", instrument="drums", register=DRUM_REGISTER, source="ai", generator=drum_generator(seed=1))
    without = Session(song=song, voices=[drums_voice]).generate(mode=MACHINE_SPEED)
    with_handover = Session(
        song=song, voices=[drums_voice], gesture_source=scripted_gesture_source({24: [Gesture("handover")]})
    ).generate(mode=MACHINE_SPEED)
    for label, timeline in (("without handover", without), ("with handover", with_handover)):
        pitches = sorted({e.pitch for e in timeline if 36 * BEATS_PER_BAR <= e.start_beat < 37 * BEATS_PER_BAR})
        names = [GM_PERCUSSION_NAMES.get(p, str(p)) for p in pitches]
        print(f"    bar 36, {label}: {names}")


def demo_sax_wolfson(chart_path: Path) -> None:
    print("\n--- Real generation: sax voice (Wolfson-adapted LSTM, DESIGN.md §12) ---")
    if not SAX_WEIGHTS_PATH.exists():
        print(f"  sax_best.pt not found at {SAX_WEIGHTS_PATH}")
        print("  Copy it in from ~/wolfson/models/sax_best.pt to run this section (see README).")
        print("  Skipping — the other four demo sections above don't need it.")
        return

    print("Bass is chord_tone_generator (a real instrument's melodic line, not a")
    print(f"synthetic fixture, unlike demo_comping's soloist) — its notes over the")
    print(f"previous {COMPING_LOOKBACK_BARS} bars become the sax's seed_phrase. Bar 0 has an empty")
    print("seed (nothing played yet) -- chord-conditioned generation only, printed")
    print("below to show that's a real, handled case, not a crash. All ~12 of the")
    print("model's rule-based bias knobs (energy arc, motif, register contrast, ...)")
    print("are left at their defaults -- see ensemble/sax.py's module docstring.\n")

    song = parse_chart(chart_path.read_text())
    bass = Voice(
        id="bass",
        instrument="bass",
        register=BASS_REGISTER,
        source="ai",
        generator=chord_tone_generator(BASS_REGISTER),
    )
    sax = Voice(
        id="sax",
        instrument="sax",
        register=SAX_REGISTER,
        source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=7),
    )
    session = Session(song=song, voices=[bass, sax])
    timeline = session.generate(mode=MACHINE_SPEED)

    for event in timeline:
        if event.start_beat >= 3 * BEATS_PER_BAR:  # first three bars are enough to show it working
            break
        print(f"beat {event.start_beat:6.1f}  {event.voice_id:>4}  {note_name(event.pitch):>4}  vel={event.velocity}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--mode", choices=[MACHINE_SPEED, REAL_TIME], default=MACHINE_SPEED)
    args = parser.parse_args()

    demo_sax_and_drums(args.chart, args.mode)
    demo_comping(args.chart)
    demo_director(args.chart)
    demo_transitions(args.chart)
    demo_sax_wolfson(args.chart)


if __name__ == "__main__":
    main()
