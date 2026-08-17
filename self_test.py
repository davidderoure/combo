"""Runs a full AI-only ensemble over a chart and plays it back through a real
MIDI output port — no MIDI input/hardware performer needed (a "self-test",
mirroring wolfson's own AI-vs-AI self-play mode, generalised here to combo's
whole ensemble rather than one bass+sax pair).

    python self_test.py --list-out           # show available MIDI output ports
    python self_test.py                       # play blues_in_f.chart once
    python self_test.py --chart songs/x.chart --out 2
    python self_test.py --loop 3              # play 3 times; sax's RehearsalMemory
                                                # carries across loops (DESIGN.md's
                                                # rehearsal idea, Phase 11 — now
                                                # audible instead of only tested)

Generation is machine_speed (instant) — DESIGN.md §4's "generation produces a
symbolic timeline, playback/scheduling is a separate stage" architecture is
literally what this script is built on: real-time pacing happens entirely in
output/midi_output.py's play_timeline, not in Session.generate.

This is step 1 of a three-step testing plan (see README's Running section):
step 2 is `listen.py` (already built) for sanity-checking a MIDI input device;
step 3 — a human performer's live input actually driving the ensemble's live
response — needs a "live performance driver" that doesn't exist yet.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from config import MIDI_OUTPUT_PORT
from ensemble import MACHINE_SPEED, NoteEvent, Session, Voice, drum_generator
from ensemble.comping import comping_generator
from ensemble.generators import place_in_register
from ensemble.memory import RehearsalMemory
from ensemble.roles import default_accompanist_roles
from ensemble.sax import sax_generator
from ensemble.timeline import BEATS_PER_BAR
from ensemble.voice import Generator
from output.midi_output import list_output_ports, play_timeline
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent / "songs" / "blues_in_f.chart"
BASS_REGISTER = (28, 52)
SAX_REGISTER = (55, 79)
KEYS_REGISTER = (48, 72)
GUITAR_REGISTER = (52, 76)  # overlaps KEYS_REGISTER -- exercises Phase 15's role split
DRUM_REGISTER = (35, 59)  # not musically meaningful for percussion, kept for Voice's shape
SAX_WEIGHTS_PATH = Path(__file__).resolve().parent / "ensemble" / "wolfson" / "models" / "sax_best.pt"

# GM channel 10 (1-indexed) for drums -- drums.py's note constants are already
# real GM percussion-map numbers. A plain dict, not a new field on Voice: a
# MIDI channel is a rendering concern, not a generation one.
CHANNELS = {"bass": 1, "sax": 2, "keys": 3, "guitar": 4, "drums": 10}

WALKING_BASS_VELOCITY = 75
# Just under a full beat, not a placeholder chosen by ear against a specific
# patch: leaves a hair of articulation between notes rather than an exact
# back-to-back note-off/note-on, still ringing through almost the whole beat.
WALKING_BASS_NOTE_DURATION = 0.92


def walking_bass_stub(register: Tuple[int, int]) -> Generator:
    """self_test.py's own bass stand-in -- still not real bass generation (that's
    an entirely unbuilt voice, like sax was before Phase 8), but a better fit for
    being actually heard than ensemble/generators.py's chord_tone_generator,
    which self_test.py used originally. Heard directly (not assumed): a real bass
    sample playing chord_tone_generator's simultaneous root+fifth double-stop,
    twice a bar with a full beat of silence after each hit, sounded thuddy and
    staccato -- the double-stop and the gaps, not note duration alone. This plays
    a single note every beat (quarter notes, not a double-stop), alternating
    root/fifth, sustained for nearly the full beat so it rings continuously."""

    def generate(song, bar_index: int, timeline, director_signal) -> List[NoteEvent]:
        events: List[NoteEvent] = []
        for beat_offset in (0.0, 1.0, 2.0, 3.0):
            beat = bar_index * BEATS_PER_BAR + beat_offset
            chord = song.chord_at(beat)
            pitch_class = chord.root if beat_offset in (0.0, 2.0) else (chord.root + 7) % 12
            pitch = place_in_register(pitch_class, register)
            events.append(
                NoteEvent(
                    voice_id="",
                    pitch=pitch,
                    velocity=WALKING_BASS_VELOCITY,
                    start_beat=beat,
                    duration_beats=WALKING_BASS_NOTE_DURATION,
                )
            )
        return events

    return generate


def build_voices(memory: RehearsalMemory):
    """Returns (voices, sax_gen) -- sax_gen is the bare generator closure behind
    the sax Voice (or None if sax_best.pt isn't present), kept separately so
    main() can read its motif_adherence_log after each loop (Phase 17) the same
    way tests reach into a generator's exposed state -- Voice itself doesn't
    expose anything beyond the closure it wraps."""
    bass = Voice(
        id="bass",
        instrument="bass (walking-bass stub -- real bass generation isn't built yet)",
        register=BASS_REGISTER,
        source="ai",
        generator=walking_bass_stub(BASS_REGISTER),
    )
    voices = [bass]

    has_sax = SAX_WEIGHTS_PATH.exists()
    sax_gen = None
    if has_sax:
        # n_candidates=8 (up from 3): dissonance-avoidance (Phase 18) is only as
        # good as what it has to choose among -- more candidates raise the odds
        # a truly clash-free one exists in the batch. motif_recall_candidates=20
        # fires on most chunks after the first once memory has anything stored
        # (within-run persistence, Phase 11), not just a rare one-off -- but even
        # paid on nearly every chunk the absolute cost stays fine (Phase 14:
        # ~164ms/chunk for 20 candidates), since it's all spent up front during
        # machine_speed generation before playback starts.
        sax_gen = sax_generator(
            SAX_REGISTER, target_voice_id="bass", memory=memory, n_candidates=8, motif_recall_candidates=20
        )
        voices.append(Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen))
    else:
        print(
            "sax_best.pt not found -- sax voice skipped (see README). Copy it into "
            "ensemble/wolfson/models/ to include the sax voice."
        )
    comping_target = "sax" if has_sax else "bass"

    roles = default_accompanist_roles([("keys", KEYS_REGISTER), ("guitar", GUITAR_REGISTER)])
    voices.append(
        Voice(
            id="keys",
            instrument="keys",
            register=KEYS_REGISTER,
            source="ai",
            generator=comping_generator(KEYS_REGISTER, target_voice_id=comping_target, seed=2, lay_out=not roles["keys"]),
        )
    )
    voices.append(
        Voice(
            id="guitar",
            instrument="guitar",
            register=GUITAR_REGISTER,
            source="ai",
            generator=comping_generator(
                GUITAR_REGISTER, target_voice_id=comping_target, seed=3, lay_out=not roles["guitar"]
            ),
        )
    )
    voices.append(
        Voice(id="drums", instrument="drums", register=DRUM_REGISTER, source="ai", generator=drum_generator(seed=4))
    )
    return voices, sax_gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--out", type=int, default=MIDI_OUTPUT_PORT)
    parser.add_argument("--list-out", action="store_true")
    parser.add_argument("--loop", type=int, default=1)
    args = parser.parse_args()

    if args.list_out:
        for i, name in enumerate(list_output_ports()):
            print(f"{i}: {name}")
        return

    song = parse_chart(args.chart.read_text())
    memory = RehearsalMemory()

    for i in range(args.loop):
        label = f" (rehearsal {i + 1}/{args.loop})" if args.loop > 1 else ""
        print(f"\nGenerating{label}...")
        voices, sax_gen = build_voices(memory)
        session = Session(song=song, voices=voices)
        timeline = session.generate(mode=MACHINE_SPEED)
        print(
            f"Playing {len(timeline)} notes over {song.total_beats:.0f} beats "
            f"at {song.tempo_bpm:.0f} bpm{label} (Ctrl-C to stop)..."
        )
        if sax_gen is not None and any(a > 0 for a in sax_gen.motif_adherence_log):
            print("  -> sax echoed a motif from an earlier rehearsal")
        try:
            play_timeline(timeline, song.tempo_bpm, CHANNELS, args.out)
        except KeyboardInterrupt:
            print("\nStopped.")
            return


if __name__ == "__main__":
    main()
