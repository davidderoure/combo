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
    python self_test.py --persist             # RehearsalMemory also survives THIS
                                                # process exiting -- run again later
                                                # (same or different day) and it picks
                                                # up where it left off (Phase 26,
                                                # rehearsal_memory/<chart>.json)
    python self_test.py --corpus               # sax selection also consults the WJD
                                                # corpus-familiarity check (Phase 29),
                                                # but only on chunks already pushed off
                                                # the model's natural distribution by a
                                                # recalled motif or modal_strength --
                                                # needs `python wjd_corpus.py --build`
                                                # to have been run first
    python self_test.py --respond-to-self       # sax also seeds each chunk from its
                                                # own recent notes, alongside the bass
                                                # (Phase 37) -- mirrors Wolfson's own
                                                # self-play ("the sax continuously
                                                # responds to itself")
    python self_test.py --lay-out-for-cues      # sax sometimes waits through genuine
                                                # silence for the next real chord
                                                # change instead of resuming right
                                                # away (Phase 43)

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
from typing import List, Optional, Tuple

from config import MIDI_OUTPUT_PORT
from ensemble import MACHINE_SPEED, NoteEvent, Session, Voice, drum_generator
from ensemble.comping import comping_generator
from ensemble.corpus_motifs import CorpusMotifs
from ensemble.generators import place_in_register
from ensemble.memory import RehearsalMemory
from ensemble.roles import default_accompanist_roles
from ensemble.sax import sax_generator
from ensemble.timeline import BEATS_PER_BAR
from ensemble.voice import Generator
from output.midi_output import list_output_ports, play_timeline
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent / "songs" / "blues_in_f.chart"
REHEARSAL_MEMORY_DIR = Path(__file__).resolve().parent / "rehearsal_memory"  # Phase 26 -- gitignored,
                                                                               # personal practice data
WJD_CACHE_PATH = Path(__file__).resolve().parent / "wjd_data" / "wjd_motifs.json"  # Phase 29 -- gitignored,
                                                                                     # built by wjd_corpus.py --build
BASS_REGISTER = (28, 52)
SAX_REGISTER = (55, 79)
KEYS_REGISTER = (48, 72)
GUITAR_REGISTER = (52, 76)  # overlaps KEYS_REGISTER -- exercises Phase 15's role split
DRUM_REGISTER = (35, 59)  # not musically meaningful for percussion, kept for Voice's shape
SAX_WEIGHTS_PATH = Path(__file__).resolve().parent / "ensemble" / "wolfson" / "models" / "sax_best.pt"
LAY_OUT_FOR_CUE_PROBABILITY = 0.3  # Phase 43 -- a real chosen placeholder,
                                     # same honest status as every other
                                     # hand-picked constant here; the actual
                                     # per-chunk chance of laying out when
                                     # --lay-out-for-cues is passed.

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


def build_voices(
    memory: RehearsalMemory,
    credit_resolved_tension: bool = False,
    disable_singability: bool = False,
    corpus: Optional[CorpusMotifs] = None,
    respond_to_self: bool = False,
    lay_out_for_cues: bool = False,
):
    """Returns (voices, sax_gen) -- sax_gen is the bare generator closure behind
    the sax Voice (or None if sax_best.pt isn't present), kept separately so
    main() can read its motif_adherence_log after each loop (Phase 17) the same
    way tests reach into a generator's exposed state -- Voice itself doesn't
    expose anything beyond the closure it wraps.

    credit_resolved_tension (Phase 22): off by default -- a "beginner" default,
    matching sax_generator's own. Pass --credit-resolved-tension to hear the
    "advanced" behaviour instead: a deliberate, resolved tension (e.g. a b9
    resolving by step to the root) survives selection rather than being
    avoided outright.

    disable_singability: Phase 13's toggle_singability was built as a live
    director-gesture switch (no CLI equivalent existed before this) -- reaches
    into sax_gen.critic_weights directly, the same exposed-mutable-state
    convention tests already use, since there's no dedicated setter.

    corpus (Phase 29): passed straight through to sax_generator -- only
    affects selection on chunks already pushed off the model's natural
    distribution (a recalled motif or a modal chart), see sax_generator's own
    docstring. None (default) is exactly today's behaviour.

    respond_to_self (Phase 37): off by default -- today's bass-only seeding
    stays the default sound. Pass --respond-to-self to also seed each chunk
    from the sax's own recent notes, alongside the bass, mirroring Wolfson's
    own self-play ("the sax continuously responds to itself") -- see
    sax_generator's own docstring for the full reasoning.

    lay_out_for_cues (Phase 43): off by default -- today's fixed small rest
    between phrases stays the default. Pass --lay-out-for-cues to let the sax
    sometimes wait through genuine silence for the next real chord change
    (LAY_OUT_FOR_CUE_PROBABILITY, a real chosen placeholder) instead of
    resuming right away -- see sax_generator's own docstring for the full
    reasoning."""
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
            SAX_REGISTER, target_voice_id="bass", memory=memory, n_candidates=8, motif_recall_candidates=20,
            credit_resolved_tension=credit_resolved_tension, corpus=corpus,
            own_voice_id="sax" if respond_to_self else None,
            lay_out_for_cue_probability=LAY_OUT_FOR_CUE_PROBABILITY if lay_out_for_cues else 0.0,
        )
        if disable_singability:
            sax_gen.critic_weights["singability"] = 0.0
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
    parser.add_argument("--credit-resolved-tension", action="store_true",
                         help="Phase 22: 'advanced' mode -- a deliberate, resolved tension "
                              "(approached from an in-scale note, resolved by step onto a "
                              "chord tone) survives selection instead of being avoided.")
    parser.add_argument("--no-singability", action="store_true",
                         help="Zero out the singability weight (Phase 13's toggle_singability, "
                              "here exposed as a flag instead of only a live director gesture) -- "
                              "don't mark down fast/exploratory playing for being unsustained.")
    parser.add_argument("--persist", action="store_true",
                         help="Phase 26: RehearsalMemory persists to disk, keyed by chart "
                              "(rehearsal_memory/<chart>.json), so separate runs on separate days "
                              "build on each other -- not just --loop within one run. Off by "
                              "default: a plain run touches nothing on disk.")
    parser.add_argument("--corpus", action="store_true",
                         help="Phase 29: sax selection also consults a WJD corpus-familiarity "
                              "check, but only on chunks already pushed off the model's natural "
                              "distribution (a recalled motif or a modal chart) -- see "
                              "ensemble/sax.py's sax_generator docstring. Needs "
                              "`python wjd_corpus.py --build` to have been run first.")
    parser.add_argument("--respond-to-self", action="store_true",
                         help="Phase 37: also seed each chunk from the sax's own recent notes, "
                              "alongside the bass -- mirroring Wolfson's own self-play ('the sax "
                              "continuously responds to itself'). Off by default: today's "
                              "bass-only seeding stays the default sound.")
    parser.add_argument("--lay-out-for-cues", action="store_true",
                         help="Phase 43: sometimes wait through genuine silence for the next real "
                              "chord change instead of resuming right away with a fixed small "
                              "rest. Off by default: today's fixed Phase 39 rest stays the "
                              "default sound.")
    args = parser.parse_args()

    if args.list_out:
        for i, name in enumerate(list_output_ports()):
            print(f"{i}: {name}")
        return

    song = parse_chart(args.chart.read_text())
    persist_path = REHEARSAL_MEMORY_DIR / f"{args.chart.stem}.json" if args.persist else None
    memory = RehearsalMemory(persist_path=persist_path)

    corpus = None
    if args.corpus:
        if WJD_CACHE_PATH.exists():
            corpus = CorpusMotifs(WJD_CACHE_PATH)
        else:
            print(f"{WJD_CACHE_PATH} not found -- run 'python wjd_corpus.py --build' first. Continuing without it.")

    for i in range(args.loop):
        label = f" (rehearsal {i + 1}/{args.loop})" if args.loop > 1 else ""
        print(f"\nGenerating{label}...")
        voices, sax_gen = build_voices(
            memory, credit_resolved_tension=args.credit_resolved_tension, disable_singability=args.no_singability,
            corpus=corpus, respond_to_self=args.respond_to_self, lay_out_for_cues=args.lay_out_for_cues,
        )
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
