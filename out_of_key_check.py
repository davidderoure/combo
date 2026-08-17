"""How many of the sax's generated notes fall outside the active chord's
scale, under self_test.py's actual settings -- so this reflects what's really
being heard, not a different configuration. The same in/out-of-key check
ensemble/critic.py's tonal_conformity/dissonance already use internally,
applied per-note here across a whole run rather than as one blended
per-chunk fraction -- a direct audit of the final output, not just a trust
that per-chunk selection worked.

    python out_of_key_check.py                    # 5 loops, blues_in_f.chart
    python out_of_key_check.py --loops 10
    python out_of_key_check.py --n-candidates 1    # before Phase 18's fix, for comparison

Not "solos deliberately out" -- David asked specifically about ordinary
melodic/singable soloing, so this uses self_test.py's sax_generator
construction (memory, motif_recall_candidates), no director/gesture pushing
toward outside playing.

First run (5 loops, n_candidates=3, no dissonance-avoidance yet): ~16-22% of
sax notes out of key, and -- checked, not assumed -- every single one landed
exactly 1 semitone from the scale (the "minor 9th" clash David identified as
the worst case in ordinary melodic playing, never further away). That finding
directly shaped Phase 18's fix: ensemble/critic.py's new `dissonance` metric
specifically targets this exact 1-semitone case, and ensemble/sax.py's
selection now checks it FIRST, ahead of motif_adherence and overall quality
("what's bad matters a lot," not just one more positively-weighted ingredient
in a blend). After the fix (n_candidates=8, the same setting self_test.py now
uses by default): 2.1% out of key, reproduced across two separate 5-loop runs
(31/1505 and 32/1525) -- a real, repeatable drop from ~1-in-5 to ~1-in-50,
not a one-off. The remaining out-of-key notes are still all exactly 1
semitone away -- dissonance-avoidance reduces the RATE by giving selection
more/better candidates to choose among, it doesn't guarantee zero (the model
can still generate all-clashing candidates in a batch, just increasingly
rarely as n_candidates grows).
"""

import argparse
from collections import Counter
from pathlib import Path

from ensemble import MACHINE_SPEED, Session, Voice
from ensemble.memory import RehearsalMemory
from ensemble.sax import chord_to_wolfson_index, sax_generator
from ensemble.wolfson.scales import chord_root, chord_to_mode, scale_pitch_classes
from song import parse_chart

import self_test as st


def note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def semitone_distance_to_scale(pitch_class: int, scale: frozenset) -> int:
    return min((pitch_class - s) % 12 if (pitch_class - s) % 12 <= 6 else 12 - (pitch_class - s) % 12 for s in scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=st.DEFAULT_CHART)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--motif-recall-candidates", type=int, default=20)
    args = parser.parse_args()

    song = parse_chart(args.chart.read_text())
    memory = RehearsalMemory()

    total_notes = 0
    out_of_key = 0
    distance_counter = Counter()
    examples = []

    for i in range(args.loops):
        bass = Voice(id="bass", instrument="bass", register=st.BASS_REGISTER, source="ai",
                     generator=st.walking_bass_stub(st.BASS_REGISTER))
        sax_gen = sax_generator(st.SAX_REGISTER, target_voice_id="bass", memory=memory,
                                 n_candidates=args.n_candidates, motif_recall_candidates=args.motif_recall_candidates)
        sax = Voice(id="sax", instrument="sax", register=st.SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)

        for event in timeline:
            if event.voice_id != "sax":
                continue
            chord = song.chord_at(event.start_beat)
            chord_idx = chord_to_wolfson_index(chord)
            scale = scale_pitch_classes(chord_root(chord_idx), chord_to_mode(chord_idx))
            pitch_class = event.pitch % 12
            total_notes += 1
            if pitch_class not in scale:
                out_of_key += 1
                dist = semitone_distance_to_scale(pitch_class, scale)
                distance_counter[dist] += 1
                if len(examples) < 15:
                    examples.append((chord, note_name(event.pitch), dist))

    print(f"Sax notes generated: {total_notes} ({args.loops} loops, {args.chart.name}, "
          f"n_candidates={args.n_candidates}, motif_recall_candidates={args.motif_recall_candidates})")
    print(f"Out-of-key notes: {out_of_key} ({100 * out_of_key / total_notes:.1f}%)")
    print("\nOut-of-key notes by semitone distance to the nearest in-scale pitch class:")
    for dist in sorted(distance_counter):
        print(f"  {dist} semitone{'s' if dist != 1 else ''} away: {distance_counter[dist]}")
    print("\nExamples (chord, note played, semitones from nearest in-scale tone):")
    for chord, name, dist in examples:
        print(f"  {chord!s:>6}  {name:>4}  {dist} semitone{'s' if dist != 1 else ''}")


if __name__ == "__main__":
    main()
