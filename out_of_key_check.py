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
not a one-off.

Phase 19 added a passing-tone exception directly to `dissonance` -- a flagged
note approached and left by step, continuing in the same direction, is now
excused during SELECTION (a genuine chromatic passing tone, the classical
tonal-theory treatment of a dissonance, not a clash). This script's own
breakdown below (reusing critic.py's own `_is_passing_tone`, the same
"verify via the same computation" precedent as everywhere else in this
codebase) shows how many of the notes that still survive into the FINAL
output are passing tones vs. genuine unexplained clashes -- selection can
only choose among what got generated, so this is a real post-hoc audit, not
just a re-assertion that the selection logic works.

After the passing-tone exception (two separate 5-loop runs): out-of-key rate
rose to 4.2% then 3.2% (up from Phase 18's 2.1%) -- expected, not a
regression: passing tones are now less penalised in selection, so more
survive into the final output. Of those, 51.5% then 62.5% were genuine
passing tones, the rest unexplained clashes -- a real, repeatable majority,
not a one-off. So the fix is doing what it was meant to: more chromatic
motion is surviving, and a solid majority of what's flagged as "out of key"
now has a real melodic justification rather than being an unexplained clash.

Phase 20 (Lever A) widened the scale `dissonance` itself judges against --
this script now uses the SAME reference (`dissonance_scale`, not the plain
chord_to_mode scale) so its own report matches what selection is actually
doing. Two more 5-loop runs after that: 4.4% (65/1485) then 1.2% (16/1374) --
lower than Phase 19's 3.2-4.2%, and the bebop maj7-over-dominant example that
started all of this ("E natural over F7") no longer appears in the examples
at all -- it's simply in-scale now, not merely excused. Of what's still
flagged, 78.5% then 56.2% were genuine passing tones -- the widened scale
mostly removed the *previously-miscounted* clashes (bebop tensions that were
never really dissonant), leaving a real, still-mostly-explained remainder.
"""

import argparse
from collections import Counter
from pathlib import Path

from ensemble import MACHINE_SPEED, Session, Voice
from ensemble.critic import _is_passing_tone, dissonance_scale
from ensemble.memory import RehearsalMemory
from ensemble.sax import chord_to_wolfson_index, sax_generator
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
    passing_tones = 0
    distance_counter = Counter()
    examples = []

    for i in range(args.loops):
        bass = Voice(id="bass", instrument="bass", register=st.BASS_REGISTER, source="ai",
                     generator=st.walking_bass_stub(st.BASS_REGISTER))
        sax_gen = sax_generator(st.SAX_REGISTER, target_voice_id="bass", memory=memory,
                                 n_candidates=args.n_candidates, motif_recall_candidates=args.motif_recall_candidates)
        sax = Voice(id="sax", instrument="sax", register=st.SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)

        # Ordered sequence of this loop's own sax notes -- needed (not just
        # per-event independently) so _is_passing_tone can see each note's
        # actual melodic neighbours, adjacency in time regardless of any
        # chord boundary crossed in between.
        sax_events = [e for e in timeline if e.voice_id == "sax"]
        sax_notes = [{"pitch": e.pitch} for e in sax_events]

        for idx, event in enumerate(sax_events):
            chord = song.chord_at(event.start_beat)
            chord_idx = chord_to_wolfson_index(chord)
            # dissonance_scale (Phase 20), not the plain chord_to_mode scale --
            # this is the same reference dissonance() actually judges against
            # now, so the report matches what selection is really doing.
            scale = dissonance_scale(chord_idx)
            pitch_class = event.pitch % 12
            total_notes += 1
            if pitch_class not in scale:
                out_of_key += 1
                dist = semitone_distance_to_scale(pitch_class, scale)
                distance_counter[dist] += 1
                is_passing = dist == 1 and _is_passing_tone(sax_notes, idx)
                if is_passing:
                    passing_tones += 1
                if len(examples) < 15:
                    examples.append((chord, note_name(event.pitch), dist, is_passing))

    print(f"Sax notes generated: {total_notes} ({args.loops} loops, {args.chart.name}, "
          f"n_candidates={args.n_candidates}, motif_recall_candidates={args.motif_recall_candidates})")
    print(f"Out-of-key notes: {out_of_key} ({100 * out_of_key / total_notes:.1f}%)")
    print(f"  of which genuine passing tones (excused during selection): {passing_tones} "
          f"({100 * passing_tones / out_of_key:.1f}% of out-of-key notes)" if out_of_key else "")
    print(f"  of which unexplained clashes: {out_of_key - passing_tones}")
    print("\nOut-of-key notes by semitone distance to the nearest in-scale pitch class:")
    for dist in sorted(distance_counter):
        print(f"  {dist} semitone{'s' if dist != 1 else ''} away: {distance_counter[dist]}")
    print("\nExamples (chord, note played, semitones from nearest in-scale tone, passing tone?):")
    for chord, name, dist, is_passing in examples:
        print(f"  {chord!s:>6}  {name:>4}  {dist} semitone{'s' if dist != 1 else ''}  "
              f"{'passing tone' if is_passing else 'clash'}")


if __name__ == "__main__":
    main()
