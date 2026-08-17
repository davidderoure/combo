"""A/B test: does RehearsalMemory persisting across loop iterations actually
change sax's output, measured against the same critic combo already uses
(no MIDI/audio involved -- see README's Running section for why this exists).

    python rehearsal_ab_test.py                       # 20 loops, blues_in_f.chart
    python rehearsal_ab_test.py --loops 40 --n-candidates 5
    python rehearsal_ab_test.py --chart songs/x.chart

Two conditions, differing ONLY in whether one RehearsalMemory is shared across
loop iterations or a fresh one is built each loop -- isolates cross-run
persistence specifically, holding everything else (seeds, n_candidates,
within-run behaviour) identical.

Instrumentation: spies on ensemble.sax's own bound name for musicality_score
(NOT ensemble.critic.musicality_score -- sax.py did `from .critic import
musicality_score`, so it has its own local reference; patching the critic
module's attribute wouldn't touch sax.py's already-bound name) and on
PhraseGenerator.generate to see whether a non-empty motif_target was actually
passed each chunk. Both call through to the real implementation -- this is
the same spy-not-mock technique test_sax_wolfson_integration.py and
ensemble/demo.py already use.

Primary metric: `repetition`, not `overall` -- it's the one sub-score
mechanistically tied to what memory actually does (feed a recurring motif
back in); overall also blends tonal_conformity/contour/call-response/
singability, which memory has no effect on and would only dilute any signal.

Sharpest single measurement: the FIRST chunk of each loop after loop 0. That's
the one place cross-run persistence uniquely acts -- every other chunk in a
loop already has within-run memory available in BOTH conditions (a fresh
RehearsalMemory still accumulates within a single loop's own chunks), so a
whole-run average mixes in a lot of chunks where the two conditions aren't
actually different by construction.

First real run (20 loops, n_candidates=3, blues_in_f.chart): wiring confirmed
correct (persistent condition's first chunk of every loop after loop 0 carried
a motif_target 100% of the time; control 0%, exactly as expected by
construction) but the musical effect was a genuine null result -- no
measurable difference in `repetition` between conditions, either at the first
chunk or as a trend across loops. Traced to DEFAULT_MOTIF_STRENGTH (sax.py)
already being at its 1.0 maximum, and the underlying bias in
phrase_generator.py being "2.0 logits on one specific token (sparse, fires
rarely)" per Wolfson's own code comment -- not a strength that can just be
turned up further without touching that mechanism itself.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import ensemble.sax as sax_module
from ensemble import MACHINE_SPEED, Session, Voice, chord_tone_generator
from ensemble.memory import RehearsalMemory
from ensemble.wolfson.phrase_generator import PhraseGenerator
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent / "songs" / "blues_in_f.chart"
BASS_REGISTER = (28, 52)
SAX_REGISTER = (55, 79)


@contextmanager
def spy_on_sax_generation():
    """Yields (score_records, motif_records, state). state["loop"] must be set
    by the caller before each loop iteration; state["chunk"] is maintained here,
    incremented on every chunk-build (every musicality_score call)."""
    original_score = sax_module.musicality_score
    original_generate = PhraseGenerator.generate

    score_records = []  # (loop, chunk_in_loop, MusicalityScore)
    motif_records = []  # (loop, chunk_in_loop, motif_strength)
    state = {"loop": -1, "chunk": -1}

    def score_wrapper(notes, chord_idx, seed_phrase, weights=None):
        state["chunk"] += 1
        score = original_score(notes, chord_idx, seed_phrase, weights=weights)
        score_records.append((state["loop"], state["chunk"], score))
        return score

    def generate_wrapper(self, *args, **kwargs):
        motif_targets = kwargs.get("motif_targets")
        motif_strength = kwargs.get("motif_strength", 0.0)
        motif_records.append((state["loop"], state["chunk"] + 1, motif_strength if motif_targets else 0.0))
        return original_generate(self, *args, **kwargs)

    sax_module.musicality_score = score_wrapper
    PhraseGenerator.generate = generate_wrapper
    try:
        yield score_records, motif_records, state
    finally:
        sax_module.musicality_score = original_score
        PhraseGenerator.generate = original_generate


def run_condition(chart_path: Path, persistent: bool, n_loops: int, n_candidates: int):
    with spy_on_sax_generation() as (score_records, motif_records, state):
        song = parse_chart(chart_path.read_text())
        memory = RehearsalMemory() if persistent else None
        for i in range(n_loops):
            state["loop"] = i
            state["chunk"] = -1
            if not persistent:
                memory = RehearsalMemory()
            bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai",
                         generator=chord_tone_generator(BASS_REGISTER))
            sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
                        generator=sax_module.sax_generator(
                            SAX_REGISTER, target_voice_id="bass", memory=memory, n_candidates=n_candidates
                        ))
            Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)
    return score_records, motif_records


def loop_averages(score_records, metric):
    per_loop = defaultdict(list)
    for loop_idx, _chunk_idx, score in score_records:
        per_loop[loop_idx].append(getattr(score, metric))
    loops = sorted(per_loop)
    return loops, [sum(per_loop[l]) / len(per_loop[l]) for l in loops]


def trend_slope(loops, values):
    if len(loops) < 2:
        return 0.0
    slope, _intercept = np.polyfit(loops, values, 1)
    return slope


def first_chunk_average(score_records, metric, skip_loop_0=True):
    """Average of the metric on chunk_in_loop==0, across loops -- the one
    chunk where cross-run persistence uniquely differs between conditions."""
    values = [
        getattr(score, metric)
        for loop_idx, chunk_idx, score in score_records
        if chunk_idx == 0 and not (skip_loop_0 and loop_idx == 0)
    ]
    return (sum(values) / len(values)) if values else float("nan"), len(values)


def first_chunk_motif_rate(motif_records, skip_loop_0=True):
    first_chunks = [m for (loop_idx, chunk_idx, m) in motif_records if chunk_idx == 0 and not (skip_loop_0 and loop_idx == 0)]
    if not first_chunks:
        return 0.0, 0
    return sum(1 for m in first_chunks if m > 0) / len(first_chunks), len(first_chunks)


def report(label, score_records, motif_records):
    print(f"\n=== {label} ===")
    loops, rep = loop_averages(score_records, "repetition")
    _loops, overall = loop_averages(score_records, "overall")
    for l, r, o in zip(loops, rep, overall):
        n_chunks = sum(1 for li, _c, _s in score_records if li == l)
        print(f"  loop {l:2d}: {n_chunks:3d} chunks   repetition={r:.4f}   overall={o:.4f}")

    rep_slope = trend_slope(loops, rep)
    overall_slope = trend_slope(loops, overall)
    print(f"  linear trend across loops: repetition slope={rep_slope:+.5f}/loop   overall slope={overall_slope:+.5f}/loop")

    fc_rep, n_fc = first_chunk_average(score_records, "repetition")
    fc_overall, _ = first_chunk_average(score_records, "overall")
    motif_rate, n_mc = first_chunk_motif_rate(motif_records)
    print(f"  first chunk of each loop (n={n_fc}, excludes loop 0): "
          f"repetition={fc_rep:.4f}   overall={fc_overall:.4f}   "
          f"non-empty motif_target rate={motif_rate:.0%} (n={n_mc})")

    return {"loops": loops, "repetition": rep, "overall": overall,
            "rep_slope": rep_slope, "overall_slope": overall_slope,
            "first_chunk_repetition": fc_rep, "first_chunk_overall": fc_overall,
            "first_chunk_motif_rate": motif_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=3)
    args = parser.parse_args()

    print(f"Running {args.loops} loops x 2 conditions over {args.chart.name}, n_candidates={args.n_candidates}...")

    persistent_scores, persistent_motifs = run_condition(args.chart, persistent=True, n_loops=args.loops, n_candidates=args.n_candidates)
    persistent_summary = report("PERSISTENT memory (shared across all loops)", persistent_scores, persistent_motifs)

    control_scores, control_motifs = run_condition(args.chart, persistent=False, n_loops=args.loops, n_candidates=args.n_candidates)
    control_summary = report("CONTROL (fresh RehearsalMemory every loop)", control_scores, control_motifs)

    print("\n=== Verdict ===")
    print(f"First-chunk-of-loop non-empty motif_target rate: "
          f"persistent={persistent_summary['first_chunk_motif_rate']:.0%} vs "
          f"control={control_summary['first_chunk_motif_rate']:.0%}  "
          f"(sanity check -- persistent should be near 100%, control near 0%, by construction)")
    print(f"First-chunk-of-loop repetition:  persistent={persistent_summary['first_chunk_repetition']:.4f}  "
          f"vs control={control_summary['first_chunk_repetition']:.4f}  "
          f"(delta={persistent_summary['first_chunk_repetition'] - control_summary['first_chunk_repetition']:+.4f})")
    print(f"Whole-run repetition trend slope: persistent={persistent_summary['rep_slope']:+.5f}/loop  "
          f"vs control={control_summary['rep_slope']:+.5f}/loop")
    print(f"Whole-run overall trend slope:    persistent={persistent_summary['overall_slope']:+.5f}/loop  "
          f"vs control={control_summary['overall_slope']:+.5f}/loop")


if __name__ == "__main__":
    main()
