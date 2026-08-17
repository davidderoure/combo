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

Instrumentation, two layers:
- generate.motif_adherence_log (ensemble/sax.py, Phase 17) is read directly
  after each loop's Session.generate() -- one entry per plan-chunk-build, the
  WINNING candidate's motif_adherence. Simplest and most direct: no spying
  needed for this, since sax_generator already exposes it for exactly this
  kind of external inspection.
- A spy on ensemble.sax's own bound name for musicality_score (NOT
  ensemble.critic.musicality_score -- sax.py did `from .critic import
  musicality_score`, so it has its own local reference; patching the critic
  module's attribute wouldn't touch sax.py's already-bound name) and on
  PhraseGenerator.generate, kept for the secondary repetition/overall/
  motif_target-rate measurements this tool already reported. Both call
  through to the real implementation -- the same spy-not-mock technique
  test_sax_wolfson_integration.py and ensemble/demo.py already use.

Primary metric: `motif_adherence` (Phase 17), not `repetition` -- adherence
measures whether a chunk actually echoed the SPECIFIC recalled target;
repetition only measures whether a chunk repeats a pattern WITHIN ITSELF,
which has no reference to what memory recalled at all (see
ensemble/critic.py's motif_adherence docstring for why these are different
things). repetition/overall are still reported for context.

Sharpest single measurement: the FIRST chunk of each loop after loop 0. That's
the one place cross-run persistence uniquely acts -- every other chunk in a
loop already has within-run memory available in BOTH conditions (a fresh
RehearsalMemory still accumulates within a single loop's own chunks), so a
whole-run average mixes in a lot of chunks where the two conditions aren't
actually different by construction.

First run (20 loops, n_candidates=3, blues_in_f.chart, pre-Phase-17 selection
logic): wiring confirmed correct (persistent condition's first chunk of every
loop after loop 0 carried a motif_target 100% of the time; control 0%, exactly
as expected by construction) but the musical effect was a genuine null result
-- no measurable difference in `repetition` between conditions. Traced to two
causes, not re-measured harder but fixed: repetition() was the wrong
instrument (self-similarity, not target adherence -- see motif_adherence's
docstring); and _apply_motif_bias (phrase_generator.py) only fires if the
model has already spontaneously started matching the target's prefix, rarely
true for a long (3-4 interval) motif. Phase 17 fixed both in combo-authored
code only, without touching the ported model: motif_adherence as the correct
metric, _pick_achievable_motif preferring short (2-interval) recalled motifs,
motif_recall_candidates spending extra search specifically on chunks with a
real target to aim for, and (motif_adherence, overall) as the selection key.

Second run (20 loops, n_candidates=3, motif_recall_candidates=20,
blues_in_f.chart, post-Phase-17): a clean, construction-clear effect at the
sharpest measurement point -- first-plan-chunk-of-loop motif_adherence is
1.0000 for the persistent condition (every loop after loop 0 successfully
echoed the recalled motif) vs 0.0000 for control (which never has anything to
echo), exactly the cross-loop "hear it echo an earlier rehearsal" signal this
was built for. Whole-run averages, by contrast, are close between conditions
(persistent ~1.00, control ~0.95-0.98) -- a real, honest side-effect of the
same fix, not a confound: motif_recall_candidates=20 with adherence-driven
search makes ANY chunk with a non-empty motif_targets very likely to hit
adherence=1.0, and within-run persistence (Phase 11) means both conditions
accumulate something to target after their own first chunk. So the mechanism
is now reliably audible throughout a run, not just at rehearsal boundaries --
the first-plan-chunk measurement remains the one place that specifically
isolates cross-loop persistence, and that's where the clear 1.0 vs 0.0 split
lives.
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
    incremented once per candidate-scoring call (n_candidates or
    motif_recall_candidates calls per plan-chunk-build, not one) -- see
    report()'s "candidate scores" label below, deliberately not called
    "chunks" to avoid conflating the two."""
    original_score = sax_module.musicality_score
    original_generate = PhraseGenerator.generate

    score_records = []  # (loop, candidate_index_in_loop, MusicalityScore)
    motif_records = []  # (loop, candidate_index_in_loop, motif_strength)
    state = {"loop": -1, "candidate": -1}

    def score_wrapper(notes, chord_idx, seed_phrase, weights=None):
        state["candidate"] += 1
        score = original_score(notes, chord_idx, seed_phrase, weights=weights)
        score_records.append((state["loop"], state["candidate"], score))
        return score

    def generate_wrapper(self, *args, **kwargs):
        motif_targets = kwargs.get("motif_targets")
        motif_strength = kwargs.get("motif_strength", 0.0)
        motif_records.append((state["loop"], state["candidate"] + 1, motif_strength if motif_targets else 0.0))
        return original_generate(self, *args, **kwargs)

    sax_module.musicality_score = score_wrapper
    PhraseGenerator.generate = generate_wrapper
    try:
        yield score_records, motif_records, state
    finally:
        sax_module.musicality_score = original_score
        PhraseGenerator.generate = original_generate


def run_condition(chart_path: Path, persistent: bool, n_loops: int, n_candidates: int, motif_recall_candidates: int):
    """Returns (score_records, motif_records, adherence_records). adherence_records
    is [(loop_index, [per-chunk winning motif_adherence, ...]), ...] -- read
    directly from each loop's sax_gen.motif_adherence_log, one list per loop."""
    adherence_records = []
    with spy_on_sax_generation() as (score_records, motif_records, state):
        song = parse_chart(chart_path.read_text())
        memory = RehearsalMemory() if persistent else None
        for i in range(n_loops):
            state["loop"] = i
            state["candidate"] = -1
            if not persistent:
                memory = RehearsalMemory()
            bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai",
                         generator=chord_tone_generator(BASS_REGISTER))
            sax_gen = sax_module.sax_generator(
                SAX_REGISTER, target_voice_id="bass", memory=memory,
                n_candidates=n_candidates, motif_recall_candidates=motif_recall_candidates,
            )
            sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
            Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)
            adherence_records.append((i, list(sax_gen.motif_adherence_log)))
    return score_records, motif_records, adherence_records


def loop_averages(score_records, metric):
    per_loop = defaultdict(list)
    for loop_idx, _candidate_idx, score in score_records:
        per_loop[loop_idx].append(getattr(score, metric))
    loops = sorted(per_loop)
    return loops, [sum(per_loop[l]) / len(per_loop[l]) for l in loops]


def trend_slope(loops, values):
    if len(loops) < 2:
        return 0.0
    slope, _intercept = np.polyfit(loops, values, 1)
    return slope


def adherence_loop_averages(adherence_records):
    loops = [i for i, _log in adherence_records]
    return loops, [(sum(log) / len(log)) if log else 0.0 for _i, log in adherence_records]


def first_chunk_adherence_average(adherence_records, skip_loop_0=True):
    """Average of the FIRST plan-chunk's winning motif_adherence, across loops --
    the one chunk where cross-run persistence uniquely differs between
    conditions (every later chunk in a loop already has within-run memory
    available in both conditions)."""
    values = [log[0] for i, log in adherence_records if log and not (skip_loop_0 and i == 0)]
    return (sum(values) / len(values)) if values else float("nan"), len(values)


def first_candidate_motif_rate(motif_records, skip_loop_0=True):
    first = [m for (loop_idx, candidate_idx, m) in motif_records if candidate_idx == 0 and not (skip_loop_0 and loop_idx == 0)]
    if not first:
        return 0.0, 0
    return sum(1 for m in first if m > 0) / len(first), len(first)


def report(label, score_records, motif_records, adherence_records):
    print(f"\n=== {label} ===")
    loops, rep = loop_averages(score_records, "repetition")
    _loops, overall = loop_averages(score_records, "overall")
    _loops2, adherence = adherence_loop_averages(adherence_records)
    for l, r, o, a in zip(loops, rep, overall, adherence):
        n_candidate_scores = sum(1 for li, _c, _s in score_records if li == l)
        print(f"  loop {l:2d}: {n_candidate_scores:4d} candidate scores   "
              f"motif_adherence={a:.4f}   repetition={r:.4f}   overall={o:.4f}")

    adherence_slope = trend_slope(loops, adherence)
    rep_slope = trend_slope(loops, rep)
    print(f"  linear trend across loops: motif_adherence slope={adherence_slope:+.5f}/loop   "
          f"repetition slope={rep_slope:+.5f}/loop")

    fc_adherence, n_fc = first_chunk_adherence_average(adherence_records)
    motif_rate, n_mc = first_candidate_motif_rate(motif_records)
    print(f"  first plan-chunk of each loop (n={n_fc}, excludes loop 0): "
          f"motif_adherence={fc_adherence:.4f}   "
          f"non-empty motif_target rate={motif_rate:.0%} (n={n_mc})")

    return {"loops": loops, "repetition": rep, "overall": overall, "adherence": adherence,
            "adherence_slope": adherence_slope, "rep_slope": rep_slope,
            "first_chunk_adherence": fc_adherence, "first_chunk_motif_rate": motif_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--motif-recall-candidates", type=int, default=20)
    args = parser.parse_args()

    print(f"Running {args.loops} loops x 2 conditions over {args.chart.name}, "
          f"n_candidates={args.n_candidates}, motif_recall_candidates={args.motif_recall_candidates}...")

    p_scores, p_motifs, p_adherence = run_condition(
        args.chart, persistent=True, n_loops=args.loops,
        n_candidates=args.n_candidates, motif_recall_candidates=args.motif_recall_candidates,
    )
    persistent_summary = report("PERSISTENT memory (shared across all loops)", p_scores, p_motifs, p_adherence)

    c_scores, c_motifs, c_adherence = run_condition(
        args.chart, persistent=False, n_loops=args.loops,
        n_candidates=args.n_candidates, motif_recall_candidates=args.motif_recall_candidates,
    )
    control_summary = report("CONTROL (fresh RehearsalMemory every loop)", c_scores, c_motifs, c_adherence)

    print("\n=== Verdict ===")
    print(f"First-plan-chunk non-empty motif_target rate: "
          f"persistent={persistent_summary['first_chunk_motif_rate']:.0%} vs "
          f"control={control_summary['first_chunk_motif_rate']:.0%}  "
          f"(sanity check -- persistent should be near 100%, control near 0%, by construction)")
    print(f"First-plan-chunk motif_adherence: persistent={persistent_summary['first_chunk_adherence']:.4f}  "
          f"vs control={control_summary['first_chunk_adherence']:.4f}  "
          f"(delta={persistent_summary['first_chunk_adherence'] - control_summary['first_chunk_adherence']:+.4f})")
    print(f"Whole-run motif_adherence trend slope: persistent={persistent_summary['adherence_slope']:+.5f}/loop  "
          f"vs control={control_summary['adherence_slope']:+.5f}/loop")
    print(f"Whole-run repetition trend slope:      persistent={persistent_summary['rep_slope']:+.5f}/loop  "
          f"vs control={control_summary['rep_slope']:+.5f}/loop")


if __name__ == "__main__":
    main()
