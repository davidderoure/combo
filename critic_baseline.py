"""Grounds "how good is combo's sax output" in real numbers on both sides --
Phase 30, DESIGN.md §13. Prompted directly by a listening-test verdict: "by
the last solo it wasn't bad, but overall it's still very beginner-noodly --
the sax doesn't seem to have a lot to say -- and it seems mainly stuck in a
narrow range of notes." Two questions, both answered by running the SAME
critic (ensemble/critic.py's musicality_score) over real data, not by ear:
does the critic itself reward real playing (score real WJD solos), and how
does combo's own output compare, measured the same way (score N self-test
takes)?

    python critic_baseline.py                        # WJD (whole corpus) + 20 combo takes
    python critic_baseline.py --takes 40
    python critic_baseline.py --wjd-only
    python critic_baseline.py --self-test-only
    python critic_baseline.py --corpus                 # combo takes also use corpus_familiarity,
                                                          # mirroring self_test.py's own flag
    python critic_baseline.py --credit-resolved-tension

Chunking uses Phase 30's new wjd_corpus.iter_solos_with_chord_idx/
split_into_chord_runs -- full chord_idx (root+quality), not Phase 29's
quality-only tagging, because dissonance_scale/tonal_conformity depend on
the actual root. Each chord-idx run is scored via musicality_score with the
PREVIOUS run's own notes as call_response_relatedness' seed_phrase (a
solo's own previous phrase standing in for "what was heard before" -- WJD
has no separate co-performer part to draw from at chunk granularity; a real,
stated approximation, not a perfect analogue of live call-and-response). A
run's first chunk in a solo gets an empty seed, the same convention
_build_seed_phrase already uses for bar 0.

Real WJD transcriptions have no explicit rest events -- a gap between notes
is just elapsed time, never a REST_PITCH row the way Wolfson's own
_inject_rests produces. Scored as-is, phrasing() would score every real
chunk near-zero regardless of actual quality -- a format mismatch, not an
honest measurement. _insert_rest_gaps below synthesizes a REST_PITCH entry
between two notes whenever their real gap (in that note's own local beat
units, the same per-note-beatdur convention duration_beats already uses) is
>= MIN_BREATH_BEATS -- the same threshold phrasing() itself already uses, so
nothing this counts that real Wolfson-generated output wouldn't also produce
a rest for. This is the ONLY adaptation needed for the WJD side; combo's own
dispensed notes already carry real REST_PITCH events from generation.

register_usage is reported twice for the WJD side: once under SAX_REGISTER
(combo's own sax voice's bound -- the fair apples-to-apples comparison) and
once under Wolfson's full trained pitch vocabulary, PITCH_MIN=44/
PITCH_MAX=93 (ensemble/wolfson/encoding.py, already a real constant) --
directly answering "how much of a real player's natural range is even being
asked about," separate from whether combo currently uses what it's given.
Combo's own generation is capped at SAX_REGISTER by construction, so there's
no second, wider pass to compare on that side.

Reuses self_test.py's own sax_generator construction (n_candidates=8,
motif_recall_candidates=20, BASS_REGISTER/SAX_REGISTER, walking_bass_stub)
so the combo side reflects exactly what self_test.py actually produces --
only bass+sax are built, not the full ensemble, since only sax's output is
ever scored by musicality_score. Reads generate.winning_score_log (Phase
30, ensemble/sax.py) -- one real MusicalityScore per chunk, the actual
dispensed candidate, not diluted by rejected search candidates -- so both
sides are one-score-per-real-chunk-played, genuinely comparable.
"""

import argparse
import time
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Tuple

import self_test as st
import wjd_corpus
from ensemble import MACHINE_SPEED, Session, Voice
from ensemble.corpus_motifs import CorpusMotifs
from ensemble.critic import MIN_BREATH_BEATS, MusicalityScore, musicality_score, register_usage
from ensemble.memory import RehearsalMemory
from ensemble.sax import sax_generator
from ensemble.wolfson.encoding import PITCH_MAX, PITCH_MIN
from ensemble.wolfson.phrase_generator import REST_PITCH
from song import parse_chart

SCORE_METRICS = [f.name for f in fields(MusicalityScore)]
WIDE_REGISTER = (PITCH_MIN, PITCH_MAX)


def _insert_rest_gaps(run_notes: list) -> list:
    """See module docstring. run_notes: a chord-idx run's notes, each
    carrying "pitch", "duration_beats", "onset" (seconds), "beatdur"
    (seconds/beat) -- from wjd_corpus.iter_solos_with_chord_idx. Returns a
    new scoring-ready note list ("pitch"/"duration_beats" only, matching
    what musicality_score expects), with a synthetic REST_PITCH entry
    spliced in wherever a real gap >= MIN_BREATH_BEATS is found. A negative
    or zero gap (overlapping/adjacent notes, a real possibility in a human
    transcription) never triggers a rest -- falls below the threshold
    naturally, no separate check needed."""
    if not run_notes:
        return []
    result = [{"pitch": run_notes[0]["pitch"], "duration_beats": run_notes[0]["duration_beats"]}]
    for prev, note in zip(run_notes, run_notes[1:]):
        prev_offset = prev["onset"] + prev["duration_beats"] * prev["beatdur"]
        gap_beats = (note["onset"] - prev_offset) / prev["beatdur"] if prev["beatdur"] else 0.0
        if gap_beats >= MIN_BREATH_BEATS:
            result.append({"pitch": REST_PITCH, "duration_beats": gap_beats})
        result.append({"pitch": note["pitch"], "duration_beats": note["duration_beats"]})
    return result


def score_wjd_corpus(db_path: Path) -> Tuple[List[MusicalityScore], List[float]]:
    """Returns (scores, wide_register_usages) -- one entry per scored chunk
    in each, same order. wide_register_usages is register_usage recomputed
    under WIDE_REGISTER, a secondary pass alongside the primary
    SAX_REGISTER-scored MusicalityScore (see module docstring)."""
    scores: List[MusicalityScore] = []
    wide_register_usages: List[float] = []
    for solo in wjd_corpus.iter_solos_with_chord_idx(db_path):
        seed_phrase: list = []
        for chord_idx, run_notes in wjd_corpus.split_into_chord_runs(solo):
            real_notes = [{"pitch": n["pitch"], "duration_beats": n["duration_beats"]} for n in run_notes]
            if len(real_notes) >= 2:
                scoring_notes = _insert_rest_gaps(run_notes)
                scores.append(musicality_score(scoring_notes, chord_idx, seed_phrase, st.SAX_REGISTER))
                wide_register_usages.append(register_usage(scoring_notes, WIDE_REGISTER))
            seed_phrase = real_notes
    return scores, wide_register_usages


def score_combo_takes(
    chart_path: Path, n_takes: int, corpus: Optional[CorpusMotifs], credit_resolved_tension: bool
) -> List[MusicalityScore]:
    """n_takes independent machine_speed Sessions -- fresh RehearsalMemory
    each, matching self_test.py's own un-`--persist`ed default -- reading
    sax_gen.winning_score_log after each. Only bass+sax are built (see
    module docstring)."""
    song = parse_chart(chart_path.read_text())
    scores: List[MusicalityScore] = []
    for _ in range(n_takes):
        memory = RehearsalMemory()
        bass = Voice(
            id="bass", instrument="bass", register=st.BASS_REGISTER, source="ai",
            generator=st.walking_bass_stub(st.BASS_REGISTER),
        )
        sax_gen = sax_generator(
            st.SAX_REGISTER, target_voice_id="bass", memory=memory, n_candidates=8, motif_recall_candidates=20,
            credit_resolved_tension=credit_resolved_tension, corpus=corpus,
        )
        sax = Voice(id="sax", instrument="sax", register=st.SAX_REGISTER, source="ai", generator=sax_gen)
        Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)
        scores.extend(sax_gen.winning_score_log)
    return scores


def summarize(scores: List[MusicalityScore]) -> dict:
    if not scores:
        return {name: float("nan") for name in SCORE_METRICS}
    return {name: sum(getattr(s, name) for s in scores) / len(scores) for name in SCORE_METRICS}


def print_comparison(
    wjd_summary: Optional[dict], combo_summary: Optional[dict], wjd_wide_register: Optional[float]
) -> None:
    print(f"\n{'metric':24s} {'WJD (real)':>12s} {'combo':>12s}")
    for name in SCORE_METRICS:
        wjd_v = f"{wjd_summary[name]:.4f}" if wjd_summary is not None else "n/a"
        combo_v = f"{combo_summary[name]:.4f}" if combo_summary is not None else "n/a"
        print(f"{name:24s} {wjd_v:>12s} {combo_v:>12s}")
    if wjd_summary is not None:
        print(f"\nregister_usage under combo's own SAX_REGISTER {st.SAX_REGISTER}: WJD={wjd_summary['register_usage']:.4f}")
    if wjd_wide_register is not None:
        print(f"register_usage under Wolfson's full trained pitch vocabulary {WIDE_REGISTER}: WJD={wjd_wide_register:.4f}")
        print("(combo's own generation is capped at SAX_REGISTER by construction -- no wide pass to compare)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes", type=int, default=20)
    parser.add_argument("--chart", type=Path, default=st.DEFAULT_CHART)
    parser.add_argument("--wjd-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--corpus", action="store_true",
                         help="Combo takes also use corpus_familiarity, mirroring self_test.py's own flag.")
    parser.add_argument("--credit-resolved-tension", action="store_true")
    args = parser.parse_args()

    wjd_summary = None
    wjd_wide_register = None
    if not args.self_test_only:
        if not wjd_corpus.DB_PATH.exists():
            print(f"{wjd_corpus.DB_PATH} not found -- see wjd_corpus.py's module docstring for how to obtain it.")
        else:
            print("Scoring the real WJD corpus...")
            start = time.perf_counter()
            wjd_scores, wjd_wide_usages = score_wjd_corpus(wjd_corpus.DB_PATH)
            elapsed = time.perf_counter() - start
            wjd_summary = summarize(wjd_scores)
            wjd_wide_register = sum(wjd_wide_usages) / len(wjd_wide_usages) if wjd_wide_usages else float("nan")
            print(f"  {len(wjd_scores)} real chunks scored in {elapsed:.2f}s")

    combo_summary = None
    if not args.wjd_only:
        corpus = None
        if args.corpus:
            if wjd_corpus.CACHE_PATH.exists():
                corpus = CorpusMotifs(wjd_corpus.CACHE_PATH)
            else:
                print(f"{wjd_corpus.CACHE_PATH} not found -- run 'python wjd_corpus.py --build' first. Continuing without --corpus.")
        print(f"Running {args.takes} combo self-test takes...")
        start = time.perf_counter()
        combo_scores = score_combo_takes(args.chart, args.takes, corpus, args.credit_resolved_tension)
        elapsed = time.perf_counter() - start
        combo_summary = summarize(combo_scores)
        print(f"  {len(combo_scores)} real chunks scored across {args.takes} takes in {elapsed:.2f}s")

    print_comparison(wjd_summary, combo_summary, wjd_wide_register)


if __name__ == "__main__":
    main()
