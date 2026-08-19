"""Builds quality-conditioned Markov transition tables (pitch-interval and
duration-token) from the real Weimar Jazz Database -- Phase 35, DESIGN.md
§13. A different local generator sharing combo's own critic and
search-and-select architecture (ensemble/markov_sax.py's
markov_sax_generator), prompted by David's own prior use of a Markov chain
in unrelated piano-musicality work, and by the WJD-in/WJD-out reflection: a
Markov chain and an LSTM are both short-context local generators, so
comparing them under the identical critic tests whether combo's elevated
`repetition` (Phase 34, temperature-insensitive) is neural-sampling-specific
or a more general property of local generation.

Order and data source decided empirically, not guessed (checked directly
against the real corpus, reusing wjd_corpus.iter_solos/
split_into_quality_runs -- no new WJD traversal needed): order-1 (context =
the single preceding interval) gives 42-67 distinct contexts per chord
quality with a 15-26% singleton rate -- well-populated, reliable. Order-2
jumps to 300-903 contexts with a 29-38% singleton rate, worst for the
smallest quality bucket (diminished) -- a real sparsity problem, since a
singleton context just replays its one training example verbatim, making a
higher-order chain MORE mechanical, not less. Order-1 is the default here;
`order` stays a real parameter so a future run can try order-2 and see the
tradeoff directly. Duration tokens are comfortably well-populated even at
order-1 (only 32 possible tokens, 0-3% singleton rate across all four
qualities). Interval values span -42 to +41 semitones but are heavily
concentrated on small steps (checked directly), so a plain Counter-based
table needs no binning/smoothing for this MVP.

Two INDEPENDENT chains (pitch-interval, duration-token), not a joint model
-- real phrasing correlates rhythm and pitch contour (e.g. longer notes at
phrase ends), which this MVP doesn't capture; a named simplification, not
hidden.

    python markov_corpus.py --build       # extract transitions, cache, report build time
    python markov_corpus.py --benchmark   # load the cache, time N sample draws

Explicitly NOT attempted here: higher-order chains beyond what --build's
--order flag is asked for; joint pitch+duration modeling; anything about
how the generator USES these tables (ensemble/markov_sax.py's job).
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import wjd_corpus
from ensemble.wolfson.encoding import dur_to_token

CACHE_PATH = Path(__file__).resolve().parent / "wjd_data" / "markov_tables.json"
DEFAULT_ORDER = 1
N_SAMPLES = 20  # matches wjd_corpus.py's own N_LOOKUPS scale for --benchmark


def build_tables(db_path: Path, order: int = DEFAULT_ORDER):
    """Returns (pitch_transitions, duration_transitions, pitch_marginals,
    duration_marginals) -- all Dict[int, ...] keyed by chord quality.
    pitch_transitions[quality] is Dict[context_tuple, Counter[next_interval]];
    duration_transitions[quality] is Dict[context_tuple, Counter[next_token]].
    Marginals are a single Counter per quality (context-free), the fallback
    for a context never observed in training -- built from the SAME pass,
    not a separate query."""
    pitch_transitions: Dict[int, Dict[tuple, Counter]] = {q: defaultdict(Counter) for q in wjd_corpus.QUALITY_NAMES}
    duration_transitions: Dict[int, Dict[tuple, Counter]] = {q: defaultdict(Counter) for q in wjd_corpus.QUALITY_NAMES}
    pitch_marginals: Dict[int, Counter] = {q: Counter() for q in wjd_corpus.QUALITY_NAMES}
    duration_marginals: Dict[int, Counter] = {q: Counter() for q in wjd_corpus.QUALITY_NAMES}

    for solo in wjd_corpus.iter_solos(db_path):
        for quality, run in wjd_corpus.split_into_quality_runs(solo):
            pitches = [n["pitch"] for n in run]
            intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
            for iv in intervals:
                pitch_marginals[quality][iv] += 1
            for i in range(len(intervals) - order):
                context = tuple(intervals[i : i + order])
                pitch_transitions[quality][context][intervals[i + order]] += 1

            tokens = [dur_to_token(n["duration_beats"]) for n in run]
            for tok in tokens:
                duration_marginals[quality][tok] += 1
            for i in range(len(tokens) - order):
                context = tuple(tokens[i : i + order])
                duration_transitions[quality][context][tokens[i + order]] += 1

    return pitch_transitions, duration_transitions, pitch_marginals, duration_marginals


def save_cache(
    path: Path,
    order: int,
    pitch_transitions: Dict[int, Dict[tuple, Counter]],
    duration_transitions: Dict[int, Dict[tuple, Counter]],
    pitch_marginals: Dict[int, Counter],
    duration_marginals: Dict[int, Counter],
) -> None:
    """Write-then-atomic-replace, same pattern as wjd_corpus.py's save_cache.
    A context tuple has no direct JSON representation, so each
    quality->context->Counter table is flattened to a list of
    [context_list, {outcome_str: count}] pairs -- context is a list (not a
    joined string), so any order round-trips without a delimiter choice."""

    def _flatten(transitions: Dict[int, Dict[tuple, Counter]]) -> dict:
        return {
            str(q): [[list(ctx), {str(k): v for k, v in counter.items()}] for ctx, counter in table.items()]
            for q, table in transitions.items()
        }

    def _flatten_marginal(marginals: Dict[int, Counter]) -> dict:
        return {str(q): {str(k): v for k, v in counter.items()} for q, counter in marginals.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "order": order,
        "pitch_transitions": _flatten(pitch_transitions),
        "duration_transitions": _flatten(duration_transitions),
        "pitch_marginals": _flatten_marginal(pitch_marginals),
        "duration_marginals": _flatten_marginal(duration_marginals),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(raw))
    tmp.replace(path)


def run_build(order: int) -> None:
    if not wjd_corpus.DB_PATH.exists():
        print(f"{wjd_corpus.DB_PATH} not found -- see wjd_corpus.py's module docstring for how to obtain it.")
        return
    start = time.perf_counter()
    pitch_transitions, duration_transitions, pitch_marginals, duration_marginals = build_tables(
        wjd_corpus.DB_PATH, order=order
    )
    elapsed = time.perf_counter() - start
    save_cache(CACHE_PATH, order, pitch_transitions, duration_transitions, pitch_marginals, duration_marginals)
    print(f"Built order-{order} Markov tables.")
    for q, name in wjd_corpus.QUALITY_NAMES.items():
        n_pitch_ctx = len(pitch_transitions[q])
        n_dur_ctx = len(duration_transitions[q])
        print(f"  {name:10s}: {n_pitch_ctx:5d} pitch contexts, {n_dur_ctx:5d} duration contexts")
    print(f"Build time: {elapsed:.3f}s")
    print(f"Cached to {CACHE_PATH}")


def run_benchmark() -> None:
    if not CACHE_PATH.exists():
        print(f"{CACHE_PATH} not found -- run 'python markov_corpus.py --build' first.")
        return
    from ensemble.markov_tables import MarkovTables
    from ensemble.wolfson.chords import QUAL_DOM

    tables = MarkovTables(CACHE_PATH)
    rng = random.Random(42)
    start = time.perf_counter()
    for _ in range(N_SAMPLES):
        tables.sample_next_interval(QUAL_DOM, (2,), rng)
        tables.sample_next_duration_token(QUAL_DOM, (10,), rng)
    elapsed = time.perf_counter() - start
    print(f"{N_SAMPLES} sample draws (pitch + duration each): {elapsed * 1000:.3f}ms total, "
          f"{elapsed * 1000 / (N_SAMPLES * 2):.5f}ms per draw.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Extract transitions from WJD and cache to disk.")
    parser.add_argument("--benchmark", action="store_true", help="Time N sample draws against the cache.")
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                         help="Markov order (context length). Default 1, empirically the well-populated choice.")
    args = parser.parse_args()

    if not args.build and not args.benchmark:
        parser.error("pass --build and/or --benchmark")

    if args.build:
        run_build(args.order)
    if args.benchmark:
        run_benchmark()


if __name__ == "__main__":
    main()
