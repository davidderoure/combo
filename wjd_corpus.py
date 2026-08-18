"""WJD corpus feasibility benchmark -- Phase 28, DESIGN.md §13.

Answers the question raised discussing a corpus-similarity critic: is
precomputing a corpus-wide motif frequency table (mirroring RehearsalMemory's
own Counter-based approach, ensemble/memory.py) actually fast enough, measured
for real rather than estimated? This script is feasibility/benchmarking ONLY
-- it builds and times the lookup structure, it does not wire anything into
sax_generator's real selection loop, and it does not decide how a corpus-based
signal should relate to the existing rule-based critic (David's own still-open
question: "we'll see what combination we need").

Data: the Weimar Jazz Database (jazzomat.hfm-weimar.de), wjazzd.db -- a SQLite
database, 456 solos, 200,809 notes, downloaded with explicit permission into
wjd_data/ (gitignored -- external research data, not source; see .gitignore).
Verified directly against real rows (not assumed) before writing the
extraction below: melody.onset/duration are in SECONDS, not beats;
melody.beatdur gives each note's own local beat duration in seconds, so
duration_beats = duration / beatdur reproduces exactly the per-note
beat_dur_sec convention ensemble/wolfson/encoding.py's phrase_to_tokens
already uses (note.get("beat_dur_sec") or a tempo fallback) -- here every note
already carries its own real value, no fallback needed.

    python wjd_corpus.py --build       # extract every solo's motifs, cache to
                                          # wjd_data/wjd_motifs.json, report build time
    python wjd_corpus.py --benchmark   # load the cache, time N_LOOKUPS candidate
                                          # lookups against a real sample chunk's
                                          # own motifs

Reuses ensemble/wolfson/motifs.py's extract_interval_motifs (pitch) and
ensemble/rhythm_motifs.py's extract_duration_motifs (duration) directly --
this script is orchestration only (connect, extract, time, report); the real
logic lives in those already-tested library functions, not here.

Explicitly NOT attempted here (named rather than silently out of reach):
wiring any corpus signal into sax_generator's real selection loop; deciding
the rule-based/corpus-based balance; a near-match/edit-distance fallback for
motifs not found exactly in the table; using the beats table's fuller chord
annotations; using the raw unquantized MIDI's expressive timing.
"""

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from ensemble.rhythm_motifs import extract_duration_motifs
from ensemble.wolfson.motifs import extract_interval_motifs

DB_PATH = Path(__file__).resolve().parent / "wjd_data" / "wjazzd.db"
CACHE_PATH = Path(__file__).resolve().parent / "wjd_data" / "wjd_motifs.json"
N_LOOKUPS = 20  # matches self_test.py/out_of_key_check.py's motif_recall_candidates default


def iter_solos(db_path: Path):
    """Yields one list of {"pitch": int, "duration_beats": float} dicts per
    solo (melid), ordered by onset within each solo."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT melid, pitch, duration, beatdur FROM melody ORDER BY melid, onset")
        current_melid = None
        current_solo: list = []
        for melid, pitch, duration, beatdur in cur:
            if melid != current_melid:
                if current_solo:
                    yield current_solo
                current_melid = melid
                current_solo = []
            current_solo.append({"pitch": int(round(pitch)), "duration_beats": duration / beatdur})
        if current_solo:
            yield current_solo
    finally:
        conn.close()


def first_solo_sample(db_path: Path, n_notes: int = 8) -> list:
    """A real sample chunk (not synthetic) for the benchmark to look up
    against: the first n_notes of the lowest-melid solo -- roughly the scale
    of one PhraseGenerator.generate() chunk (see ensemble/sax.py's plan_bars
    default). A targeted query, not iter_solos(), so it doesn't pay the cost
    of sorting the whole 200k-row table just to peek at 8 notes."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT pitch, duration, beatdur FROM melody "
            "WHERE melid = (SELECT MIN(melid) FROM melody) ORDER BY onset LIMIT ?",
            (n_notes,),
        )
        return [{"pitch": int(round(p)), "duration_beats": d / bd} for p, d, bd in cur]
    finally:
        conn.close()


def build_corpus(db_path: Path):
    """Returns (pitch_motif_counter, duration_motif_counter, n_solos, n_notes)."""
    pitch_counter: Counter = Counter()
    duration_counter: Counter = Counter()
    n_solos = 0
    n_notes = 0
    for solo in iter_solos(db_path):
        n_solos += 1
        n_notes += len(solo)
        pitch_counter.update(extract_interval_motifs(solo))
        duration_counter.update(extract_duration_motifs(solo))
    return pitch_counter, duration_counter, n_solos, n_notes


def save_cache(path: Path, pitch_counter: Counter, duration_counter: Counter, n_solos: int, n_notes: int) -> None:
    """Write-then-atomic-replace, same pattern as ensemble/memory.py's
    RehearsalMemory._write -- a crash mid-write can never corrupt a prior
    cache. JSON has no tuple type, so motifs round-trip list<->tuple
    explicitly, same convention as RehearsalMemory._read/_write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "n_solos": n_solos,
        "n_notes": n_notes,
        "pitch_motifs": [[list(motif), count] for motif, count in pitch_counter.items()],
        "duration_motifs": [[list(motif), count] for motif, count in duration_counter.items()],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(raw))
    tmp.replace(path)


def load_cache(path: Path):
    raw = json.loads(path.read_text())
    pitch_counter = Counter({tuple(motif): count for motif, count in raw["pitch_motifs"]})
    duration_counter = Counter({tuple(motif): count for motif, count in raw["duration_motifs"]})
    return pitch_counter, duration_counter, raw["n_solos"], raw["n_notes"]


def run_build() -> None:
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found -- see wjd_corpus.py's module docstring for how to obtain it.")
        return
    start = time.perf_counter()
    pitch_counter, duration_counter, n_solos, n_notes = build_corpus(DB_PATH)
    elapsed = time.perf_counter() - start
    save_cache(CACHE_PATH, pitch_counter, duration_counter, n_solos, n_notes)
    print(f"Built corpus from {n_solos} solos, {n_notes} notes.")
    print(f"  pitch motifs:    {len(pitch_counter)} distinct, {sum(pitch_counter.values())} total occurrences")
    print(f"  duration motifs: {len(duration_counter)} distinct, {sum(duration_counter.values())} total occurrences")
    print(f"Build time: {elapsed:.3f}s")
    print(f"Cached to {CACHE_PATH}")


def run_benchmark() -> None:
    if not CACHE_PATH.exists():
        print(f"{CACHE_PATH} not found -- run 'python wjd_corpus.py --build' first.")
        return
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found -- benchmark needs it to build a real sample chunk to look up.")
        return
    pitch_counter, duration_counter, n_solos, n_notes = load_cache(CACHE_PATH)
    print(f"Loaded corpus: {n_solos} solos, {n_notes} notes, "
          f"{len(pitch_counter)} pitch motifs, {len(duration_counter)} duration motifs.")

    sample_chunk = first_solo_sample(DB_PATH)
    candidate_pitch_motifs = extract_interval_motifs(sample_chunk)
    candidate_duration_motifs = extract_duration_motifs(sample_chunk)
    print(f"Sample chunk: {len(sample_chunk)} notes, "
          f"{len(candidate_pitch_motifs)} pitch motifs, {len(candidate_duration_motifs)} duration motifs to look up.")

    start = time.perf_counter()
    for _ in range(N_LOOKUPS):
        for motif in candidate_pitch_motifs:
            pitch_counter.get(motif, 0)
        for motif in candidate_duration_motifs:
            duration_counter.get(motif, 0)
    elapsed = time.perf_counter() - start

    total_lookups = N_LOOKUPS * (len(candidate_pitch_motifs) + len(candidate_duration_motifs))
    per_lookup_ms = elapsed * 1000 / total_lookups if total_lookups else 0.0
    print(f"{N_LOOKUPS} candidates x {len(candidate_pitch_motifs) + len(candidate_duration_motifs)} motifs each "
          f"({total_lookups} individual lookups): {elapsed * 1000:.3f}ms total, {per_lookup_ms:.5f}ms per lookup.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Extract every solo's motifs and cache to disk.")
    parser.add_argument("--benchmark", action="store_true", help="Time N_LOOKUPS candidate lookups against the cache.")
    args = parser.parse_args()

    if not args.build and not args.benchmark:
        parser.error("pass --build and/or --benchmark")

    if args.build:
        run_build()
    if args.benchmark:
        run_benchmark()


if __name__ == "__main__":
    main()
