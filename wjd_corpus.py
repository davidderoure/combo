"""WJD corpus feasibility benchmark and chord-quality-tagged motif table --
Phases 28-29, DESIGN.md §13.

Phase 28 answered the question raised discussing a corpus-similarity critic:
is precomputing a corpus-wide motif frequency table (mirroring
RehearsalMemory's own Counter-based approach, ensemble/memory.py) actually
fast enough, measured for real rather than estimated? Yes -- see the numbers
this script prints.

Phase 29 chord-tags that table (major/dominant/minor/diminished, Wolfson's
own 4-class system, ensemble/wolfson/chords.py) so it's queryable the same
way the LSTM itself conditions on chord, and is now the source for a
narrowly-scoped ensemble/critic.py corpus_familiarity function wired into
ensemble/sax.py's sax_generator -- but ONLY for chunks where generation is
already pushed off the model's natural distribution (a recalled motif target
or modal_strength), not a blanket addition to every candidate's score. See
DESIGN.md/README for the full reasoning (the LSTM-vs-corpus-table redundancy
question, and why a broad application would risk favouring "average WJD"
over deliberate boldness). This script itself remains build/benchmark
orchestration only -- the real logic lives in the tested library functions
it calls (ensemble/wolfson/motifs.py, ensemble/rhythm_motifs.py,
ensemble/corpus_motifs.py) and in critic.py/sax.py's own wiring, not here.

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

Chord quality per note comes from the beats table's chord column, annotated
only at CHANGE points (verified: 30,548 of 132,329 beats rows are non-empty),
holding until the next one -- the same "chart" convention combo's own
song/chart.py uses ('%' repeats the previous chord). Classified via a new,
combo-authored _wjd_chord_quality, NOT ensemble.wolfson.chords.parse_chord:
checked directly before relying on the existing ported parser,
parse_chord("Abj7") returns QUAL_DOM, not QUAL_MAJOR -- its quality check
looks for the substring "maj", but Jazzomat's own notation marks a
major-seventh with a bare "j" ("Abj7", not "Abmaj7"), so it falls through to
the dominant regex instead. Not a rare edge case: suffixes starting with "j"
(excluding "-j" minor-major, which correctly hits the minor check first)
total 3,263 of 30,548 real annotations (10.7%). Per this project's standing
rule (never edit ensemble/wolfson/*.py), _wjd_chord_quality below is
combo-authored, reusing only the QUAL_* integer constants from
ensemble.wolfson.chords, not parse_chord's logic. Verified against the full
real suffix vocabulary (48 distinct suffixes covering all 30,548
annotations, zero fallthrough): DOM 14,807 / MIN 8,295 / MAJ 5,393 /
DIM 1,652 / NC 401 -- a sane jazz-harmony distribution.

    python wjd_corpus.py --build       # extract every solo's motifs (per
                                          # chord quality), cache to
                                          # wjd_data/wjd_motifs.json, report
                                          # build time
    python wjd_corpus.py --benchmark   # load the cache, time N_LOOKUPS
                                          # candidate lookups against a real
                                          # sample chunk's own motifs

Explicitly NOT attempted here (named rather than silently out of reach):
deciding the overall rule-based/corpus-based balance; a near-match/
edit-distance fallback for motifs not found exactly in the table; using the
beats table's fuller chord-EXTENSION annotations (quality only, here); using
the raw unquantized MIDI's expressive timing.
"""

import argparse
import bisect
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ensemble.rhythm_motifs import extract_duration_motifs
from ensemble.wolfson.chords import N_CHORD_TYPES, N_QUALITIES, QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR, parse_chord
from ensemble.wolfson.motifs import extract_interval_motifs

DB_PATH = Path(__file__).resolve().parent / "wjd_data" / "wjazzd.db"
CACHE_PATH = Path(__file__).resolve().parent / "wjd_data" / "wjd_motifs.json"
N_LOOKUPS = 20  # matches self_test.py/out_of_key_check.py's motif_recall_candidates default

QUALITY_NAMES = {QUAL_MAJOR: "major", QUAL_DOM: "dominant", QUAL_MINOR: "minor", QUAL_DIM: "diminished"}

_ROOT_RE = re.compile(r"^[A-G][#b]?")


def _wjd_chord_quality(chord: str) -> Optional[int]:
    """Classifies a WJD/Jazzomat chord string into Wolfson's 4-class
    quality system. NOT ensemble.wolfson.chords.parse_chord -- see module
    docstring for the empirically-found "j"=major7 misclassification that
    rules it out.

    Strips slash-bass and root, then classifies the remaining suffix:
    'm7b5'/'o'* -> DIM; '-'* -> MINOR; 'j'* -> MAJOR; '+'*/'sus'* -> DOM
    (augmented has no combo quality class; approximated as dominant -- rare,
    a few hundred of 30,548 real annotations); ''/'6'* -> MAJOR; else (a bare
    digit-led extension: 7/9/11/13/alt) -> DOM. 'NC' (no chord) -> None.
    Verified against the full real suffix vocabulary (48 distinct suffixes,
    30,548 annotations) with zero fallthrough."""
    if chord == "NC":
        return None
    body = chord.split("/")[0]
    m = _ROOT_RE.match(body)
    suffix = body[m.end() :] if m else body
    if suffix.startswith("m7b5") or suffix.startswith("o"):
        return QUAL_DIM
    if suffix.startswith("-"):
        return QUAL_MINOR
    if suffix.startswith("j"):
        return QUAL_MAJOR
    if suffix.startswith("+") or suffix.startswith("sus"):
        return QUAL_DOM
    if suffix == "" or suffix.startswith("6"):
        return QUAL_MAJOR
    return QUAL_DOM


def _wjd_chord_idx(chord: str) -> Optional[int]:
    """Full chord_idx (root*N_QUALITIES + quality, 0-47) for a WJD chord
    string -- Phase 30, needed because critic functions like
    dissonance_scale/tonal_conformity depend on the actual root, not just
    quality (unlike Phase 29's motif-frequency tagging, which is deliberately
    quality-only since motifs are already transposition-invariant).

    Combines parse_chord's root extraction (its OWN quality half has the
    "j"=major7 bug _wjd_chord_quality above works around -- but its root
    half is a separate step, verified reliable directly: spot-checked across
    sharps/flats/enharmonics, e.g. parse_chord("F#7")'s root correctly comes
    out as Gb, matching combo's own flat-only ROOTS spelling) with our own
    corrected _wjd_chord_quality. None for 'NC' or anything parse_chord
    itself can't find a root for."""
    quality = _wjd_chord_quality(chord)
    if quality is None:
        return None
    parsed = parse_chord(chord)
    if parsed >= N_CHORD_TYPES:  # NC_INDEX -- parse_chord couldn't find a root
        return None
    root_idx = parsed // N_QUALITIES
    return root_idx * N_QUALITIES + quality


def _changes_by_melid(db_path: Path, classify) -> Dict[int, Tuple[List[float], List[Optional[int]]]]:
    """melid -> (onsets, values) parallel lists, sorted by onset -- one
    query, reused for every note in that solo rather than re-queried per
    note. `classify` (a chord string -> Optional[int] function -- either
    _wjd_chord_quality or _wjd_chord_idx) determines what's tracked at each
    change point."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT melid, onset, chord FROM beats WHERE chord != '' ORDER BY melid, onset")
        changes: Dict[int, Tuple[List[float], List[Optional[int]]]] = {}
        for melid, onset, chord in cur:
            onsets, values = changes.setdefault(melid, ([], []))
            onsets.append(onset)
            values.append(classify(chord))
        return changes
    finally:
        conn.close()


def _value_at(onset: float, onsets: List[float], values: List[Optional[int]]) -> Optional[int]:
    """The tagged value (quality or chord_idx, whichever `values` holds)
    active at `onset` -- the latest annotated change at or before it. None
    if `onset` precedes the first annotated change in this solo (e.g. a
    count-in bar) -- verified 3.9% of all 200,809 real notes fall here at
    the quality granularity, correctly excluded rather than mis-tagged."""
    idx = bisect.bisect_right(onsets, onset) - 1
    return values[idx] if idx >= 0 else None


def _split_into_runs(tagged_notes: list, key: str) -> List[Tuple[int, list]]:
    """Groups notes into maximal contiguous runs sharing the same tagged
    value at `key` -- (value, notes) pairs. Notes tagged None at `key` (no
    chord active yet) are dropped, not turned into a run -- they can't be
    attributed. Generic version of what was originally quality-specific
    (Phase 29); split_into_quality_runs/split_into_chord_runs below are thin
    wrappers naming which tag each expects."""
    runs: List[Tuple[int, list]] = []
    current_value: Optional[int] = None
    current_run: list = []
    for note in tagged_notes:
        v = note.get(key)
        if v is None:
            if current_run:
                runs.append((current_value, current_run))
                current_run = []
            current_value = None
            continue
        if v != current_value:
            if current_run:
                runs.append((current_value, current_run))
            current_value = v
            current_run = []
        current_run.append(note)
    if current_run:
        runs.append((current_value, current_run))
    return runs


def split_into_quality_runs(solo: list) -> List[Tuple[int, list]]:
    """Groups a chord_quality-tagged solo's notes into maximal contiguous
    same-quality runs -- (quality, notes) pairs. Verified directly against
    the real corpus: 18,944 runs, mean length 10.6 notes, 94.1% with >=2
    notes (the minimum extract_interval_motifs/extract_duration_motifs need
    to produce anything)."""
    return _split_into_runs(solo, "chord_quality")


def split_into_chord_runs(solo: list) -> List[Tuple[int, list]]:
    """Like split_into_quality_runs, but keyed on full chord_idx (Phase 30)
    instead of just quality -- a root change breaks a run even when quality
    doesn't, so these runs are finer-grained. Notes must come from
    iter_solos_with_chord_idx (tagged "chord_idx"), not iter_solos (tagged
    "chord_quality"). Verified directly against the real corpus: 26,357
    runs, mean length 7.6 notes, median 5, 93.3% with >=2 notes."""
    return _split_into_runs(solo, "chord_idx")


def _iter_solos_tagged(db_path: Path, classify, tag_key: str):
    """Shared streaming logic behind iter_solos/iter_solos_with_chord_idx --
    one pass over the melody table, tagging each note with whatever
    `classify` (applied to the active chord string) determines, stored under
    `tag_key`. Every note also carries its raw "onset" (seconds) and
    "beatdur" (seconds/beat) alongside "pitch"/"duration_beats" -- harmless
    extra keys for iter_solos's existing consumers (extract_interval_motifs/
    extract_duration_motifs only ever read "pitch"/"duration_beats"), and
    exactly what critic_baseline.py (Phase 30) needs to detect real gaps
    between notes in beat units."""
    changes = _changes_by_melid(db_path, classify)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT melid, onset, pitch, duration, beatdur FROM melody ORDER BY melid, onset")
        current_melid = None
        current_solo: list = []
        current_onsets: List[float] = []
        current_values: List[Optional[int]] = []
        for melid, onset, pitch, duration, beatdur in cur:
            if melid != current_melid:
                if current_solo:
                    yield current_solo
                current_melid = melid
                current_solo = []
                current_onsets, current_values = changes.get(melid, ([], []))
            v = _value_at(onset, current_onsets, current_values)
            current_solo.append({
                "pitch": int(round(pitch)), "duration_beats": duration / beatdur,
                "onset": onset, "beatdur": beatdur, tag_key: v,
            })
        if current_solo:
            yield current_solo
    finally:
        conn.close()


def iter_solos(db_path: Path):
    """Yields one list of {"pitch": int, "duration_beats": float,
    "chord_quality": Optional[int]} dicts per solo (melid), ordered by onset
    within each solo. chord_quality is the Wolfson QUAL_MAJOR/QUAL_DOM/
    QUAL_MINOR/QUAL_DIM class active at that note's onset, or None before
    that solo's first annotated chord change."""
    yield from _iter_solos_tagged(db_path, _wjd_chord_quality, "chord_quality")


def iter_solos_with_chord_idx(db_path: Path):
    """Like iter_solos, but tags each note with "chord_idx" (Phase 30, full
    root+quality) instead of just "chord_quality" -- needed by
    critic_baseline.py, whose critic functions depend on the actual root."""
    yield from _iter_solos_tagged(db_path, _wjd_chord_idx, "chord_idx")


def sample_quality_run(db_path: Path, n_notes: int = 8) -> Tuple[Optional[int], list]:
    """A real sample chunk (not synthetic) for the benchmark to look up
    against: the longest chord-quality run in the lowest-melid solo, capped
    to n_notes -- roughly the scale of one PhraseGenerator.generate() chunk
    (see ensemble/sax.py's plan_bars default). A targeted single-melid
    query, not iter_solos()/build_corpus(), so it doesn't pay the cost of
    tagging the whole 200k-row table just to pick one sample. Returns
    (None, []) if that solo has no classified run at all -- reported
    honestly rather than silently retried across solos."""
    conn = sqlite3.connect(str(db_path))
    try:
        melid = conn.execute("SELECT MIN(melid) FROM melody").fetchone()[0]
        note_rows = conn.execute(
            "SELECT onset, pitch, duration, beatdur FROM melody WHERE melid = ? ORDER BY onset", (melid,)
        ).fetchall()
        change_rows = conn.execute(
            "SELECT onset, chord FROM beats WHERE melid = ? AND chord != '' ORDER BY onset", (melid,)
        ).fetchall()
    finally:
        conn.close()
    onsets = [o for o, _ in change_rows]
    qualities = [_wjd_chord_quality(c) for _, c in change_rows]
    solo = [
        {"pitch": int(round(p)), "duration_beats": d / bd, "chord_quality": _value_at(onset, onsets, qualities)}
        for onset, p, d, bd in note_rows
    ]
    runs = split_into_quality_runs(solo)
    if not runs:
        return None, []
    quality, run = max(runs, key=lambda qr: len(qr[1]))
    return quality, run[:n_notes]


def build_corpus(db_path: Path):
    """Returns (pitch_counters, duration_counters, n_solos, n_notes), where
    the two counters are Dict[int, Counter] keyed by chord quality
    (QUAL_MAJOR..QUAL_DIM). Motifs are extracted per chord-quality-tagged
    RUN (split_into_quality_runs), never across a quality boundary -- the
    direct analogue of ensemble/sax.py's _bars_until_chord_change never
    letting one generation chunk span a chord change."""
    pitch_counters: Dict[int, Counter] = {q: Counter() for q in QUALITY_NAMES}
    duration_counters: Dict[int, Counter] = {q: Counter() for q in QUALITY_NAMES}
    n_solos = 0
    n_notes = 0
    for solo in iter_solos(db_path):
        n_solos += 1
        n_notes += len(solo)
        for quality, run in split_into_quality_runs(solo):
            pitch_counters[quality].update(extract_interval_motifs(run))
            duration_counters[quality].update(extract_duration_motifs(run))
    return pitch_counters, duration_counters, n_solos, n_notes


def save_cache(
    path: Path, pitch_counters: Dict[int, Counter], duration_counters: Dict[int, Counter], n_solos: int, n_notes: int
) -> None:
    """Write-then-atomic-replace, same pattern as ensemble/memory.py's
    RehearsalMemory._write -- a crash mid-write can never corrupt a prior
    cache. JSON has no tuple type (motifs) or int-keyed object (quality), so
    both round-trip explicitly: motifs as list<->tuple, quality as
    int<->str, same convention as RehearsalMemory._read/_write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "n_solos": n_solos,
        "n_notes": n_notes,
        "pitch_motifs": {
            str(q): [[list(motif), count] for motif, count in counter.items()] for q, counter in pitch_counters.items()
        },
        "duration_motifs": {
            str(q): [[list(motif), count] for motif, count in counter.items()]
            for q, counter in duration_counters.items()
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(raw))
    tmp.replace(path)


def load_cache(path: Path):
    raw = json.loads(path.read_text())
    pitch_counters = {
        int(q): Counter({tuple(motif): count for motif, count in entries}) for q, entries in raw["pitch_motifs"].items()
    }
    duration_counters = {
        int(q): Counter({tuple(motif): count for motif, count in entries})
        for q, entries in raw["duration_motifs"].items()
    }
    return pitch_counters, duration_counters, raw["n_solos"], raw["n_notes"]


def run_build() -> None:
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found -- see wjd_corpus.py's module docstring for how to obtain it.")
        return
    start = time.perf_counter()
    pitch_counters, duration_counters, n_solos, n_notes = build_corpus(DB_PATH)
    elapsed = time.perf_counter() - start
    save_cache(CACHE_PATH, pitch_counters, duration_counters, n_solos, n_notes)
    print(f"Built corpus from {n_solos} solos, {n_notes} notes.")
    for quality, name in QUALITY_NAMES.items():
        pc, dc = pitch_counters[quality], duration_counters[quality]
        print(
            f"  {name:10s}: {len(pc):6d} distinct pitch motifs ({sum(pc.values()):7d} occurrences), "
            f"{len(dc):6d} distinct duration motifs ({sum(dc.values()):7d} occurrences)"
        )
    print(f"Build time: {elapsed:.3f}s")
    print(f"Cached to {CACHE_PATH}")


def run_benchmark() -> None:
    if not CACHE_PATH.exists():
        print(f"{CACHE_PATH} not found -- run 'python wjd_corpus.py --build' first.")
        return
    if not DB_PATH.exists():
        print(f"{DB_PATH} not found -- benchmark needs it to build a real sample chunk to look up.")
        return
    pitch_counters, duration_counters, n_solos, n_notes = load_cache(CACHE_PATH)
    print(f"Loaded corpus: {n_solos} solos, {n_notes} notes.")
    for quality, name in QUALITY_NAMES.items():
        print(f"  {name:10s}: {len(pitch_counters[quality])} pitch motifs, {len(duration_counters[quality])} duration motifs")

    quality, sample_chunk = sample_quality_run(DB_PATH)
    if quality is None:
        print("The lowest-melid solo has no classified chord-quality run -- can't build a real sample chunk.")
        return
    candidate_pitch_motifs = extract_interval_motifs(sample_chunk)
    candidate_duration_motifs = extract_duration_motifs(sample_chunk)
    print(
        f"Sample chunk: {len(sample_chunk)} notes over a {QUALITY_NAMES[quality]} chord, "
        f"{len(candidate_pitch_motifs)} pitch motifs, {len(candidate_duration_motifs)} duration motifs to look up."
    )

    pitch_counter = pitch_counters[quality]
    duration_counter = duration_counters[quality]

    start = time.perf_counter()
    for _ in range(N_LOOKUPS):
        for motif in candidate_pitch_motifs:
            pitch_counter.get(motif, 0)
        for motif in candidate_duration_motifs:
            duration_counter.get(motif, 0)
    elapsed = time.perf_counter() - start

    total_lookups = N_LOOKUPS * (len(candidate_pitch_motifs) + len(candidate_duration_motifs))
    per_lookup_ms = elapsed * 1000 / total_lookups if total_lookups else 0.0
    print(
        f"{N_LOOKUPS} candidates x {len(candidate_pitch_motifs) + len(candidate_duration_motifs)} motifs each "
        f"({total_lookups} individual lookups): {elapsed * 1000:.3f}ms total, {per_lookup_ms:.5f}ms per lookup."
    )


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
