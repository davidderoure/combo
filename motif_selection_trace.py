"""Drills down on sax_generator's in-flight candidate search (DESIGN.md §13's
"chess" idea) specifically for chunks where RehearsalMemory recalled a motif
target -- answers David's own question directly: "where does the recalled
motif sit in the candidate futures?"

    python motif_selection_trace.py                       # ii_v_i.chart, one run
    python motif_selection_trace.py --chart songs/x.chart --seed 4
    python motif_selection_trace.py --max-chunks-shown 10

The selection key sax_generator actually uses is lexicographic:
(-dissonance, motif_adherence, overall) -- dissonance-avoidance is checked
FIRST, motif adherence only breaks ties among candidates that are equally
(un)dissonant, and overall is the final tie-break. That means a candidate
that ACHIEVES the recalled motif does not automatically win: if a different,
non-matching candidate has strictly lower dissonance, the non-matching one
wins regardless of motif_adherence. This script makes that mechanic visible
by reconstructing, for every candidate generated in a recall-active chunk,
exactly the same key sax_generator computed -- not a re-derivation, the SAME
computation (ensemble/critic.py's dissonance/motif_adherence/musicality_score,
with the exact same extra_tolerated context via _functional_tonic_scale) --
and printing the full ranking, not just the winner.

Grouping candidates into chunks: consecutive PhraseGenerator.generate() calls
sharing the same (chord_idx, motif_targets) are treated as one chunk's
candidate batch -- true by construction (both are computed once per chunk,
before its candidate loop starts) unless two genuinely separate chunks happen
to share both values back-to-back, which doesn't happen on ii_v_i.chart (the
chord changes every bar, so consecutive chunks never share chord_idx).
bar_start is tracked by accumulating each chunk's own max_phrase_beats, so
_functional_tonic_scale's lookup is exact, not approximated.
"""

import argparse
from pathlib import Path

import ensemble.wolfson.phrase_generator as wolfson_phrase_generator
from ensemble import MACHINE_SPEED, Session, Voice, chord_tone_generator
from ensemble.critic import dissonance, motif_adherence, musicality_score
from ensemble.memory import RehearsalMemory
from ensemble.sax import _functional_tonic_scale, sax_generator
from song import parse_chart

DEFAULT_CHART = Path(__file__).resolve().parent / "songs" / "ii_v_i.chart"
BASS_REGISTER = (28, 52)
SAX_REGISTER = (55, 79)


def note_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--motif-recall-candidates", type=int, default=20)
    parser.add_argument("--max-chunks-shown", type=int, default=6)
    args = parser.parse_args()

    song = parse_chart(args.chart.read_text())
    memory = RehearsalMemory()

    original = wolfson_phrase_generator.PhraseGenerator.generate
    chunks = []  # each: {"chord_idx", "targets", "bar_start", "candidates": [(notes, seed_phrase)]}
    bar_cursor = [0.0]

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        targets = kwargs.get("motif_targets") or []
        is_new_chunk = (
            not chunks or chunks[-1]["chord_idx"] != kwargs["chord_idx"] or chunks[-1]["targets"] != targets
        )
        if is_new_chunk:
            if chunks:
                bar_cursor[0] += chunks[-1]["max_phrase_beats"]
            chunks.append({
                "chord_idx": kwargs["chord_idx"], "targets": targets, "bar_start": bar_cursor[0],
                "max_phrase_beats": kwargs["max_phrase_beats"], "candidates": [],
            })
        chunks[-1]["candidates"].append((notes, seed_phrase))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai",
                     generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(
            SAX_REGISTER, target_voice_id="bass", memory=memory, seed=args.seed,
            n_candidates=args.n_candidates, motif_recall_candidates=args.motif_recall_candidates,
        )
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    recall_chunks = [c for c in chunks if c["targets"]]
    print(f"{len(chunks)} total chunks, {len(recall_chunks)} with a recalled motif target "
          f"(seed={args.seed}, n_candidates={args.n_candidates}, "
          f"motif_recall_candidates={args.motif_recall_candidates})\n")

    winner_matches_count = 0
    matching_but_lost = 0
    no_candidate_matched = 0
    match_fractions = []

    for i, chunk in enumerate(recall_chunks):
        chord_idx = chunk["chord_idx"]
        target = chunk["targets"][0]
        extra_tolerated = _functional_tonic_scale(song, chunk["bar_start"])

        rows = []
        for notes, seed_phrase in chunk["candidates"]:
            matches = motif_adherence(notes, chunk["targets"]) > 0
            d = dissonance(notes, chord_idx, extra_tolerated=extra_tolerated)
            score = musicality_score(notes, chord_idx, seed_phrase, SAX_REGISTER)
            key = (-d, 1.0 if matches else 0.0, score.overall)
            rows.append({"matches": matches, "dissonance": d, "overall": score.overall, "key": key})

        n = len(rows)
        n_matching = sum(1 for r in rows if r["matches"])
        match_fractions.append(n_matching / n)
        best_key = max(r["key"] for r in rows)
        winner_idx = [r["key"] for r in rows].index(best_key)
        winner_matches = rows[winner_idx]["matches"]

        if winner_matches:
            winner_matches_count += 1
        elif n_matching > 0:
            matching_but_lost += 1
        else:
            no_candidate_matched += 1

        if i < args.max_chunks_shown:
            print(f"--- chunk {i}: chord_idx={chord_idx}  target motif={target}  "
                  f"({n} candidates, {n_matching} achieve it) ---")
            ranked = sorted(range(n), key=lambda idx: rows[idx]["key"], reverse=True)
            for rank, idx in enumerate(ranked):
                r = rows[idx]
                marker = "  <- WINNER" if idx == winner_idx else ""
                print(f"  #{rank:2d} (candidate {idx:2d}): matches_target={r['matches']!s:5}  "
                      f"dissonance={r['dissonance']:.3f}  overall={r['overall']:.4f}{marker}")
            print()

    print("=== Summary across all recall-active chunks ===")
    print(f"Chunks where the WINNER achieved the recalled motif:        "
          f"{winner_matches_count}/{len(recall_chunks)}")
    print(f"Chunks where a candidate achieved it but a LOWER-DISSONANCE, "
          f"non-matching candidate won instead: {matching_but_lost}/{len(recall_chunks)}")
    print(f"Chunks where NO candidate achieved it at all:                "
          f"{no_candidate_matched}/{len(recall_chunks)}")
    avg_fraction = sum(match_fractions) / len(match_fractions) if match_fractions else 0.0
    print(f"Average fraction of candidates per recall-chunk that achieve the target motif: "
          f"{avg_fraction:.1%}")


if __name__ == "__main__":
    main()
