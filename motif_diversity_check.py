"""How many distinct recalled motifs a single performance actually targets,
under self_test.py's real settings -- the direct instrumentation David's
listtest4.mid feedback asked for ("I would be happy if there was more motif
repetition to give a sense of development through the phrases"), since a
static MIDI recording can't see inside RehearsalMemory at all.

    python motif_diversity_check.py                    # 10 takes, blues_in_f.chart
    python motif_diversity_check.py --takes 20

Real probe that shaped Phase 45 (one seeded take, blues_in_f.chart,
n_candidates=8, motif_recall_candidates=20 -- self_test.py's own real
settings): only 3 distinct motifs were EVER targeted across the whole
45-chunk performance -- (0, -1) for 26 CONSECUTIVE chunks, then (-1, 1) for
the remaining 17. Traced to a genuine positive feedback loop, not simply "no
decay": RehearsalMemory already caps stored phrases at DEFAULT_MAX_PHRASES
(16, a bounded window), but a winning motif keeps getting echoed by
_apply_motif_bias, which re-stores it right back into the window every
chunk it wins -- the window can only forget content that stops recurring,
and a self-reinforcing winner never stops recurring.

Fixed via a per-motif cooldown (ensemble/sax.py's motif_streak/
MOTIF_STREAK_LIMIT): once a motif has been picked MOTIF_STREAK_LIMIT times
in a row, it's excluded from the next pick, forcing a genuinely different
motif to be tried. Re-running the same probe after the fix: 6 distinct
motifs across the same performance, with real rotation between them --
report the real, current numbers below, not the prototype's.
"""

import argparse
from collections import Counter
from pathlib import Path

from ensemble import MACHINE_SPEED, Session, Voice
from ensemble.memory import RehearsalMemory
from ensemble.sax import MOTIF_STREAK_LIMIT, sax_generator
from song import parse_chart

import self_test as st


def longest_streak(picks) -> int:
    longest = 0
    current, count = None, 0
    for p in picks:
        if p is not None and p == current:
            count += 1
        else:
            current, count = p, (1 if p is not None else 0)
        longest = max(longest, count)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=Path, default=st.DEFAULT_CHART)
    parser.add_argument("--takes", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--motif-recall-candidates", type=int, default=20)
    args = parser.parse_args()

    song = parse_chart(args.chart.read_text())

    distinct_counts = []
    streak_counts = []
    target_fractions = []

    for _ in range(args.takes):
        memory = RehearsalMemory()
        bass = Voice(id="bass", instrument="bass", register=st.BASS_REGISTER, source="ai",
                     generator=st.walking_bass_stub(st.BASS_REGISTER))
        sax_gen = sax_generator(st.SAX_REGISTER, target_voice_id="bass", memory=memory,
                                 n_candidates=args.n_candidates,
                                 motif_recall_candidates=args.motif_recall_candidates)
        sax = Voice(id="sax", instrument="sax", register=st.SAX_REGISTER, source="ai", generator=sax_gen)
        Session(song=song, voices=[bass, sax]).generate(mode=MACHINE_SPEED)

        picks = sax_gen.motif_target_log
        distinct = {p for p in picks if p is not None}
        distinct_counts.append(len(distinct))
        streak_counts.append(longest_streak(picks))
        target_fractions.append(sum(1 for p in picks if p is not None) / len(picks) if picks else 0.0)

    n = len(distinct_counts)
    print(f"Motif diversity over {n} takes ({args.chart.name}, n_candidates={args.n_candidates}, "
          f"motif_recall_candidates={args.motif_recall_candidates}, MOTIF_STREAK_LIMIT={MOTIF_STREAK_LIMIT})")
    print(f"  distinct motifs targeted per take: mean {sum(distinct_counts) / n:.2f}, "
          f"min {min(distinct_counts)}, max {max(distinct_counts)}")
    print(f"  longest same-motif streak per take: mean {sum(streak_counts) / n:.2f}, "
          f"min {min(streak_counts)}, max {max(streak_counts)} (cap: {MOTIF_STREAK_LIMIT})")
    print(f"  fraction of chunks with a non-empty target: mean "
          f"{sum(target_fractions) / n:.3f}")
    print(f"\nPer-take distinct-motif counts: {distinct_counts}")
    print(f"Per-take longest streaks: {Counter(streak_counts)}")


if __name__ == "__main__":
    main()
