"""MarkovTables — the runtime-consumable half of Phase 35's Markov-chain sax
generator, DESIGN.md §13.

Loads a cache built by markov_corpus.py --build (quality-conditioned,
order-N pitch-interval and duration-token transition tables extracted from
the Weimar Jazz Database) and exposes weighted sampling. Mirrors
ensemble/corpus_motifs.py's CorpusMotifs — read-only, built once offline
from a fixed external dataset, not accumulated live.

Falls back to a quality's MARGINAL (context-free) distribution when a
specific context was never observed in training -- a real, defined case
(not a crash), though empirically rare at order-1 (checked directly against
the real corpus: 15-26% of contexts are singletons, but the vast majority
of actual transitions still land on well-populated contexts).
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict


class MarkovTables:
    def __init__(self, cache_path: Path):
        raw = json.loads(cache_path.read_text())
        self.order: int = raw["order"]
        self._pitch_transitions = self._unflatten_transitions(raw["pitch_transitions"])
        self._duration_transitions = self._unflatten_transitions(raw["duration_transitions"])
        self._pitch_marginals = self._unflatten_marginal(raw["pitch_marginals"])
        self._duration_marginals = self._unflatten_marginal(raw["duration_marginals"])

    @staticmethod
    def _unflatten_transitions(raw_table: dict) -> Dict[int, Dict[tuple, Counter]]:
        return {
            int(q): {tuple(ctx): Counter({int(k): v for k, v in outcomes.items()}) for ctx, outcomes in entries}
            for q, entries in raw_table.items()
        }

    @staticmethod
    def _unflatten_marginal(raw_marginal: dict) -> Dict[int, Counter]:
        return {int(q): Counter({int(k): v for k, v in counter.items()}) for q, counter in raw_marginal.items()}

    def sample_next_interval(self, quality: int, context: tuple, rng: random.Random) -> int:
        return self._sample(self._pitch_transitions, self._pitch_marginals, quality, context, rng)

    def sample_next_duration_token(self, quality: int, context: tuple, rng: random.Random) -> int:
        return self._sample(self._duration_transitions, self._duration_marginals, quality, context, rng)

    @staticmethod
    def _sample(transitions, marginals, quality: int, context: tuple, rng: random.Random) -> int:
        counter = transitions.get(quality, {}).get(context)
        if not counter:
            counter = marginals.get(quality)
        if not counter:
            raise ValueError(f"MarkovTables has no data at all for quality {quality!r} -- empty or corrupt cache")
        outcomes = list(counter.keys())
        weights = list(counter.values())
        return rng.choices(outcomes, weights=weights, k=1)[0]
