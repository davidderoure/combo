"""CorpusMotifs — the runtime-consumable half of Phase 29's WJD corpus work,
DESIGN.md §13.

Loads a cache built by wjd_corpus.py --build (per-chord-quality pitch- and
duration-motif frequency tables extracted from the Weimar Jazz Database) and
exposes O(1) chord-quality-aware lookups. The corpus-side counterpart to
ensemble/memory.py's RehearsalMemory, but built once OFFLINE from a fixed
external dataset rather than accumulated live during a performance —
read-only, no persist_path/write path, corpus data doesn't change at
runtime.

A missing cache file raises naturally (FileNotFoundError), unlike
RehearsalMemory's optional persist_path (where a missing file just means
"first rehearsal," a normal case): a missing corpus cache means "not built
yet," a real error the caller should check for and report clearly (see
self_test.py's --corpus flag) rather than have this class swallow silently.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict


class CorpusMotifs:
    def __init__(self, cache_path: Path):
        raw = json.loads(cache_path.read_text())
        self._pitch_counters: Dict[int, Counter] = {
            int(q): Counter({tuple(motif): count for motif, count in entries}) for q, entries in raw["pitch_motifs"].items()
        }
        self._duration_counters: Dict[int, Counter] = {
            int(q): Counter({tuple(motif): count for motif, count in entries})
            for q, entries in raw["duration_motifs"].items()
        }

    def has_pitch_motif(self, motif: tuple, chord_quality: int) -> bool:
        counter = self._pitch_counters.get(chord_quality)
        return counter is not None and motif in counter

    def has_duration_motif(self, motif: tuple, chord_quality: int) -> bool:
        counter = self._duration_counters.get(chord_quality)
        return counter is not None and motif in counter
