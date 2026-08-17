"""Rehearsal memory — persistence across performances, DESIGN.md §12, Phase 11.

The first thing in combo that persists across separate Session.generate() calls.
Every other piece of state is deliberately fresh-per-performance
(ensemble/transitions.py's TransitionController: "each call is a fresh
performance, so fresh transition state"; ensemble/sax.py's plan buffer resets with
a new Session). RehearsalMemory is the opposite on purpose: construct one, pass it
into multiple Session/sax_generator calls (rehearsals), then into a final one (the
gig) — nothing resets it automatically. A fresh RehearsalMemory is a fresh
rehearsal, by construction, not by a reset() method to remember to call.

Inspired by wolfson's memory/phrase_memory.py (PhraseMemory) — same store/recall
shape (a capped buffer, motifs pulled out via a Counter) — but re-authored, not
ported: PhraseMemory resets between ArcController's 5-minute arc loops within one
live performance, which is a different lifecycle than "persist across separate
rehearsals of the same piece." The source-filtering/recall_random/recall_early
machinery PhraseMemory has for its bass-vs-sax call-response setup isn't needed for
this MVP's single-voice case either.

What "worth remembering" means in this MVP, stated as a deliberate simplification,
not a gap: no evaluation of quality. recall_motifs()'s most_common(1) just picks
whichever interval pattern has recurred most often across recently stored phrases
— on a single freshly-generated chunk that's usually just "whatever short shape
happened to repeat," not "the good bit." An actual critic that judges what's worth
carrying forward is real, separate work (DESIGN.md §11's still-deferred batch-mode
scoring is the natural home for it) — this phase ships the plumbing (does persisted
material reach and influence generation?) before the judgment (is it good?), same
order every earlier phase built in.
"""

from collections import Counter
from typing import List

from .wolfson.motifs import extract_interval_motifs
from .wolfson.phrase_generator import REST_PITCH

DEFAULT_MAX_PHRASES = 16


class RehearsalMemory:
    def __init__(self, max_phrases: int = DEFAULT_MAX_PHRASES):
        self._max_phrases = max_phrases
        self._phrases: List[list] = []  # each entry: extracted motifs for one stored phrase

    def store(self, notes: list) -> None:
        """notes: PhraseGenerator.generate()'s raw output. REST_PITCH sentinels are
        filtered before extraction — extract_interval_motifs is a plain pitch-
        sequence function, it doesn't know about wolfson's rest-sentinel
        convention."""
        real_notes = [n for n in notes if n.get("pitch") != REST_PITCH]
        self._phrases.append(extract_interval_motifs(real_notes))
        if len(self._phrases) > self._max_phrases:
            self._phrases.pop(0)

    def recall_motifs(self, n_recent: int = DEFAULT_MAX_PHRASES) -> Counter:
        """Counter of interval-motif tuples seen across the last n_recent stored
        phrases — most_common() gives a caller something to lean toward next."""
        counter: Counter = Counter()
        for motifs in self._phrases[-n_recent:]:
            counter.update(motifs)
        return counter
