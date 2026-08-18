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

What "worth remembering" means: as of Phase 12, quality-weighted, not pure
recency/frequency. store() takes an optional score (ensemble/critic.py's
MusicalityScore.overall — a real, if placeholder-tuned, critic, not a manual
constant); recall_motifs() weights each stored phrase's contribution by that
score instead of counting every phrase equally, so a high-scoring phrase's motifs
dominate recall over a low-scoring phrase's, even at equal repeat counts.
score defaults to 1.0 — calling store() without one (as every caller before
Phase 12 did) reproduces the exact old unweighted-count behaviour, no shim
needed. DESIGN.md §11's still-deferred batch-mode scoring remains a distinct,
larger idea (accumulating one signal across a whole song for curation) — this is
narrower: per-phrase quality weighting the moment a phrase is stored.

Recall is also chord-quality-aware, not just quality(-score)-weighted, since
Phase 25: store()/recall_motifs() take an optional chord_quality (Wolfson's
4-class major/dominant/minor/diminished system) so recall can mean "what
worked over a dominant chord" rather than pooling across every chord ever
played. Deliberately tagged by QUALITY, not root or full chord_idx — checked
directly before choosing this: extract_interval_motifs is already
transposition-invariant, so a shape that worked over one root of a quality is
exactly as valid, transposed, over any other root of the same quality; tagging
by root too would only fragment the buffer for no musical reason. Default
None on both methods reproduces the exact pre-Phase-25 quality-agnostic
pooling — no existing caller needs to change.
"""

from collections import Counter
from typing import List, Optional

from .wolfson.motifs import extract_interval_motifs
from .wolfson.phrase_generator import REST_PITCH

DEFAULT_MAX_PHRASES = 16


class RehearsalMemory:
    def __init__(self, max_phrases: int = DEFAULT_MAX_PHRASES):
        self._max_phrases = max_phrases
        self._phrases: List[dict] = []  # each entry: {"motifs": [...], "score": float}

    def store(self, notes: list, score: float = 1.0, chord_quality: Optional[int] = None) -> None:
        """notes: PhraseGenerator.generate()'s raw output. REST_PITCH sentinels are
        filtered before extraction — extract_interval_motifs is a plain pitch-
        sequence function, it doesn't know about wolfson's rest-sentinel
        convention. score: typically ensemble/critic.py's MusicalityScore.overall
        for this same phrase (see module docstring) — defaults to 1.0, matching
        every stored phrase counting equally, the pre-Phase-12 behaviour.

        chord_quality (Phase 25): Wolfson's 4-class quality system (QUAL_MAJOR/
        QUAL_DOM/QUAL_MINOR/QUAL_DIM, ensemble/wolfson/chords.py) the phrase was
        generated over — chosen over the full root+quality chord_idx because
        extract_interval_motifs is already transposition-invariant (the whole
        reason interval-shapes were chosen over absolute pitches): a shape that
        worked over one root of a quality is exactly as valid, transposed, over
        any other root of the same quality, so tagging by root too would only
        fragment an already-small buffer for no musical reason. Defaults to
        None (no chord context recorded) — see recall_motifs()."""
        real_notes = [n for n in notes if n.get("pitch") != REST_PITCH]
        self._phrases.append({
            "motifs": extract_interval_motifs(real_notes), "score": score, "chord_quality": chord_quality,
        })
        if len(self._phrases) > self._max_phrases:
            self._phrases.pop(0)

    def recall_motifs(self, n_recent: int = DEFAULT_MAX_PHRASES, chord_quality: Optional[int] = None) -> Counter:
        """Counter of interval-motif tuples seen across the last n_recent stored
        phrases, weighted by each phrase's score — most_common() gives a caller
        something to lean toward next, favouring motifs from higher-scoring
        phrases over merely-frequent ones.

        chord_quality (Phase 25): if given, restricts recall to phrases stored
        under that SAME quality — "what worked over a dominant chord", not
        "what worked anywhere". Untagged phrases (chord_quality=None at store
        time) never match a specific filter here — no chord context was
        recorded for them, so they can't honestly be claimed to fit one; they
        still count in the default (chord_quality=None) quality-agnostic pool.
        Default None reproduces exactly the pre-Phase-25 quality-agnostic
        pooling for every existing caller — strict filtering, deliberately no
        cross-quality fallback when the requested quality has no history yet
        (the smallest correct thing; a blended fallback is a real, separate
        future refinement, not attempted here)."""
        counter: Counter = Counter()
        for entry in self._phrases[-n_recent:]:
            if chord_quality is not None and entry.get("chord_quality") != chord_quality:
                continue
            for motif in entry["motifs"]:
                counter[motif] += entry["score"]
        return counter
