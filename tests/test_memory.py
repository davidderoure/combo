"""Tests for ensemble/memory.py (RehearsalMemory) and the ported
ensemble/wolfson/motifs.py (extract_interval_motifs) — no sax_best.pt needed,
pure logic only. ensemble/critic.py's musicality scoring (Phase 12) has its own
test file, tests/test_critic.py — this file only tests that RehearsalMemory
correctly WEIGHTS by whatever score it's given, not how that score is computed.
Real-inference wiring (does a generated chunk's real score actually reach
memory, and does memory's content reach PhraseGenerator.generate()'s
motif_targets/motif_strength) is in tests/test_sax_wolfson_integration.py, which
needs the real weights."""

from ensemble.memory import RehearsalMemory
from ensemble.wolfson.motifs import extract_interval_motifs
from ensemble.wolfson.phrase_generator import REST_PITCH


def test_extract_interval_motifs_returns_all_2_3_4_grams():
    # pitches [62, 65, 67, 69] -> intervals [3, 2, 2]
    phrase = [{"pitch": p} for p in (62, 65, 67, 69)]
    motifs = extract_interval_motifs(phrase)
    assert (3, 2) in motifs
    assert (2, 2) in motifs
    assert (3, 2, 2) in motifs
    assert len(motifs) == 3  # 2-grams (3,2),(2,2) + one 3-gram (3,2,2); no 4-gram (only 3 intervals)


def test_extract_interval_motifs_is_transposition_invariant():
    low = [{"pitch": p} for p in (50, 53, 55, 57)]
    high = [{"pitch": p} for p in (74, 77, 79, 81)]  # same shape, +24 semitones
    assert extract_interval_motifs(low) == extract_interval_motifs(high)


def test_extract_interval_motifs_too_short_returns_empty():
    assert extract_interval_motifs([{"pitch": 60}]) == []
    assert extract_interval_motifs([]) == []


def test_rehearsal_memory_recall_counts_across_stored_phrases():
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # motifs: (2,), wait -- 2-grams need len>=2 intervals
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # same shape again
    counts = mem.recall_motifs()
    assert counts[(2, 2)] == 2  # seen in both stored phrases


def test_rehearsal_memory_filters_rest_sentinels_before_extraction():
    mem = RehearsalMemory()
    notes = [
        {"pitch": 60},
        {"pitch": REST_PITCH},
        {"pitch": 62},
        {"pitch": 64},
    ]
    mem.store(notes)
    # Without filtering, REST_PITCH (-1) would corrupt the interval sequence.
    counts = mem.recall_motifs()
    assert counts[(2, 2)] == 1  # (60,62,64) -> intervals (2,2), rest ignored


def test_rehearsal_memory_caps_stored_phrases():
    mem = RehearsalMemory(max_phrases=2)
    mem.store([{"pitch": p} for p in (60, 61, 62)])  # motif (1,1) -- should be evicted
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # motif (2,2)
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # motif (2,2) again
    counts = mem.recall_motifs()
    assert counts[(1, 1)] == 0  # evicted -- only max_phrases=2 most recent kept
    assert counts[(2, 2)] == 2


def test_rehearsal_memory_recall_with_no_stored_phrases_is_empty():
    mem = RehearsalMemory()
    assert mem.recall_motifs() == {}


def test_rehearsal_memory_weights_recall_by_score_not_just_count():
    """Phase 12: a high-scoring phrase's motifs should outweigh a low-scoring
    phrase's, even though both were stored exactly once (equal repeat count) --
    the concrete behaviour that closes the "no evaluation of what's worth
    remembering" gap Phase 11 named."""
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 63, 67)], score=0.1)  # motif (3,4), low score
    mem.store([{"pitch": p} for p in (60, 62, 64)], score=0.9)  # motif (2,2), high score
    counts = mem.recall_motifs()
    assert counts[(2, 2)] > counts[(3, 4)]
    assert counts.most_common(1)[0][0] == (2, 2)


def test_rehearsal_memory_store_score_defaults_to_one():
    """No score given -> equivalent to every phrase counting equally, the
    pre-Phase-12 behaviour every existing test above still exercises."""
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)])
    counts = mem.recall_motifs()
    assert counts[(2, 2)] == 1.0


# ---------------------------------------------------------------------------
# chord-tagged recall (Phase 25)
# ---------------------------------------------------------------------------


def test_recall_motifs_with_chord_quality_returns_only_that_quality():
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)], chord_quality=1)  # motif (2,2), dominant
    mem.store([{"pitch": p} for p in (60, 63, 67)], chord_quality=2)  # motif (3,4), minor
    dom_counts = mem.recall_motifs(chord_quality=1)
    assert dom_counts[(2, 2)] == 1
    assert dom_counts[(3, 4)] == 0  # the minor-tagged phrase's motif does not leak in


def test_recall_motifs_with_chord_quality_does_not_leak_across_qualities():
    """The core 'context-aware, not global pooling' proof: two qualities, each
    recalled separately, never see the other's motifs."""
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)], chord_quality=1)  # (2,2), dominant
    mem.store([{"pitch": p} for p in (60, 61, 62)], chord_quality=2)  # (1,1), minor
    assert mem.recall_motifs(chord_quality=1) == {(2, 2): 1.0}
    assert mem.recall_motifs(chord_quality=2) == {(1, 1): 1.0}


def test_recall_motifs_without_chord_quality_still_pools_everything():
    """Default (chord_quality=None) reproduces exactly the pre-Phase-25
    quality-agnostic pooling, regardless of what tags stored phrases carry --
    the regression check for every existing caller that never passes
    chord_quality at all."""
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)], chord_quality=1)
    mem.store([{"pitch": p} for p in (60, 61, 62)], chord_quality=2)
    counts = mem.recall_motifs()
    assert counts[(2, 2)] == 1.0
    assert counts[(1, 1)] == 1.0


def test_recall_motifs_with_chord_quality_excludes_untagged_phrases():
    """An untagged phrase (no chord_quality recorded) is not a wildcard -- it
    can't honestly be claimed to fit a specific quality it was never tagged
    with, so a quality-SPECIFIC recall excludes it, even though it still
    counts in the quality-agnostic pool (default recall_motifs())."""
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # untagged, motif (2,2)
    assert mem.recall_motifs(chord_quality=1) == {}
    assert mem.recall_motifs()[(2, 2)] == 1.0


def test_recall_motifs_with_chord_quality_and_no_matching_history_is_empty():
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)], chord_quality=1)
    assert mem.recall_motifs(chord_quality=3) == {}  # quality 3 never stored


# ---------------------------------------------------------------------------
# disk persistence (Phase 26) -- pytest's tmp_path fixture, no weights needed
# ---------------------------------------------------------------------------


def test_persist_path_not_yet_existing_starts_empty(tmp_path):
    """A fresh chart's first rehearsal has nothing to load -- a normal case,
    not an error."""
    mem = RehearsalMemory(persist_path=tmp_path / "new_chart.json")
    assert mem.recall_motifs() == {}


def test_store_with_persist_path_round_trips_to_a_second_independent_instance(tmp_path):
    """The core proof: persistence across separate OBJECTS, not just within
    one -- construct, store, then build a second, independent RehearsalMemory
    pointed at the same path and confirm it recalls what the first stored,
    including a chord_quality-specific recall."""
    path = tmp_path / "blues_in_f.json"
    first = RehearsalMemory(persist_path=path)
    first.store([{"pitch": p} for p in (60, 62, 64)], score=0.8, chord_quality=1)

    second = RehearsalMemory(persist_path=path)
    assert second.recall_motifs()[(2, 2)] == 0.8
    assert second.recall_motifs(chord_quality=1)[(2, 2)] == 0.8
    assert second.recall_motifs(chord_quality=2) == {}


def test_a_single_store_call_is_already_on_disk(tmp_path):
    """Crash-safety: saving isn't deferred to the end of a run -- after just
    ONE store() call, a second independent instance already sees it (the
    concrete proof a Ctrl-C right after this point wouldn't lose it)."""
    path = tmp_path / "blues_in_f.json"
    mem = RehearsalMemory(persist_path=path)
    mem.store([{"pitch": p} for p in (60, 61, 62)])  # motif (1,1)
    reloaded = RehearsalMemory(persist_path=path)
    assert reloaded.recall_motifs()[(1, 1)] == 1.0


def test_max_phrases_cap_holds_across_a_reload(tmp_path):
    """Store more than max_phrases across several independent instances
    sharing one path -- oldest-first eviction must survive the round trip,
    not just hold in a single in-memory session."""
    path = tmp_path / "blues_in_f.json"
    mem = RehearsalMemory(max_phrases=2, persist_path=path)
    mem.store([{"pitch": p} for p in (60, 61, 62)])  # (1,1) -- should end up evicted
    mem = RehearsalMemory(max_phrases=2, persist_path=path)  # simulates a fresh process
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # (2,2)
    mem = RehearsalMemory(max_phrases=2, persist_path=path)
    mem.store([{"pitch": p} for p in (60, 62, 64)])  # (2,2) again

    final = RehearsalMemory(max_phrases=2, persist_path=path)
    counts = final.recall_motifs()
    assert counts[(1, 1)] == 0  # evicted
    assert counts[(2, 2)] == 2


def test_persist_path_none_never_touches_disk(monkeypatch):
    """Default behaviour (no persist_path) must reproduce the exact
    pre-Phase-26 in-memory-only behaviour -- zero disk I/O. Checked directly,
    not inferred: _write would raise if it were ever called, the concrete
    proof for every existing caller that never passes persist_path at all."""
    def fail_if_called(self, path):
        raise AssertionError("_write must never be called when persist_path is None")

    monkeypatch.setattr(RehearsalMemory, "_write", fail_if_called)
    mem = RehearsalMemory()
    mem.store([{"pitch": p} for p in (60, 62, 64)])
    mem.store([{"pitch": p} for p in (60, 61, 62)])
