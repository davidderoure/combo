"""Tests for ensemble/memory.py (RehearsalMemory) and the ported
ensemble/wolfson/motifs.py (extract_interval_motifs) — no sax_best.pt needed,
pure logic only. Real-inference wiring (does memory's content actually reach
PhraseGenerator.generate()'s motif_targets/motif_strength) is in
tests/test_sax_wolfson_integration.py, which needs the real weights."""

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
