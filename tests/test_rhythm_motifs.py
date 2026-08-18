"""Tests for ensemble/rhythm_motifs.py (extract_duration_motifs) — pure logic,
no external data or model weights needed, same profile as
tests/test_memory.py's extract_interval_motifs tests."""

from ensemble.rhythm_motifs import extract_duration_motifs
from ensemble.wolfson.encoding import dur_to_token


def test_extract_duration_motifs_returns_all_2_3_4_grams():
    durations = [1.0, 0.5, 0.5, 0.25]
    tokens = [dur_to_token(d) for d in durations]
    phrase = [{"pitch": 60, "duration_beats": d} for d in durations]
    motifs = extract_duration_motifs(phrase)
    assert tuple(tokens[0:2]) in motifs
    assert tuple(tokens[1:3]) in motifs
    assert tuple(tokens[2:4]) in motifs
    assert tuple(tokens[0:3]) in motifs
    assert tuple(tokens[1:4]) in motifs
    assert tuple(tokens[0:4]) in motifs
    # 3 x 2-grams + 2 x 3-grams + 1 x 4-gram
    assert len(motifs) == 6


def test_extract_duration_motifs_too_short_returns_empty():
    assert extract_duration_motifs([{"pitch": 60, "duration_beats": 1.0}]) == []
    assert extract_duration_motifs([]) == []


def test_extract_duration_motifs_ignores_notes_without_duration_key():
    phrase = [{"pitch": 60}, {"pitch": 62, "duration_beats": 0.5}]
    # only one note actually has "duration_beats" -- same as <2 real durations
    assert extract_duration_motifs(phrase) == []


def test_extract_duration_motifs_is_pitch_invariant():
    """The rhythmic analogue of extract_interval_motifs' transposition-
    invariance test — but proving invariance to PITCH, not to transposition:
    the same rhythmic figure at different pitches produces the same
    duration-motif tuples."""
    durations = [0.5, 0.5, 1.0, 0.25]
    low = [{"pitch": p, "duration_beats": d} for p, d in zip((50, 53, 55, 57), durations)]
    high = [{"pitch": p, "duration_beats": d} for p, d in zip((74, 77, 79, 81), durations)]
    assert extract_duration_motifs(low) == extract_duration_motifs(high)


def test_extract_duration_motifs_distinguishes_different_rhythmic_figures():
    fast = [{"pitch": 60, "duration_beats": d} for d in (0.25, 0.25, 0.25)]
    slow = [{"pitch": 60, "duration_beats": d} for d in (2.0, 2.0, 2.0)]
    assert extract_duration_motifs(fast) != extract_duration_motifs(slow)
