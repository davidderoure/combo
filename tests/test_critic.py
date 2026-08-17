"""Tests for ensemble/critic.py — no sax_best.pt needed. Every function here is
pure and deterministic over already-generated data (no model inference), the
first sax-adjacent test file that's true of. Real end-to-end wiring (does a
generated chunk's score actually reach RehearsalMemory) is checked in
tests/test_sax_wolfson_integration.py, which needs the real weights."""

import pytest

from ensemble.critic import (
    DEFAULT_WEIGHTS,
    _contour_string,
    _levenshtein,
    call_response_relatedness,
    contour_smoothness,
    musicality_score,
    repetition,
    singability,
    tonal_conformity,
)
from ensemble.wolfson.phrase_generator import REST_PITCH

C_MAJOR = 0  # root=0 (C), quality_class=0 (QUAL_MAJOR) -> chord_idx = 0*4+0 = 0


def notes_from_pitches(pitches, duration_beats=1.0):
    return [{"pitch": p, "duration_beats": duration_beats, "velocity_scale": 1.0} for p in pitches]


# ---------------------------------------------------------------------------
# _levenshtein / _contour_string — the small helpers everything else builds on
# ---------------------------------------------------------------------------


def test_levenshtein_matches_known_distances():
    # Same values the notebook's own editdistance.eval sanity checks used.
    assert _levenshtein("UDU", "UDU") == 0
    assert _levenshtein("UDU", "UDS") == 1
    assert _levenshtein("UDU", "DUU") == 2
    assert _levenshtein("UDU", "UDUS") == 1
    assert _levenshtein("", "UDU") == 3
    assert _levenshtein("UDU", "") == 3


def test_contour_string_encodes_up_down_same():
    assert _contour_string([60, 62, 61, 61]) == "UDS"


# ---------------------------------------------------------------------------
# tonal_conformity
# ---------------------------------------------------------------------------


def test_tonal_conformity_scores_in_key_resolving_phrase_highly():
    # C, D, E (pcs 0,2,4) -- all in C-ionian scale, ends on E (a chord tone: root/maj3/maj7 = 0,4,11).
    notes = notes_from_pitches([60, 62, 64])
    assert tonal_conformity(notes, C_MAJOR) == 1.0


def test_tonal_conformity_scores_out_of_key_non_resolving_phrase_zero():
    # pcs 1, 3, 6 (C#, D#, F#) -- none in C-ionian, ends on pc 6, not a chord tone.
    notes = notes_from_pitches([61, 63, 66])
    assert tonal_conformity(notes, C_MAJOR) == 0.0


def test_tonal_conformity_no_real_notes_is_zero():
    assert tonal_conformity([{"pitch": REST_PITCH, "duration_beats": 1.0}], C_MAJOR) == 0.0


# ---------------------------------------------------------------------------
# contour_smoothness
# ---------------------------------------------------------------------------


def test_contour_smoothness_all_stepwise_is_one():
    notes = notes_from_pitches([60, 62, 63, 65])  # intervals 2,1,2 -- all small
    assert contour_smoothness(notes) == 1.0


def test_contour_smoothness_all_wide_leaps_is_zero():
    notes = notes_from_pitches([48, 60, 47, 61])  # intervals 12,-13,14 -- all leaps
    assert contour_smoothness(notes) == 0.0


def test_contour_smoothness_fewer_than_two_notes_is_vacuously_one():
    assert contour_smoothness(notes_from_pitches([60])) == 1.0
    assert contour_smoothness([]) == 1.0


# ---------------------------------------------------------------------------
# repetition
# ---------------------------------------------------------------------------


def test_repetition_exact_motif_repeat_scores_one():
    # intervals [2,2,-4,2,2] -- the 2-gram (2,2) recurs.
    notes = notes_from_pitches([60, 62, 64, 60, 62, 64])
    assert repetition(notes) == 1.0


def test_repetition_no_repeat_short_random_walk_scores_zero():
    notes = notes_from_pitches([60, 61, 63, 66, 70])  # intervals 1,2,3,4 -- all distinct n-grams
    assert repetition(notes) == 0.0


def test_repetition_fewer_than_three_notes_is_zero():
    assert repetition(notes_from_pitches([60, 62])) == 0.0


def test_repetition_detects_near_repeat_via_contour_edit_distance():
    """A passage whose exact interval n-grams never repeat, but whose U/D/S
    contour contains two windows only 1 edit apart -- the case exact
    extract_interval_motifs matching (Phase 11) misses and this phase's
    near-repeat detection catches. Contour = "UDUSUUDUU": window[0]="UDUS" vs
    window[5]="UDUU" differ by exactly one character."""
    pitches = [60, 62, 61, 64, 64, 70, 75, 73, 74, 75]
    assert _contour_string(pitches) == "UDUSUUDUU"
    notes = notes_from_pitches(pitches)
    assert repetition(notes) == 1.0


# ---------------------------------------------------------------------------
# call_response_relatedness
# ---------------------------------------------------------------------------


def test_call_response_relatedness_identical_shape_is_one():
    seed = notes_from_pitches([60, 62, 64])
    response = notes_from_pitches([48, 50, 52])  # same contour (U,U), different register
    assert call_response_relatedness(seed, response) == 1.0


def test_call_response_relatedness_opposite_shape_is_low():
    seed = notes_from_pitches([60, 62, 64, 66])  # U,U,U
    response = notes_from_pitches([66, 64, 62, 60])  # D,D,D
    assert call_response_relatedness(seed, response) == 0.0


def test_call_response_relatedness_both_shapeless_is_vacuously_one():
    assert call_response_relatedness(notes_from_pitches([60]), notes_from_pitches([70])) == 1.0


def test_call_response_relatedness_one_shapeless_is_zero():
    seed = notes_from_pitches([60])  # no contour (single note)
    response = notes_from_pitches([60, 62, 64])
    assert call_response_relatedness(seed, response) == 0.0


# ---------------------------------------------------------------------------
# singability
# ---------------------------------------------------------------------------


def test_singability_near_singable_center_scores_highly():
    notes = notes_from_pitches([60, 62, 64], duration_beats=0.95)  # SINGABLE_DUR_CENTER's ~beat value
    assert singability(notes) > 0.9


def test_singability_very_short_durations_score_lower():
    long_score = singability(notes_from_pitches([60, 62, 64], duration_beats=0.95))
    short_score = singability(notes_from_pitches([60, 62, 64], duration_beats=0.05))
    assert short_score < long_score


def test_singability_no_real_notes_is_zero():
    assert singability([{"pitch": REST_PITCH, "duration_beats": 1.0}]) == 0.0


# ---------------------------------------------------------------------------
# musicality_score combination
# ---------------------------------------------------------------------------


def test_musicality_score_overall_is_the_documented_weighted_sum():
    notes = notes_from_pitches([60, 62, 64, 60, 62, 64], duration_beats=0.95)
    seed = notes_from_pitches([60, 62, 64])
    score = musicality_score(notes, C_MAJOR, seed)

    expected_overall = (
        score.tonal_conformity * DEFAULT_WEIGHTS["tonal_conformity"]
        + score.contour_smoothness * DEFAULT_WEIGHTS["contour_smoothness"]
        + score.repetition * DEFAULT_WEIGHTS["repetition"]
        + score.call_response_relatedness * DEFAULT_WEIGHTS["call_response_relatedness"]
        + score.singability * DEFAULT_WEIGHTS["singability"]
    )
    assert score.overall == pytest.approx(expected_overall)
