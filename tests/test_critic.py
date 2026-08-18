"""Tests for ensemble/critic.py — no sax_best.pt needed. Every function here is
pure and deterministic over already-generated data (no model inference), the
first sax-adjacent test file that's true of. Real end-to-end wiring (does a
generated chunk's score actually reach RehearsalMemory) is checked in
tests/test_sax_wolfson_integration.py, which needs the real weights."""

import pytest

from ensemble.critic import (
    DEFAULT_WEIGHTS,
    _contour_string,
    _is_passing_tone,
    _levenshtein,
    _semitones_to_scale,
    _widened_mode_scale,
    call_response_relatedness,
    contour_smoothness,
    dissonance,
    dissonance_scale,
    motif_adherence,
    musicality_score,
    repetition,
    singability,
    tonal_conformity,
)
from ensemble.wolfson.chords import QUAL_DIM, QUAL_DOM, QUAL_MINOR
from ensemble.wolfson.phrase_generator import REST_PITCH
from ensemble.wolfson.scales import chord_root, chord_to_mode, scale_pitch_classes

C_MAJOR = 0  # root=0 (C), quality_class=0 (QUAL_MAJOR) -> chord_idx = 0*4+0 = 0
F_DOM = 5 * 4 + QUAL_DOM  # root=F(5) -- matches ensemble/sax.py's chord_idx mapping
D_MINOR = 2 * 4 + QUAL_MINOR
G_DIM = 7 * 4 + QUAL_DIM


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
# dissonance -- higher is WORSE, unlike every other function here. A badness
# signal for ensemble/sax.py's selection to minimise, not a goodness signal
# blended into MusicalityScore/DEFAULT_WEIGHTS.
# ---------------------------------------------------------------------------


def test_semitones_to_scale_computes_shortest_wrap_around_distance():
    # A synthetic wide-gap scale (not one any real chord_to_mode reaches) --
    # proves the underlying distance math directly, independent of whether
    # combo's actual modes ever produce a gap this wide (they don't: ionian/
    # mixolydian/dorian/diminished, the only 4 chord_to_mode reaches, all
    # have a max gap of 2 semitones, so every out-of-scale note in real use
    # is always exactly 1 semitone away -- see dissonance()'s own docstring).
    scale = frozenset({0, 5})
    assert _semitones_to_scale(0, scale) == 0
    assert _semitones_to_scale(1, scale) == 1  # 1 semitone from 0
    assert _semitones_to_scale(3, scale) == 2  # 2 from 5, 3 from 0 -- shorter side wins
    assert _semitones_to_scale(11, scale) == 1  # wraps: 1 semitone from 0


def test_dissonance_no_real_notes_is_zero():
    assert dissonance([{"pitch": REST_PITCH, "duration_beats": 1.0}], C_MAJOR) == 0.0


def test_dissonance_all_in_scale_is_zero():
    notes = notes_from_pitches([60, 62, 64])  # C, D, E -- all in C-ionian
    assert dissonance(notes, C_MAJOR) == 0.0


def test_dissonance_semitone_clash_counts_as_dissonant():
    # C#4 (pc 1) is 1 semitone from both C (0) and D (2) -- the "minor 9th"
    # clash David flagged as the worst case in ordinary melodic playing.
    notes = notes_from_pitches([61])
    assert dissonance(notes, C_MAJOR) == 1.0


def test_dissonance_is_the_fraction_of_clashing_notes():
    # C (in scale), C# (clash -- leapt away to G next, not a passing tone),
    # G (in scale), D# (clash -- last note, nothing to pass into) -- 2 of 4
    # clash. Deliberately NOT a stepwise run -- see the _is_passing_tone tests
    # below for that case specifically. Uses C# and D#, not G#: G# is inside
    # bebop_major's widened C-major reference (Phase 20's own b6 passing
    # tone), so it wouldn't count as a clash at all -- see dissonance_scale.
    notes = notes_from_pitches([60, 61, 67, 63])
    assert dissonance(notes, C_MAJOR) == 0.5


# ---------------------------------------------------------------------------
# dissonance_scale -- Phase 20, Lever A: widen the reference for dominant/
# major chords to include a named bebop passing tone; minor/diminished are a
# deliberate scope-cut, not an oversight.
# ---------------------------------------------------------------------------


def test_dissonance_scale_dominant_includes_the_bebop_maj7_passing_tone():
    # F7: plain mixolydian is {5,7,9,10,0,2,3}; bebop_dom adds the maj7 (pc 4,
    # E natural) -- the literal "E natural over F7" case that started this.
    plain = scale_pitch_classes(chord_root(F_DOM), chord_to_mode(F_DOM))
    widened = dissonance_scale(F_DOM)
    assert 4 not in plain
    assert 4 in widened
    assert plain <= widened  # nothing previously in-scale is lost


def test_dissonance_scale_major_includes_the_bebop_b6_passing_tone():
    # C major: plain ionian is {0,2,4,5,7,9,11}; bebop_major adds the b6 (pc 8).
    plain = scale_pitch_classes(chord_root(C_MAJOR), chord_to_mode(C_MAJOR))
    widened = dissonance_scale(C_MAJOR)
    assert 8 not in plain
    assert 8 in widened
    assert plain <= widened


def test_dissonance_scale_minor_is_unchanged():
    plain = scale_pitch_classes(chord_root(D_MINOR), chord_to_mode(D_MINOR))
    assert dissonance_scale(D_MINOR) == plain


def test_dissonance_scale_diminished_is_unchanged():
    plain = scale_pitch_classes(chord_root(G_DIM), chord_to_mode(G_DIM))
    assert dissonance_scale(G_DIM) == plain


def test_dissonance_no_longer_flags_the_bebop_passing_tone_at_all():
    # E natural (pc 4) over F7 -- previously out of scale (and previously
    # excused only via the passing-tone exception, Phase 19); now simply IN
    # the widened scale, not flagged as dissonant in the first place.
    notes = notes_from_pitches([65, 64, 65])  # F, E, F -- E approached/left by leap either way is irrelevant now
    assert dissonance(notes, F_DOM) == 0.0


# ---------------------------------------------------------------------------
# dissonance_scale -- Phase 21, Lever D: tritone/b5 substitution. A SINGLE
# extra pitch class for dominant chords, not a whole substitute scale --
# checked directly that unioning a whole scale saturates the metric (see
# dissonance_scale's own docstring for the F7/B7 arithmetic).
# ---------------------------------------------------------------------------


def test_widened_mode_scale_matches_dissonance_scale_minus_the_tritone_term():
    # Regression check on the Phase 21 refactor: _widened_mode_scale alone
    # (Lever A only) must be exactly what dissonance_scale had BEFORE the
    # tritone term was added, for every quality.
    for chord_idx in (C_MAJOR, F_DOM, D_MINOR, G_DIM):
        plain_widened = _widened_mode_scale(chord_idx)
        if chord_idx % 4 == QUAL_DOM:
            tritone_pc = (chord_root(chord_idx) + 6) % 12
            assert dissonance_scale(chord_idx) == plain_widened | {tritone_pc}
        else:
            assert dissonance_scale(chord_idx) == plain_widened


def test_dissonance_scale_dominant_gains_exactly_one_tritone_pitch_class():
    # F7 (root 5): tritone pitch class is (5+6)%12 = 11 (B). Checked as a
    # set DIFFERENCE, not just "contains 11" -- proves this is a single-note
    # addition on top of Lever A, not a wider one (the whole-scale approach
    # that was checked and rejected -- see dissonance_scale's docstring).
    lever_a_only = _widened_mode_scale(F_DOM)
    with_tritone = dissonance_scale(F_DOM)
    assert with_tritone - lever_a_only == {11}


def test_dissonance_scale_major_minor_diminished_unaffected_by_tritone_term():
    # The tritone term is quality-gated to QUAL_DOM only.
    assert dissonance_scale(C_MAJOR) == _widened_mode_scale(C_MAJOR)
    assert dissonance_scale(D_MINOR) == _widened_mode_scale(D_MINOR)
    assert dissonance_scale(G_DIM) == _widened_mode_scale(G_DIM)


def test_dissonance_extra_tolerated_excuses_a_note_only_when_supplied():
    # F# (pc 6) over F7 is not in dissonance_scale(F_DOM) at all (6 is
    # neither in the widened mode nor the tritone pitch 11) -- a real clash,
    # excused only when explicitly supplied via extra_tolerated.
    notes = notes_from_pitches([66])  # F#4
    assert 6 not in dissonance_scale(F_DOM)
    assert dissonance(notes, F_DOM) == 1.0
    assert dissonance(notes, F_DOM, extra_tolerated=frozenset({6})) == 0.0


# ---------------------------------------------------------------------------
# _is_passing_tone / dissonance's passing-tone exception
# ---------------------------------------------------------------------------


def test_is_passing_tone_descending_chromatic_run():
    # David's own example: a chromatically descending line -- C, B, Bb, A.
    # Bb (index 2) is approached (-1) and left (-1) in the same direction.
    notes = notes_from_pitches([60, 59, 58, 57])
    assert _is_passing_tone(notes, 2) is True


def test_is_passing_tone_ascending_chromatic_run():
    notes = notes_from_pitches([57, 58, 59, 60])
    assert _is_passing_tone(notes, 2) is True


def test_is_passing_tone_false_when_approached_by_step_but_left_by_leap():
    notes = notes_from_pitches([60, 61, 70])
    assert _is_passing_tone(notes, 1) is False


def test_is_passing_tone_false_when_approached_by_leap_but_left_by_step():
    notes = notes_from_pitches([60, 68, 67])
    assert _is_passing_tone(notes, 1) is False


def test_is_passing_tone_false_for_a_neighbour_tone_opposite_directions():
    # C-D-C: D is approached (+2) and left (-2) in OPPOSITE directions -- a
    # neighbour tone, not a passing tone. Deliberately excluded (see
    # dissonance's own docstring), not silently included.
    notes = notes_from_pitches([60, 62, 60])
    assert _is_passing_tone(notes, 1) is False


def test_is_passing_tone_false_at_start_or_end_of_phrase():
    notes = notes_from_pitches([60, 61, 62])
    assert _is_passing_tone(notes, 0) is False
    assert _is_passing_tone(notes, 2) is False


def test_is_passing_tone_false_when_either_side_exceeds_max_step():
    notes = notes_from_pitches([60, 63, 65])  # first interval is a minor 3rd (3 semitones)
    assert _is_passing_tone(notes, 1) is False


def test_dissonance_excuses_a_genuine_passing_tone():
    # F7 (mixolydian: F G A Bb C D Eb). G#4 (pc 8) is 1 semitone from both G
    # and A -- clashes in isolation. Approached from G (+1) and left toward A
    # (+1): a genuine ascending passing tone connecting two in-scale notes.
    F7 = 5 * 4 + QUAL_DOM  # root=F(5) -- matches ensemble/sax.py's chord_idx mapping
    passing = notes_from_pitches([67, 68, 69])  # G, G#, A
    assert dissonance(passing, F7) == 0.0


def test_dissonance_still_counts_the_same_pitch_when_not_a_passing_tone():
    # Same dissonant note (G#), but approached by a leap instead of a step --
    # no longer excused, back to counting as a clash. The concrete before/
    # after proof the exception is about MELODIC CONTEXT, not the pitch alone.
    F7 = 5 * 4 + QUAL_DOM
    leapt_into = notes_from_pitches([60, 68, 69])  # C (leap), G#, A
    assert dissonance(leapt_into, F7) == pytest.approx(1 / 3)


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
# motif_adherence -- distinct from repetition() above: adherence to an
# EXTERNAL target, not internal self-similarity
# ---------------------------------------------------------------------------


def test_motif_adherence_empty_targets_is_zero():
    notes = notes_from_pitches([60, 62, 64, 60, 62, 64])
    assert motif_adherence(notes, []) == 0.0


def test_motif_adherence_phrase_containing_target_scores_one():
    # intervals [2, 2] -- the phrase's own 2-gram is (2, 2), matching the target.
    notes = notes_from_pitches([60, 62, 64])
    assert motif_adherence(notes, [(2, 2)]) == 1.0


def test_motif_adherence_phrase_using_target_only_once_still_scores_one():
    """The exact case repetition() misses: the target motif appears exactly
    once (no internal recurrence needed) -- repetition() would score this 0.0
    since (2, 2) never recurs, but motif_adherence should score it 1.0 since
    it's checking against an external target, not self-repetition."""
    notes = notes_from_pitches([60, 62, 64, 70, 75])  # intervals [2, 2, 6, 5]
    assert repetition(notes) == 0.0
    assert motif_adherence(notes, [(2, 2)]) == 1.0


def test_motif_adherence_phrase_not_containing_target_scores_zero():
    notes = notes_from_pitches([60, 61, 63, 66, 70])  # intervals 1, 2, 3, 4
    assert motif_adherence(notes, [(2, 2)]) == 0.0


def test_motif_adherence_matches_any_one_of_multiple_targets():
    notes = notes_from_pitches([60, 62, 64])  # intervals [2, 2]
    assert motif_adherence(notes, [(9, 9), (2, 2), (5, -5)]) == 1.0


def test_motif_adherence_fewer_than_two_notes_is_zero():
    assert motif_adherence(notes_from_pitches([60]), [(2, 2)]) == 0.0


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


def test_musicality_score_weights_can_be_overridden_per_call():
    """Phase 13: the per-session/per-gesture configuration point (DESIGN.md §11)
    -- every sub-score is still computed regardless of weights (a metric "turned
    off" is still visible on the returned score), only overall's combination
    changes."""
    notes = notes_from_pitches([60, 62, 64, 60, 62, 64], duration_beats=0.95)
    seed = notes_from_pitches([60, 62, 64])

    default_score = musicality_score(notes, C_MAJOR, seed)
    zeroed_weights = dict(DEFAULT_WEIGHTS, singability=0.0)
    zeroed_score = musicality_score(notes, C_MAJOR, seed, weights=zeroed_weights)

    assert zeroed_score.singability == default_score.singability  # sub-score still reported
    assert zeroed_score.overall != default_score.overall  # but no longer counted toward overall
    expected_overall_without_singability = default_score.overall - (
        default_score.singability * DEFAULT_WEIGHTS["singability"]
    )
    assert zeroed_score.overall == pytest.approx(expected_overall_without_singability)
