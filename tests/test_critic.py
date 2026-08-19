"""Tests for ensemble/critic.py — no sax_best.pt needed. Every function here is
pure and deterministic over already-generated data (no model inference), the
first sax-adjacent test file that's true of. Real end-to-end wiring (does a
generated chunk's score actually reach RehearsalMemory) is checked in
tests/test_sax_wolfson_integration.py, which needs the real weights."""

import json
import math

import pytest

from ensemble.corpus_motifs import CorpusMotifs
from ensemble.critic import (
    BREATH_FRACTION_WIDTH,
    DEFAULT_WEIGHTS,
    MIN_BREATH_BEATS,
    MODAL_LEAP_SEMITONES,
    TARGET_BREATH_FRACTION,
    TONAL_RESOLUTION_WEIGHT,
    _contour_string,
    _is_passing_tone,
    _is_resolved_tension,
    _levenshtein,
    _semitones_to_scale,
    _widened_mode_scale,
    call_response_relatedness,
    contour_smoothness,
    corpus_familiarity,
    dissonance,
    dissonance_scale,
    motif_adherence,
    musicality_score,
    phrasing,
    register_usage,
    repetition,
    singability,
    tonal_conformity,
)
from ensemble.wolfson.chords import QUAL_DIM, QUAL_DOM, QUAL_MINOR
from ensemble.wolfson.encoding import dur_to_token
from ensemble.wolfson.phrase_generator import REST_PITCH
from ensemble.wolfson.scales import chord_root, chord_to_mode, chord_tones, scale_pitch_classes

C_MAJOR = 0  # root=0 (C), quality_class=0 (QUAL_MAJOR) -> chord_idx = 0*4+0 = 0
F_DOM = 5 * 4 + QUAL_DOM  # root=F(5) -- matches ensemble/sax.py's chord_idx mapping
D_MINOR = 2 * 4 + QUAL_MINOR
G_DIM = 7 * 4 + QUAL_DIM
REGISTER = (55, 79)  # matches tests/test_sax_wolfson_integration.py's SAX_REGISTER


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
# tonal_conformity's Phase 27 consistency with dissonance()'s own tolerances
# ---------------------------------------------------------------------------
# F_DOM's dissonance_scale is {0,2,3,4,5,7,9,10,11} (missing 1, 6, 8);
# chord_tones(F_DOM) is {3, 5, 9}.


def test_tonal_conformity_extra_tolerated_widens_what_counts():
    # F# (pc6) over F7 is outside dissonance_scale(F_DOM) entirely -- 0.0
    # without extra_tolerated; explicitly tolerated, it counts toward
    # scale_fraction (though it still isn't a chord tone, so it doesn't earn
    # the resolution bonus).
    notes = notes_from_pitches([66])
    assert tonal_conformity(notes, F_DOM) == 0.0
    assert tonal_conformity(notes, F_DOM, extra_tolerated=frozenset({6})) == pytest.approx(0.7)


def test_tonal_conformity_counts_a_genuine_passing_tone_unconditionally():
    # Same passing-tone shape as dissonance()'s own precedent test: G, G#, A
    # over F7 -- G# (pc8) is outside the scale but a genuine passing tone,
    # excused the same way it is in dissonance(), with no flag needed.
    notes = notes_from_pitches([67, 68, 69])
    assert tonal_conformity(notes, F_DOM) == 1.0


def test_tonal_conformity_counts_a_resolved_tension_only_when_credited():
    # C, Db (leap in), Eb (step out, a chord tone) -- Db (pc1) clashes and
    # isn't a passing tone (leapt into), but IS a resolved tension. Before/
    # after proof: uncounted by default, counted when credit_resolved_tension
    # =True (matching dissonance()'s own credit_resolved_tension exactly).
    notes = notes_from_pitches([60, 73, 75])
    without = (1.0 - TONAL_RESOLUTION_WEIGHT) * (2 / 3) + TONAL_RESOLUTION_WEIGHT * 1.0
    assert tonal_conformity(notes, F_DOM) == pytest.approx(without)
    assert tonal_conformity(notes, F_DOM, credit_resolved_tension=True) == 1.0


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
# _is_resolved_tension / dissonance's tension-and-resolution exception (Phase 22)
# ---------------------------------------------------------------------------
# F7's dissonance_scale is {0,2,3,4,5,7,9,10,11} (missing 1, 6, 8 -- each
# exactly 1 semitone away, so each is a "clash" pitch class). chord_tones(F7)
# is {3, 5, 9} (Eb, F, A -- root/maj3/min7).


def test_is_resolved_tension_true_when_approached_in_scale_and_resolved_onto_chord_tone():
    scale = dissonance_scale(F_DOM)
    tones = chord_tones(F_DOM)
    notes = notes_from_pitches([60, 73, 75])  # C (in scale), Db (leap in), Eb (step out, chord tone)
    assert _is_resolved_tension(notes, 1, scale, tones) is True


def test_is_resolved_tension_false_when_approached_from_out_of_scale():
    # Db is preceded by F# (itself out of scale) -- part of a longer
    # excursion, not a single isolated reach, so it's not excused.
    scale = dissonance_scale(F_DOM)
    tones = chord_tones(F_DOM)
    notes = notes_from_pitches([66, 73, 75])  # F# (out of scale), Db, Eb (chord tone)
    assert _is_resolved_tension(notes, 1, scale, tones) is False


def test_is_resolved_tension_false_when_resolution_exceeds_max_step():
    scale = dissonance_scale(F_DOM)
    tones = chord_tones(F_DOM)
    notes = notes_from_pitches([60, 73, 90])  # resolves by a 17-semitone leap, not a step
    assert _is_resolved_tension(notes, 1, scale, tones) is False


def test_is_resolved_tension_false_when_resolved_onto_a_non_chord_tone():
    scale = dissonance_scale(F_DOM)
    tones = chord_tones(F_DOM)
    notes = notes_from_pitches([60, 73, 74])  # resolves by step onto D -- in scale, not a chord tone
    assert _is_resolved_tension(notes, 1, scale, tones) is False


def test_is_resolved_tension_false_at_start_or_end_of_phrase():
    scale = dissonance_scale(F_DOM)
    tones = chord_tones(F_DOM)
    notes = notes_from_pitches([60, 73, 75])
    assert _is_resolved_tension(notes, 0, scale, tones) is False
    assert _is_resolved_tension(notes, 2, scale, tones) is False


def test_dissonance_excuses_a_resolved_tension_when_credited():
    # Db (pc1) over F7 clashes in isolation. Approached from C (in
    # dissonance_scale) and resolved by step onto Eb (a chord tone of F7) --
    # a genuine tension-and-resolution, only excused when
    # credit_resolved_tension=True. Default (False) reproduces the plain
    # 1/3 clash rate -- the concrete before/after proof, and a regression
    # check that the new parameter defaults to today's behaviour.
    notes = notes_from_pitches([60, 73, 75])  # C, Db (leap in), Eb (step out, chord tone)
    assert dissonance(notes, F_DOM, credit_resolved_tension=True) == 0.0
    assert dissonance(notes, F_DOM, credit_resolved_tension=False) == pytest.approx(1 / 3)


def test_dissonance_does_not_excuse_the_same_tension_approached_mid_excursion():
    # Same target/resolution notes (Db -> Eb), but approached from F# (out
    # of scale) instead -- reads as part of a longer excursion, not an
    # isolated reach, so it's NOT excused even with credit_resolved_tension
    # =True. Both the F# and the Db count as clashes: 2 of 3 real notes.
    notes = notes_from_pitches([66, 73, 75])  # F# (out of scale), Db, Eb (chord tone)
    assert dissonance(notes, F_DOM, credit_resolved_tension=True) == pytest.approx(2 / 3)


def test_dissonance_does_not_excuse_an_unresolved_tension():
    # Same approach (from an in-scale note), but the tension note is left
    # UNRESOLVED -- landing on D (in scale, but not a chord tone) rather
    # than a genuine chord-tone resolution. Still counts as a clash even
    # with credit_resolved_tension=True -- the concrete "still catches
    # getting lost" proof, not "any dissonant note is now free."
    notes = notes_from_pitches([60, 73, 74])  # C, Db, D (in scale, not a chord tone)
    assert dissonance(notes, F_DOM, credit_resolved_tension=True) == pytest.approx(1 / 3)


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


def test_contour_smoothness_p4_and_p5_leaps_smooth_only_when_modal():
    assert MODAL_LEAP_SEMITONES == frozenset({5, 7})
    p4 = notes_from_pitches([60, 65])  # interval 5, a P4
    p5 = notes_from_pitches([60, 67])  # interval 7, a P5
    assert contour_smoothness(p4) == 0.0
    assert contour_smoothness(p5) == 0.0
    assert contour_smoothness(p4, modal=True) == 1.0
    assert contour_smoothness(p5, modal=True) == 1.0


def test_contour_smoothness_modal_does_not_excuse_wider_leaps():
    # A major 6th (9 semitones) -- outside MODAL_LEAP_SEMITONES -- stays
    # unsmooth even with modal=True. Widens tolerance for a specific, named
    # vocabulary, not a general loosening.
    notes = notes_from_pitches([60, 69])
    assert contour_smoothness(notes) == 0.0
    assert contour_smoothness(notes, modal=True) == 0.0


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
# corpus_familiarity
# ---------------------------------------------------------------------------

def _write_corpus_cache(path, chord_quality, pitch_motifs=(), duration_motifs=()):
    raw = {
        "n_solos": 1,
        "n_notes": 1,
        "pitch_motifs": {str(chord_quality): [[list(m), 1] for m in pitch_motifs]},
        "duration_motifs": {str(chord_quality): [[list(m), 1] for m in duration_motifs]},
    }
    path.write_text(json.dumps(raw))


def test_corpus_familiarity_all_motifs_found_scores_one(tmp_path):
    notes = notes_from_pitches([60, 62, 64], duration_beats=1.0)  # pitch motif (2,2); dur token t x3
    t = dur_to_token(1.0)
    path = tmp_path / "cache.json"
    _write_corpus_cache(path, QUAL_DOM, pitch_motifs=[(2, 2)], duration_motifs=[(t, t), (t, t, t)])
    corpus = CorpusMotifs(path)
    assert corpus_familiarity(notes, QUAL_DOM, corpus) == 1.0


def test_corpus_familiarity_no_motifs_found_scores_zero(tmp_path):
    notes = notes_from_pitches([60, 62, 64], duration_beats=1.0)
    path = tmp_path / "cache.json"
    _write_corpus_cache(path, QUAL_DOM, pitch_motifs=[(9, 9)], duration_motifs=[(1, 1)])
    corpus = CorpusMotifs(path)
    assert corpus_familiarity(notes, QUAL_DOM, corpus) == 0.0


def test_corpus_familiarity_mixed_scores_the_exact_fraction(tmp_path):
    # notes_from_pitches([60,62,64], duration_beats=1.0) produces exactly 4
    # motifs total: pitch (2,2) once, duration (t,t) twice (two overlapping
    # 2-note windows) and (t,t,t) once -- see extract_interval_motifs /
    # extract_duration_motifs. Only (2,2) and (t,t) are in the corpus here,
    # so 3 of the 4 pooled motifs (1 pitch + 2 duration occurrences of (t,t))
    # are found -> 3/4.
    notes = notes_from_pitches([60, 62, 64], duration_beats=1.0)
    t = dur_to_token(1.0)
    path = tmp_path / "cache.json"
    _write_corpus_cache(path, QUAL_DOM, pitch_motifs=[(2, 2)], duration_motifs=[(t, t)])
    corpus = CorpusMotifs(path)
    assert corpus_familiarity(notes, QUAL_DOM, corpus) == pytest.approx(0.75)


def test_corpus_familiarity_looks_up_the_given_quality_only(tmp_path):
    # the corpus has the motif under QUAL_MINOR, but we ask about QUAL_DOM --
    # no cross-quality credit.
    notes = notes_from_pitches([60, 62, 64], duration_beats=1.0)
    path = tmp_path / "cache.json"
    _write_corpus_cache(path, QUAL_MINOR, pitch_motifs=[(2, 2)])
    corpus = CorpusMotifs(path)
    assert corpus_familiarity(notes, QUAL_DOM, corpus) == 0.0


def test_corpus_familiarity_too_few_notes_for_any_motif_is_zero(tmp_path):
    path = tmp_path / "cache.json"
    _write_corpus_cache(path, QUAL_DOM)
    corpus = CorpusMotifs(path)
    assert corpus_familiarity(notes_from_pitches([60]), QUAL_DOM, corpus) == 0.0


def test_corpus_familiarity_rests_are_excluded_like_every_other_metric(tmp_path):
    notes = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": 64, "duration_beats": 1.0, "velocity_scale": 1.0},
    ]
    t = dur_to_token(1.0)
    path = tmp_path / "cache.json"
    _write_corpus_cache(path, QUAL_DOM, pitch_motifs=[(2, 2)], duration_motifs=[(t, t), (t, t, t)])
    corpus = CorpusMotifs(path)
    # with the rest excluded, the real notes are exactly [60, 62, 64] again --
    # same result as the all-found test above, proving REST_PITCH is ignored.
    assert corpus_familiarity(notes, QUAL_DOM, corpus) == 1.0


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
# phrasing (Phase 23) -- the only function here that looks at REST_PITCH
# sentinels directly rather than _real_notes()-filtering them out first.
# ---------------------------------------------------------------------------


def test_phrasing_no_notes_is_zero():
    assert phrasing([]) == 0.0


def test_phrasing_zero_breath_scores_below_near_target():
    # No REST_PITCH entries at all -- breath fraction 0.0, away from
    # TARGET_BREATH_FRACTION -- reads as running on, no gaps.
    no_breath = notes_from_pitches([60, 62, 64, 65, 67, 69, 71, 72])
    # total = 7.5 beats, one 1.0-beat rest -> fraction ~0.133, close to 0.15.
    near_target = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": 64, "duration_beats": 4.5, "velocity_scale": 1.0},
    ]
    assert phrasing(no_breath) < phrasing(near_target)
    assert phrasing(near_target) > 0.9


def test_phrasing_excessive_breath_scores_below_near_target_too():
    # total = 5.0 beats, 3.0 beats of rest -> fraction 0.6, far ABOVE the
    # target -- the bell curve penalises too much silence symmetrically,
    # not just too little.
    excessive = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 3.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},
    ]
    near_target = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": 64, "duration_beats": 4.5, "velocity_scale": 1.0},
    ]
    assert phrasing(excessive) < phrasing(near_target)


def test_phrasing_sub_threshold_rest_does_not_count_as_breath():
    # The rest is shorter than MIN_BREATH_BEATS -- doesn't count, so this
    # should score identically to an equal-duration phrase with NO rest at
    # all (both have breath fraction 0.0).
    below_threshold = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 0.25, "velocity_scale": 1.0},  # < MIN_BREATH_BEATS
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},
    ]
    assert 0.25 < MIN_BREATH_BEATS
    no_rest_same_total = notes_from_pitches([60, 62], duration_beats=1.125)  # same 2.25 total
    assert phrasing(below_threshold) == pytest.approx(phrasing(no_rest_same_total))


def test_phrasing_denominator_is_total_duration_not_just_real_note_time():
    # 1 real note (1.0 beat) + 1 qualifying rest (1.0 beat) -> fraction must
    # be 1.0/2.0 = 0.5 (total duration), NOT 1.0/1.0 = 1.0 (real-note time
    # only) -- computed directly against the exact formula to pin this down.
    notes = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},
        {"pitch": REST_PITCH, "duration_beats": 1.0, "velocity_scale": 1.0},
    ]
    expected = math.exp(-0.5 * ((0.5 - TARGET_BREATH_FRACTION) / BREATH_FRACTION_WIDTH) ** 2)
    assert phrasing(notes) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# register_usage (Phase 24)
# ---------------------------------------------------------------------------
# REGISTER = (55, 79), width 24 semitones.


def test_register_usage_no_notes_is_zero():
    assert register_usage([], REGISTER) == 0.0


def test_register_usage_single_real_note_is_zero():
    assert register_usage(notes_from_pitches([60]), REGISTER) == 0.0


def test_register_usage_is_the_in_register_span_fraction():
    # Pitches 60 and 72 -- both in REGISTER, span 12 semitones -> 12/24 = 0.5.
    notes = notes_from_pitches([60, 65, 72])
    assert register_usage(notes, REGISTER) == pytest.approx(0.5)


def test_register_usage_full_register_is_one():
    notes = notes_from_pitches([55, 67, 79])  # spans the entire register
    assert register_usage(notes, REGISTER) == pytest.approx(1.0)


def test_register_usage_ignores_out_of_register_outliers():
    # Same in-register pair (60, 66 -- span 6/24=0.25) as the baseline, but
    # with an extra note OUTSIDE the register (40, well below REGISTER's
    # low of 55). The raw span (66-40=26) would exceed the register's own
    # width entirely; the out-of-register note must not inflate the score,
    # since it will never actually sound (_split_phrase_into_bars drops it).
    baseline = notes_from_pitches([60, 66])
    with_outlier = notes_from_pitches([40, 60, 66])
    assert register_usage(baseline, REGISTER) == pytest.approx(0.25)
    assert register_usage(with_outlier, REGISTER) == pytest.approx(register_usage(baseline, REGISTER))


def test_register_usage_zero_width_register_is_zero():
    assert register_usage(notes_from_pitches([60, 62]), (60, 60)) == 0.0


def test_register_usage_prior_range_none_reproduces_per_chunk_only_behaviour():
    notes = notes_from_pitches([60, 65, 72])
    assert register_usage(notes, REGISTER, prior_range=None) == register_usage(notes, REGISTER)


def test_register_usage_prior_range_widens_a_narrow_candidates_score():
    # candidate span alone: 62-60=2/24 (narrow); prior_range=(55,67) -- 12/24=0.5
    # -- both notes fall INSIDE prior_range, so nothing new is added: the
    # combined span is exactly prior_range's own, not the candidate's tiny one.
    notes = notes_from_pitches([60, 62])
    assert register_usage(notes, REGISTER, prior_range=(55, 67)) == pytest.approx(0.5)
    assert register_usage(notes, REGISTER) == pytest.approx(2 / 24)  # the old, narrow, per-chunk-only score


def test_register_usage_prior_range_a_genuine_excursion_above_raises_the_score():
    # prior_range=(55,67) span 0.5; candidate pushes to 70/72, ABOVE prior's high.
    notes = notes_from_pitches([70, 72])
    assert register_usage(notes, REGISTER, prior_range=(55, 67)) == pytest.approx((72 - 55) / 24)


def test_register_usage_prior_range_a_genuine_excursion_below_raises_the_score():
    # prior_range=(60,79) span (79-60)/24; candidate pushes to 56/58, BELOW prior's low.
    notes = notes_from_pitches([56, 58])
    assert register_usage(notes, REGISTER, prior_range=(60, 79)) == pytest.approx((79 - 56) / 24)


def test_register_usage_prior_range_no_candidate_notes_reports_priors_own_span():
    # A candidate that contributes nothing new (empty, or entirely out of
    # register) still gets an honest answer -- prior_range's own span, not
    # forced to 0.0 the way a bare candidate-only call would be.
    assert register_usage([], REGISTER, prior_range=(55, 67)) == pytest.approx(0.5)
    out_of_register = notes_from_pitches([40])  # below REGISTER entirely
    assert register_usage(out_of_register, REGISTER, prior_range=(55, 67)) == pytest.approx(0.5)


def test_musicality_score_threads_prior_range_into_register_usage():
    notes = notes_from_pitches([60, 62])
    score = musicality_score(notes, C_MAJOR, [], REGISTER, prior_range=(55, 67))
    assert score.register_usage == register_usage(notes, REGISTER, prior_range=(55, 67))


# ---------------------------------------------------------------------------
# musicality_score combination
# ---------------------------------------------------------------------------


def test_musicality_score_overall_is_the_documented_weighted_sum():
    notes = notes_from_pitches([60, 62, 64, 60, 62, 64], duration_beats=0.95)
    seed = notes_from_pitches([60, 62, 64])
    score = musicality_score(notes, C_MAJOR, seed, REGISTER)

    expected_overall = (
        score.tonal_conformity * DEFAULT_WEIGHTS["tonal_conformity"]
        + score.contour_smoothness * DEFAULT_WEIGHTS["contour_smoothness"]
        + score.repetition * DEFAULT_WEIGHTS["repetition"]
        + score.call_response_relatedness * DEFAULT_WEIGHTS["call_response_relatedness"]
        + score.singability * DEFAULT_WEIGHTS["singability"]
        + score.phrasing * DEFAULT_WEIGHTS["phrasing"]
        + score.register_usage * DEFAULT_WEIGHTS["register_usage"]
    )
    assert score.overall == pytest.approx(expected_overall)


def test_repetitions_negative_weight_makes_a_repetitive_phrase_score_lower():
    """Phase 33: DEFAULT_WEIGHTS["repetition"] is negative -- a phrase that
    SHOWS repetition must score LOWER in overall than one that doesn't, not
    higher (Phase 12's original positive framing, inverted once Phase 30
    found combo shows repetition far more than real solos do). Isolated via
    a weights override that zeroes every other metric, so the only thing
    that can move `overall` here is repetition's own value and sign."""
    repetition_only_weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    repetition_only_weights["repetition"] = DEFAULT_WEIGHTS["repetition"]

    repetitive = notes_from_pitches([60, 62, 64, 60, 62, 64])  # repetition() == 1.0
    varied = notes_from_pitches([60, 61, 63, 66, 70])  # repetition() == 0.0

    repetitive_score = musicality_score(repetitive, C_MAJOR, [], REGISTER, weights=repetition_only_weights)
    varied_score = musicality_score(varied, C_MAJOR, [], REGISTER, weights=repetition_only_weights)

    assert repetitive_score.repetition == 1.0
    assert varied_score.repetition == 0.0
    assert repetitive_score.overall < varied_score.overall
    assert DEFAULT_WEIGHTS["repetition"] < 0.0  # the actual polarity this test depends on
    assert repetitive_score.overall == pytest.approx(DEFAULT_WEIGHTS["repetition"])
    assert varied_score.overall == pytest.approx(0.0)


def test_musicality_score_weights_can_be_overridden_per_call():
    """Phase 13: the per-session/per-gesture configuration point (DESIGN.md §11)
    -- every sub-score is still computed regardless of weights (a metric "turned
    off" is still visible on the returned score), only overall's combination
    changes."""
    notes = notes_from_pitches([60, 62, 64, 60, 62, 64], duration_beats=0.95)
    seed = notes_from_pitches([60, 62, 64])

    default_score = musicality_score(notes, C_MAJOR, seed, REGISTER)
    zeroed_weights = dict(DEFAULT_WEIGHTS, singability=0.0)
    zeroed_score = musicality_score(notes, C_MAJOR, seed, REGISTER, weights=zeroed_weights)

    assert zeroed_score.singability == default_score.singability  # sub-score still reported
    assert zeroed_score.overall != default_score.overall  # but no longer counted toward overall
    expected_overall_without_singability = default_score.overall - (
        default_score.singability * DEFAULT_WEIGHTS["singability"]
    )
    assert zeroed_score.overall == pytest.approx(expected_overall_without_singability)


def test_musicality_score_threads_extra_tolerated_and_credit_resolved_tension_into_tonal_conformity():
    """Phase 27: proof the new params actually reach tonal_conformity, not
    just that tonal_conformity itself works in isolation."""
    notes = notes_from_pitches([60, 73, 75])  # C, Db (leap in), Eb (step out, chord tone)
    seed = notes_from_pitches([60, 62, 64])
    default_score = musicality_score(notes, F_DOM, seed, REGISTER)
    credited_score = musicality_score(notes, F_DOM, seed, REGISTER, credit_resolved_tension=True)
    assert credited_score.tonal_conformity > default_score.tonal_conformity
    assert credited_score.overall > default_score.overall


def test_musicality_score_threads_modal_into_contour_smoothness():
    """Phase 27: proof `modal` actually reaches contour_smoothness."""
    notes = notes_from_pitches([60, 65, 60, 67])  # P4, P4, P5 leaps throughout
    seed = notes_from_pitches([60, 62, 64])
    default_score = musicality_score(notes, C_MAJOR, seed, REGISTER)
    modal_score = musicality_score(notes, C_MAJOR, seed, REGISTER, modal=True)
    assert modal_score.contour_smoothness > default_score.contour_smoothness
    assert modal_score.overall > default_score.overall
