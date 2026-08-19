"""Tests for wjd_corpus.py's chord-quality/chord-idx classification and
run-splitting logic (Phases 29-30) -- pure, literal chord strings/hand-built
note lists as input, no real wjazzd.db needed. Breaks from Phase 28's "tool
scripts aren't unit-tested" precedent deliberately: this is genuinely new
classification logic (not pure orchestration), and it exists specifically to
correct a real, empirically-found bug in ensemble/wolfson/chords.py's ported
parse_chord -- worth protecting with tests, same as any other real logic in
this codebase."""

from ensemble.wolfson.chords import N_QUALITIES, QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR
from wjd_corpus import (
    _wjd_chord_idx,
    _wjd_chord_quality,
    _wjd_expected_bass_pc,
    split_into_chord_runs,
    split_into_quality_runs,
)


def test_wjd_chord_quality_plain_dominant():
    assert _wjd_chord_quality("G7") == QUAL_DOM


def test_wjd_chord_quality_minor():
    assert _wjd_chord_quality("C-7") == QUAL_MINOR


def test_wjd_chord_quality_major_seventh_j_prefix():
    """The specific case ensemble.wolfson.chords.parse_chord gets wrong --
    checked directly (see module docstring / DESIGN.md): parse_chord("Abj7")
    returns QUAL_DOM because its quality check looks for the substring
    "maj", not Jazzomat's actual "j" marker. This function must get it
    right where the ported one doesn't."""
    assert _wjd_chord_quality("Abj7") == QUAL_MAJOR
    assert _wjd_chord_quality("Aj7911#") == QUAL_MAJOR


def test_wjd_chord_quality_minor_major_seventh_is_minor_not_major():
    # "-j7" (minor-major7) starts with '-', so the minor check must fire
    # before the 'j' check ever sees it.
    assert _wjd_chord_quality("A-j7") == QUAL_MINOR


def test_wjd_chord_quality_half_diminished_and_diminished():
    assert _wjd_chord_quality("Abm7b5") == QUAL_DIM
    assert _wjd_chord_quality("Abo7") == QUAL_DIM
    assert _wjd_chord_quality("Abo") == QUAL_DIM


def test_wjd_chord_quality_no_chord_is_none():
    assert _wjd_chord_quality("NC") is None


def test_wjd_chord_quality_slash_bass_ignored_for_quality():
    assert _wjd_chord_quality("A-/G") == QUAL_MINOR
    assert _wjd_chord_quality("Ab79/C") == QUAL_DOM
    assert _wjd_chord_quality("A/C") == QUAL_MAJOR  # plain major triad over a C bass


def test_wjd_chord_quality_sus_is_dominant():
    assert _wjd_chord_quality("Absus7") == QUAL_DOM
    assert _wjd_chord_quality("Bbsus") == QUAL_DOM


def test_wjd_chord_quality_augmented_is_dominant():
    assert _wjd_chord_quality("A+7") == QUAL_DOM


def test_wjd_chord_quality_plain_major_and_sixth():
    assert _wjd_chord_quality("Bb") == QUAL_MAJOR
    assert _wjd_chord_quality("Bb6") == QUAL_MAJOR


def test_split_into_quality_runs_groups_contiguous_same_quality():
    notes = [
        {"pitch": 60, "chord_quality": QUAL_DOM},
        {"pitch": 62, "chord_quality": QUAL_DOM},
        {"pitch": 64, "chord_quality": QUAL_MINOR},
        {"pitch": 65, "chord_quality": QUAL_MINOR},
        {"pitch": 67, "chord_quality": QUAL_MINOR},
    ]
    runs = split_into_quality_runs(notes)
    assert runs == [
        (QUAL_DOM, notes[0:2]),
        (QUAL_MINOR, notes[2:5]),
    ]


def test_split_into_quality_runs_drops_none_quality_notes():
    notes = [
        {"pitch": 60, "chord_quality": None},
        {"pitch": 62, "chord_quality": None},
        {"pitch": 64, "chord_quality": QUAL_MAJOR},
        {"pitch": 65, "chord_quality": QUAL_MAJOR},
    ]
    runs = split_into_quality_runs(notes)
    assert runs == [(QUAL_MAJOR, notes[2:4])]


def test_split_into_quality_runs_none_interrupts_a_run():
    notes = [
        {"pitch": 60, "chord_quality": QUAL_DOM},
        {"pitch": 62, "chord_quality": QUAL_DOM},
        {"pitch": 64, "chord_quality": None},
        {"pitch": 65, "chord_quality": QUAL_DOM},
    ]
    runs = split_into_quality_runs(notes)
    # the None note breaks contiguity -- two separate DOM runs, not one
    assert runs == [(QUAL_DOM, notes[0:2]), (QUAL_DOM, [notes[3]])]


def test_split_into_quality_runs_single_quality_solo_is_one_run():
    notes = [{"pitch": p, "chord_quality": QUAL_MINOR} for p in (60, 62, 64)]
    assert split_into_quality_runs(notes) == [(QUAL_MINOR, notes)]


def test_split_into_quality_runs_empty_input():
    assert split_into_quality_runs([]) == []


# ---------------------------------------------------------------------------
# _wjd_chord_idx / split_into_chord_runs (Phase 30)
# ---------------------------------------------------------------------------


def test_wjd_chord_idx_combines_root_and_quality():
    # G=7 (ROOTS index), quality DOM
    assert _wjd_chord_idx("G7") == 7 * N_QUALITIES + QUAL_DOM
    # C=0, quality MIN
    assert _wjd_chord_idx("C-7") == 0 * N_QUALITIES + QUAL_MINOR


def test_wjd_chord_idx_uses_the_corrected_quality_for_the_j_prefix_case():
    # Ab=8 -- parse_chord's OWN quality half would get this wrong (DOM, not
    # MAJOR, the Phase 29 bug); _wjd_chord_idx must use our corrected
    # classifier for the quality half, not parse_chord's.
    assert _wjd_chord_idx("Abj7") == 8 * N_QUALITIES + QUAL_MAJOR


def test_wjd_chord_idx_handles_sharp_roots_via_enharmonic_flat_spelling():
    # F#7 -- parse_chord's root comes out as Gb (index 6), combo's own
    # flat-only ROOTS spelling, verified directly against real inference.
    assert _wjd_chord_idx("F#7") == 6 * N_QUALITIES + QUAL_DOM


def test_wjd_chord_idx_no_chord_is_none():
    assert _wjd_chord_idx("NC") is None


def test_split_into_chord_runs_groups_by_full_chord_idx_not_just_quality():
    # Same quality (dominant) but different roots -- chord_idx tagging must
    # still split them, unlike split_into_quality_runs which would pool them.
    g7 = 7 * N_QUALITIES + QUAL_DOM
    c7 = 0 * N_QUALITIES + QUAL_DOM
    notes = [
        {"pitch": 60, "chord_idx": g7},
        {"pitch": 62, "chord_idx": g7},
        {"pitch": 64, "chord_idx": c7},
        {"pitch": 65, "chord_idx": c7},
    ]
    runs = split_into_chord_runs(notes)
    assert runs == [(g7, notes[0:2]), (c7, notes[2:4])]


def test_split_into_chord_runs_drops_none_and_empty_input():
    assert split_into_chord_runs([{"pitch": 60, "chord_idx": None}]) == []
    assert split_into_chord_runs([]) == []


# ---------------------------------------------------------------------------
# _wjd_expected_bass_pc (Phase 31)
# ---------------------------------------------------------------------------


def test_wjd_expected_bass_pc_plain_chord_is_its_own_root():
    # G=7 -- matches _wjd_chord_idx's own root half for a non-slash chord.
    assert _wjd_expected_bass_pc("G7") == 7
    assert _wjd_expected_bass_pc("C-7") == 0


def test_wjd_expected_bass_pc_slash_chord_uses_the_post_slash_bass_note():
    # "Ab79/C" -- root is Ab(8), but the WRITTEN bass is C(0); the expected
    # pitch class is the bass override, not the chord's own root.
    assert _wjd_expected_bass_pc("Ab79/C") == 0
    assert _wjd_expected_bass_pc("A-/G") == 7  # bass=G(7), not A(9)


def test_wjd_expected_bass_pc_no_chord_is_none():
    assert _wjd_expected_bass_pc("NC") is None


def test_wjd_expected_bass_pc_unparseable_bass_part_is_none():
    assert _wjd_expected_bass_pc("A7/xyz") is None
