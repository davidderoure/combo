"""Tests for wjd_corpus.py's chord-quality classification and run-splitting
logic (Phase 29) -- pure, literal chord strings/hand-built note lists as
input, no real wjazzd.db needed. Breaks from Phase 28's "tool scripts aren't
unit-tested" precedent deliberately: this is genuinely new classification
logic (not pure orchestration), and it exists specifically to correct a real,
empirically-found bug in ensemble/wolfson/chords.py's ported parse_chord --
worth protecting with tests, same as any other real logic in this codebase."""

from ensemble.wolfson.chords import QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR
from wjd_corpus import _wjd_chord_quality, split_into_quality_runs


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
