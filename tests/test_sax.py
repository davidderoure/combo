"""Tests for ensemble/sax.py's pure functions — no sax_best.pt needed. The
real-inference behaviour (PhraseGenerator.generate, sax_generator end-to-end) is
in tests/test_sax_wolfson_integration.py, which needs the real weights."""

from song.chord import Chord, _QUALITY_ALIASES

from ensemble.sax import _build_seed_phrase, _place_phrase_in_bar, chord_to_wolfson_index
from ensemble.timeline import NoteEvent, Timeline
from ensemble.wolfson.chords import QUAL_DIM, QUAL_DOM, QUAL_MAJOR, QUAL_MINOR
from ensemble.wolfson.phrase_generator import REST_PITCH

REGISTER = (55, 79)


def test_chord_to_wolfson_index_maps_every_canonical_quality():
    # Exhaustiveness guard: every value song/chord.py's _QUALITY_ALIASES can
    # produce must be mapped, so a new quality added there doesn't silently
    # KeyError or fall through unmapped.
    for quality in set(_QUALITY_ALIASES.values()):
        chord_to_wolfson_index(Chord(root=0, quality=quality))  # must not raise


def test_chord_to_wolfson_index_spot_checks():
    assert chord_to_wolfson_index(Chord.parse("F7")) == 5 * 4 + QUAL_DOM
    assert chord_to_wolfson_index(Chord.parse("Bbmaj7")) == 10 * 4 + QUAL_MAJOR
    assert chord_to_wolfson_index(Chord.parse("Gm7b5")) == 7 * 4 + QUAL_DIM
    assert chord_to_wolfson_index(Chord.parse("Dm7")) == 2 * 4 + QUAL_MINOR


def test_build_seed_phrase_filters_by_voice_and_window():
    tl = Timeline(
        [
            NoteEvent("bass", 43, 80, 0.0, 1.0),  # in window
            NoteEvent("bass", 45, 80, 1.0, 0.5),  # in window
            NoteEvent("bass", 48, 80, 8.0, 1.0),  # outside window (until_beat)
            NoteEvent("sax", 60, 80, 0.5, 1.0),  # wrong voice
        ]
    )
    seed = _build_seed_phrase(tl, "bass", since_beat=0.0, until_beat=4.0)
    assert seed == [
        {"pitch": 43, "onset": 0.0, "offset": 1.0, "beat_dur_sec": 1.0},
        {"pitch": 45, "onset": 1.0, "offset": 1.5, "beat_dur_sec": 1.0},
    ]


def test_build_seed_phrase_translates_duration_exactly():
    tl = Timeline([NoteEvent("bass", 43, 80, 2.0, 1.75)])
    seed = _build_seed_phrase(tl, "bass", since_beat=0.0, until_beat=4.0)
    onset, offset = seed[0]["onset"], seed[0]["offset"]
    assert (offset - onset) / seed[0]["beat_dur_sec"] == 1.75


def test_place_phrase_in_bar_skips_rest_sentinels():
    notes = [{"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0}]
    notes.insert(0, {"pitch": REST_PITCH, "duration_beats": 1.0, "velocity_scale": 1.0})
    events = _place_phrase_in_bar(notes, bar_start=0.0, bar_end=4.0, register=REGISTER)
    assert len(events) == 1
    assert events[0].pitch == 60
    assert events[0].start_beat == 1.0  # cursor advanced past the rest gap


def test_place_phrase_in_bar_clips_overrunning_note():
    notes = [{"pitch": 60, "duration_beats": 3.5, "velocity_scale": 1.0}]
    events = _place_phrase_in_bar(notes, bar_start=1.0, bar_end=4.0, register=REGISTER)
    assert len(events) == 1
    assert events[0].start_beat == 1.0
    assert events[0].duration_beats == 3.0  # clipped to bar_end - bar_start


def test_place_phrase_in_bar_drops_notes_starting_past_bar_end():
    notes = [
        {"pitch": 60, "duration_beats": 4.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},  # cursor now >= bar_end
    ]
    events = _place_phrase_in_bar(notes, bar_start=0.0, bar_end=4.0, register=REGISTER)
    assert len(events) == 1
    assert events[0].pitch == 60


def test_place_phrase_in_bar_drops_out_of_register_notes():
    notes = [
        {"pitch": REGISTER[0] - 1, "duration_beats": 1.0, "velocity_scale": 1.0},  # below
        {"pitch": REGISTER[1] + 1, "duration_beats": 1.0, "velocity_scale": 1.0},  # above
        {"pitch": REGISTER[0], "duration_beats": 1.0, "velocity_scale": 1.0},  # in range
    ]
    events = _place_phrase_in_bar(notes, bar_start=0.0, bar_end=4.0, register=REGISTER)
    assert len(events) == 1
    assert events[0].pitch == REGISTER[0]


def test_place_phrase_in_bar_scales_velocity():
    notes = [{"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.25}]
    events = _place_phrase_in_bar(notes, bar_start=0.0, bar_end=4.0, register=REGISTER)
    from ensemble.sax import DEFAULT_VELOCITY

    assert events[0].velocity == round(DEFAULT_VELOCITY * 1.25)
