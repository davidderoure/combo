"""Tests for ensemble/sax.py's pure functions — no sax_best.pt needed. The
real-inference behaviour (PhraseGenerator.generate, sax_generator end-to-end) is
in tests/test_sax_wolfson_integration.py, which needs the real weights."""

from collections import Counter

from song import Changes, ChangesEvent, Section, Song
from song.chord import Chord, _QUALITY_ALIASES

from ensemble.critic import dissonance_scale
from ensemble.sax import (
    _bars_until_chord_change,
    _build_combined_seed_phrase,
    _build_seed_phrase,
    _decay_pitch_weighted,
    _functional_tonic_scale,
    _ii_v_i_target,
    _is_structural_cue,
    _pick_achievable_motif,
    _split_phrase_into_bars,
    chord_to_wolfson_index,
)
from ensemble.timeline import BEATS_PER_BAR, NoteEvent, Timeline
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


def test_build_combined_seed_phrase_none_reproduces_build_seed_phrase():
    tl = Timeline([NoteEvent("bass", 43, 80, 0.0, 1.0), NoteEvent("sax", 60, 80, 0.5, 1.0)])
    combined = _build_combined_seed_phrase(tl, "bass", None, since_beat=0.0, until_beat=4.0)
    assert combined == _build_seed_phrase(tl, "bass", since_beat=0.0, until_beat=4.0)


def test_build_combined_seed_phrase_appends_own_notes_after_target_notes():
    tl = Timeline(
        [
            NoteEvent("bass", 43, 80, 0.0, 1.0),
            NoteEvent("bass", 45, 80, 1.0, 1.0),
            NoteEvent("sax", 60, 80, 0.5, 0.5),
            NoteEvent("sax", 62, 80, 2.0, 1.0),
        ]
    )
    combined = _build_combined_seed_phrase(tl, "bass", "sax", since_beat=0.0, until_beat=4.0)
    target_only = _build_seed_phrase(tl, "bass", since_beat=0.0, until_beat=4.0)
    own_only = _build_seed_phrase(tl, "sax", since_beat=0.0, until_beat=4.0)
    assert combined == target_only + own_only  # target's notes, THEN own's -- not interleaved by onset


def test_build_combined_seed_phrase_own_voice_empty_window_is_target_only():
    tl = Timeline([NoteEvent("bass", 43, 80, 0.0, 1.0)])
    combined = _build_combined_seed_phrase(tl, "bass", "sax", since_beat=0.0, until_beat=4.0)
    assert combined == _build_seed_phrase(tl, "bass", since_beat=0.0, until_beat=4.0)


def test_build_combined_seed_phrase_target_empty_window_is_own_only():
    tl = Timeline([NoteEvent("sax", 60, 80, 0.0, 1.0)])
    combined = _build_combined_seed_phrase(tl, "bass", "sax", since_beat=0.0, until_beat=4.0)
    assert combined == _build_seed_phrase(tl, "sax", since_beat=0.0, until_beat=4.0)


def test_build_combined_seed_phrase_both_empty_is_empty():
    tl = Timeline([])
    assert _build_combined_seed_phrase(tl, "bass", "sax", since_beat=0.0, until_beat=4.0) == []


def _song_with_changes(*chord_durations) -> Song:
    """chord_durations: (Chord, duration_beats) pairs making up one chorus."""
    events = [ChangesEvent(chord, duration) for chord, duration in chord_durations]
    return Song(title="t", changes=Changes(events), form=[Section("A", 1)], tempo_bpm=120)


def test_bars_until_chord_change_every_bar_returns_one():
    song = _song_with_changes(
        (Chord.parse("F7"), 4.0), (Chord.parse("Bb7"), 4.0), (Chord.parse("F7"), 4.0), (Chord.parse("C7"), 4.0)
    )
    assert _bars_until_chord_change(song, 0.0, max_bars=4) == 1


def test_bars_until_chord_change_finds_a_two_bar_hold():
    song = _song_with_changes((Chord.parse("F7"), 8.0), (Chord.parse("Bb7"), 4.0))
    assert _bars_until_chord_change(song, 0.0, max_bars=4) == 2
    # Starting mid-hold (bar 1 of the 2-bar F7): only 1 bar left before the change.
    assert _bars_until_chord_change(song, BEATS_PER_BAR, max_bars=4) == 1


def test_bars_until_chord_change_caps_at_max_bars():
    song = _song_with_changes((Chord.parse("F7"), 16.0))  # one chord, 4 bars, no change ever
    assert _bars_until_chord_change(song, 0.0, max_bars=3) == 3


def test_is_structural_cue_true_at_a_real_chord_change():
    song = _song_with_changes((Chord.parse("F7"), 4.0), (Chord.parse("Bb7"), 4.0))
    assert _is_structural_cue(song, BEATS_PER_BAR) is True  # bar 1: F7 -> Bb7


def test_is_structural_cue_false_when_the_chord_holds():
    song = _song_with_changes((Chord.parse("F7"), 8.0), (Chord.parse("Bb7"), 4.0))
    assert _is_structural_cue(song, BEATS_PER_BAR) is False  # bar 1: still F7


def test_is_structural_cue_true_at_bar_zero_regardless_of_chord():
    song = _song_with_changes((Chord.parse("F7"), 16.0))  # one chord, never changes
    assert _is_structural_cue(song, 0.0) is True  # nothing before it to compare against


def test_decay_pitch_weighted_zero_elapsed_is_unchanged():
    assert _decay_pitch_weighted(120.0, 4.0, elapsed_beats=0.0, half_life_beats=16.0) == (120.0, 4.0)


def test_decay_pitch_weighted_one_half_life_halves_both():
    pitch_sum, beats = _decay_pitch_weighted(120.0, 4.0, elapsed_beats=16.0, half_life_beats=16.0)
    assert pitch_sum == 60.0
    assert beats == 2.0


def test_decay_pitch_weighted_two_half_lives_quarters_both():
    pitch_sum, beats = _decay_pitch_weighted(120.0, 4.0, elapsed_beats=32.0, half_life_beats=16.0)
    assert pitch_sum == 30.0
    assert beats == 1.0


def test_decay_pitch_weighted_zero_input_stays_zero():
    assert _decay_pitch_weighted(0.0, 0.0, elapsed_beats=100.0, half_life_beats=16.0) == (0.0, 0.0)


def test_ii_v_i_target_matches_a_textbook_ii_v_i():
    song = _song_with_changes((Chord.parse("Dm7"), 4.0), (Chord.parse("G7"), 4.0), (Chord.parse("Cmaj7"), 4.0))
    assert _ii_v_i_target(song, 0.0) == chord_to_wolfson_index(Chord.parse("Cmaj7"))


def test_ii_v_i_target_rejects_wrong_root_motion():
    # Dm7-G7-Ebmaj7 -- qualities right, but the I chord's root doesn't
    # continue the descending-fifths motion (V's root+5 semitones is C, not Eb).
    song = _song_with_changes((Chord.parse("Dm7"), 4.0), (Chord.parse("G7"), 4.0), (Chord.parse("Ebmaj7"), 4.0))
    assert _ii_v_i_target(song, 0.0) is None


def test_ii_v_i_target_rejects_wrong_qualities():
    # D7-G7-Cmaj7 -- root motion is right, but the "ii" is dominant, not minor.
    song = _song_with_changes((Chord.parse("D7"), 4.0), (Chord.parse("G7"), 4.0), (Chord.parse("Cmaj7"), 4.0))
    assert _ii_v_i_target(song, 0.0) is None


def test_functional_tonic_scale_resolves_the_same_target_from_ii_v_or_i():
    song = _song_with_changes((Chord.parse("Dm7"), 4.0), (Chord.parse("G7"), 4.0), (Chord.parse("Cmaj7"), 4.0))
    target_scale = dissonance_scale(chord_to_wolfson_index(Chord.parse("Cmaj7")))
    assert _functional_tonic_scale(song, 0.0) == target_scale  # bar_start is the ii
    assert _functional_tonic_scale(song, BEATS_PER_BAR) == target_scale  # bar_start is the V
    assert _functional_tonic_scale(song, 2 * BEATS_PER_BAR) == target_scale  # bar_start is the I


def test_functional_tonic_scale_empty_when_nothing_recognisable_nearby():
    song = _song_with_changes((Chord.parse("F7"), 4.0), (Chord.parse("Bb7"), 4.0), (Chord.parse("F7"), 4.0))
    assert _functional_tonic_scale(song, 0.0) == frozenset()


def test_split_phrase_into_bars_skips_rest_sentinels():
    notes = [{"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0}]
    notes.insert(0, {"pitch": REST_PITCH, "duration_beats": 1.0, "velocity_scale": 1.0})
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=1, register=REGISTER)
    assert len(bars) == 1
    assert len(bars[0]) == 1
    assert bars[0][0].pitch == 60
    assert bars[0][0].start_beat == 1.0  # cursor advanced past the rest gap


def test_split_phrase_into_bars_drops_notes_starting_past_plan_end():
    notes = [
        {"pitch": 60, "duration_beats": 4.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},  # cursor now >= plan_end
    ]
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=1, register=REGISTER)
    assert [e.pitch for e in bars[0]] == [60]


def test_split_phrase_into_bars_drops_out_of_register_notes():
    notes = [
        {"pitch": REGISTER[0] - 1, "duration_beats": 1.0, "velocity_scale": 1.0},  # below
        {"pitch": REGISTER[1] + 1, "duration_beats": 1.0, "velocity_scale": 1.0},  # above
        {"pitch": REGISTER[0], "duration_beats": 1.0, "velocity_scale": 1.0},  # in range
    ]
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=1, register=REGISTER)
    assert [e.pitch for e in bars[0]] == [REGISTER[0]]


def test_split_phrase_into_bars_scales_velocity():
    notes = [{"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.25}]
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=1, register=REGISTER)
    from ensemble.sax import DEFAULT_VELOCITY

    assert bars[0][0].velocity == round(DEFAULT_VELOCITY * 1.25)


def test_split_phrase_into_bars_returns_n_bars_lists_even_if_some_are_empty():
    notes = [{"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0}]
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=3, register=REGISTER)
    assert len(bars) == 3
    assert [len(b) for b in bars] == [1, 0, 0]


def test_split_phrase_into_bars_assigns_each_bar_its_own_notes():
    notes = [
        {"pitch": 60, "duration_beats": 1.0, "velocity_scale": 1.0},  # bar 0
        {"pitch": 62, "duration_beats": 1.0, "velocity_scale": 1.0},  # bar 0
        {"pitch": 64, "duration_beats": 2.5, "velocity_scale": 1.0},  # bar 0 -> spans into bar 1
        {"pitch": 65, "duration_beats": 1.0, "velocity_scale": 1.0},  # bar 1
    ]
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=2, register=REGISTER)
    assert [e.pitch for e in bars[0]] == [60, 62, 64]
    assert [e.pitch for e in bars[1]] == [64, 65]


def test_split_phrase_into_bars_splits_a_note_crossing_a_boundary_into_fragments():
    # plan_start=0.0, bar 0 = [0,4), bar 1 = [4,8). First note fits entirely in
    # bar 0 (cursor 0->3). Second note (cursor 3->6) crosses the boundary at
    # beat 4 -> should produce a 1-beat fragment finishing bar 0 and a 2-beat
    # fragment starting bar 1, rather than being truncated with the remainder
    # dropped. Third note (cursor 6->7) is entirely within bar 1.
    notes = [
        {"pitch": 60, "duration_beats": 3.0, "velocity_scale": 1.0},
        {"pitch": 62, "duration_beats": 3.0, "velocity_scale": 1.0},
        {"pitch": 65, "duration_beats": 1.0, "velocity_scale": 1.0},
    ]
    bars = _split_phrase_into_bars(notes, plan_start=0.0, n_bars=2, register=REGISTER)

    assert [(e.pitch, e.start_beat, e.duration_beats) for e in bars[0]] == [
        (60, 0.0, 3.0),
        (62, 3.0, 1.0),  # clipped to bar 0's boundary at beat 4.0
    ]
    assert [(e.pitch, e.start_beat, e.duration_beats) for e in bars[1]] == [
        (62, 4.0, 2.0),  # the remainder of the same note, continuing in bar 1
        (65, 6.0, 1.0),
    ]


# ---------------------------------------------------------------------------
# _pick_achievable_motif -- prefers the shortest (most achievable) recalled
# motif over simply the single most-common one of any length
# ---------------------------------------------------------------------------


def test_pick_achievable_motif_prefers_shorter_even_if_less_common():
    counter = Counter({(2, -1, 5, 3): 10.0, (2, 2): 1.0})  # 4-interval vs 2-interval
    assert _pick_achievable_motif(counter) == (2, 2)


def test_pick_achievable_motif_falls_back_to_three_interval_bucket():
    counter = Counter({(2, -1, 5, 3): 5.0, (1, -2, 4): 2.0})  # no 2-interval motifs at all
    assert _pick_achievable_motif(counter) == (1, -2, 4)


def test_pick_achievable_motif_falls_back_to_four_interval_bucket():
    counter = Counter({(2, -1, 5, 3): 5.0})  # only a 4-interval motif present
    assert _pick_achievable_motif(counter) == (2, -1, 5, 3)


def test_pick_achievable_motif_picks_the_highest_weighted_within_the_shortest_bucket():
    counter = Counter({(2, 2): 1.0, (3, -1): 4.0})  # both 2-interval -- higher weight wins
    assert _pick_achievable_motif(counter) == (3, -1)


def test_pick_achievable_motif_empty_counter_is_none():
    assert _pick_achievable_motif(Counter()) is None
