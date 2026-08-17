"""Integration tests for ensemble/sax.py against the real Wolfson-adapted model —
DESIGN.md §12, Phase 8. First test file in this codebase needing a binary artifact
absent from a fresh clone: skips cleanly if ensemble/wolfson/models/sax_best.pt
hasn't been copied in (gitignored — see README). Runs real inference throughout,
matching this codebase's no-mocking norm — measured ~11ms/call on CPU, fast enough
that faking it would be needless."""

from contextlib import contextmanager
from pathlib import Path

import pytest

import ensemble.wolfson.phrase_generator as wolfson_phrase_generator
from ensemble.director import Director, DirectorSignal, constant_director_source
from ensemble.generators import chord_tone_generator
from ensemble.memory import RehearsalMemory
from ensemble.sax import DEFAULT_MOTIF_STRENGTH, sax_generator
from ensemble.session import Session
from ensemble.timeline import BEATS_PER_BAR
from ensemble.voice import Voice
from ensemble.wolfson.phrase_generator import REST_PITCH
from song import Changes, ChangesEvent, Section, Song, parse_chart
from song.chord import Chord

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "ensemble" / "wolfson" / "models" / "sax_best.pt"
CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"
BASS_REGISTER = (28, 52)
SAX_REGISTER = (55, 79)

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"sax_best.pt not present at {WEIGHTS_PATH} — gitignored, copy it in manually, see README",
)


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


def make_session(seed: int, director: Director = None) -> Session:
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax = Voice(
        id="sax",
        instrument="sax",
        register=SAX_REGISTER,
        source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=seed),
    )
    return Session(song=load_blues(), voices=[bass, sax], directors=[director] if director else [])


def test_same_seed_is_deterministic():
    first = make_session(seed=7).generate()
    second = make_session(seed=7).generate()
    sax_first = [e for e in first if e.voice_id == "sax"]
    sax_second = [e for e in second if e.voice_id == "sax"]
    assert sax_first == sax_second


def test_bar_zero_empty_seed_does_not_crash():
    timeline = make_session(seed=1).generate()
    # No error is the main assertion; also sanity-check bar 0 produced valid events.
    bar0_sax = [e for e in timeline if e.voice_id == "sax" and 0.0 <= e.start_beat < BEATS_PER_BAR]
    for event in bar0_sax:
        assert SAX_REGISTER[0] <= event.pitch <= SAX_REGISTER[1]


def test_sax_events_never_cross_their_own_bar_boundary():
    timeline = make_session(seed=3).generate()
    sax_events = [e for e in timeline if e.voice_id == "sax"]
    assert sax_events, "expected at least one generated sax event across the whole chart"
    for event in sax_events:
        bar_start = (event.start_beat // BEATS_PER_BAR) * BEATS_PER_BAR
        bar_end = bar_start + BEATS_PER_BAR
        assert bar_start <= event.start_beat < bar_end
        assert event.start_beat + event.duration_beats <= bar_end + 1e-9


def test_no_rest_sentinel_ever_becomes_an_event():
    timeline = make_session(seed=3).generate()
    assert all(e.pitch != REST_PITCH for e in timeline)


def average_sax_duration(timeline) -> float:
    sax_notes = [e for e in timeline if e.voice_id == "sax"]
    return sum(e.duration_beats for e in sax_notes) / len(sax_notes)


def test_director_intensity_shifts_average_note_duration():
    """Grounded in a real empirical probe (see the Phase 9 plan): rhythmic_density
    0.0 vs 1.0 produced average note durations of 0.723 vs 0.419 beats over 25
    one-shot calls. 0.1 beats is a generous margin against that ~0.3-beat gap,
    safe even with the different bar-by-bar chord/seed context a real chart
    provides versus that probe's fixed synthetic seed — not flaky."""
    low = Director(id="d", source="ai", signal_source=constant_director_source(0.0))
    high = Director(id="d", source="ai", signal_source=constant_director_source(1.0))

    low_avg = average_sax_duration(make_session(seed=7, director=low).generate())
    high_avg = average_sax_duration(make_session(seed=7, director=high).generate())

    assert low_avg - high_avg >= 0.1


@contextmanager
def spying_on_phrase_generator_calls():
    """Spy on PhraseGenerator.generate's calls — wraps the real method,
    delegates to it, records each call's kwargs, restores the original after.
    Not a mock: the real model still runs every call, matching this codebase's
    no-mocking-framework norm and its own "verify via a spy, not independent
    re-derivation" lesson (see the Phase 8 postmortem). Yields a list of kwargs
    dicts, one per call in order — a call count is just len(calls)."""
    original = wolfson_phrase_generator.PhraseGenerator.generate
    calls = []

    def recording_generate(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        yield calls
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original


def test_plan_buffer_makes_fewer_generate_calls_than_bars_on_blues():
    """blues_in_f.chart changes chord almost every bar (checked directly against
    the chart: F7 Bb7 F7 F7 | Bb7 Bb7 F7 F7 | C7 Bb7 F7 C7 -- only three 2-bar
    same-chord holds in the whole 12-bar form) -- modest savings expected, not
    dramatic. Said plainly: a real, honest limit of what planning can do on
    THIS chart, not a flaw in the mechanism — see the slow-harmonic-rhythm test
    below for where the mechanism's real effect is visible."""
    song = load_blues()
    n_bars = int(song.total_beats // BEATS_PER_BAR)
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax = Voice(
        id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=7),
    )
    with spying_on_phrase_generator_calls() as calls:
        Session(song=song, voices=[bass, sax]).generate()
    assert len(calls) < n_bars


def build_slow_song() -> Song:
    """One chord held for 8 bars (32 beats) — same construction pattern as
    test_transitions.py's own edge-case tests. DEFAULT_PLAN_BARS=4 caps each
    chunk, so this always produces exactly 2 plan chunks per run — used both
    for the call-count test below and the rehearsal-memory wiring tests."""
    return Song(
        title="slow changes", changes=Changes([ChangesEvent(Chord.parse("F7"), 32.0)]),
        form=[Section("A", 1)], tempo_bpm=120,
    )


def make_slow_session(memory=None, seed: int = 7) -> Session:
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax = Voice(
        id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", memory=memory, seed=seed),
    )
    return Session(song=build_slow_song(), voices=[bass, sax])


def test_plan_buffer_makes_far_fewer_calls_on_a_slow_harmonic_rhythm_chart():
    """DEFAULT_PLAN_BARS=4 caps each chunk, so build_slow_song()'s 8-bar hold
    should collapse into exactly 2 generate() calls — a clear demonstration of
    the mechanism's real effect on a chart with a realistic-for-many-tunes
    harmonic rhythm, unlike blues's fast changes above."""
    with spying_on_phrase_generator_calls() as calls:
        make_slow_session().generate()
    assert len(calls) == 2  # ceil(8 bars / plan_bars=4) -- exact, not just "fewer"
    assert len(calls) < 8


def test_memory_supplies_motif_targets_to_a_later_chunk_within_one_run():
    """Within-run persistence: the slow chart produces exactly 2 plan chunks
    (see build_slow_song). The first chunk's memory is empty (nothing stored
    yet) -> empty motif_targets; by the second chunk, the first chunk's notes
    have been stored -> non-empty motif_targets, drawn from what was just
    played. Proves the plumbing, independent of the model's stochastic
    response to it (confirmed separately, empirically, to be a real but rare
    effect — see the Phase 11 plan)."""
    mem = RehearsalMemory()
    with spying_on_phrase_generator_calls() as calls:
        make_slow_session(memory=mem).generate()
    assert len(calls) == 2
    assert calls[0]["motif_targets"] == []
    assert calls[0]["motif_strength"] == 0.0
    assert calls[1]["motif_targets"] != []
    assert calls[1]["motif_strength"] == DEFAULT_MOTIF_STRENGTH


def test_memory_preloads_a_fresh_session_from_a_previous_one():
    """Cross-Session persistence: the actual "rehearsal informs the gig" case.
    One RehearsalMemory, two entirely separate Sessions (fresh Song, fresh
    Voice, fresh sax_generator/PhraseGenerator each time) sharing only the
    memory object. The second Session's very first plan chunk should already
    carry motif_targets from the first Session's material — proof this is
    genuinely preloaded experience, not just same-run carry-over (which the
    test above already covers separately)."""
    mem = RehearsalMemory()
    make_slow_session(memory=mem, seed=1).generate()  # "rehearsal" run

    with spying_on_phrase_generator_calls() as calls:
        make_slow_session(memory=mem, seed=2).generate()  # "gig" run, fresh Session
    assert calls[0]["motif_targets"] != []


def test_memory_stores_a_real_computed_musicality_score():
    """Phase 12: a real end-to-end check that sax_generator actually computes
    and passes a musicality score into memory.store(), not the RehearsalMemory
    default. RehearsalMemory has no public accessor for stored scores (nothing
    else has needed one) -- inspecting _phrases directly is the simplest way
    to check this, same as reaching into "private" state elsewhere in this
    codebase's tests when there's no dedicated API for it yet."""
    mem = RehearsalMemory()
    make_slow_session(memory=mem, seed=1).generate()
    assert len(mem._phrases) == 2  # build_slow_song() always produces exactly 2 chunks
    for entry in mem._phrases:
        assert 0.0 <= entry["score"] <= 1.0
    # Not every chunk's score should coincidentally land on the bare default (1.0)
    # used when no score is passed at all -- proof a real computation happened.
    assert any(entry["score"] != 1.0 for entry in mem._phrases)


def test_director_gesture_toggles_singability_weight():
    """Phase 13: the first real consumer of DirectorSignal.gesture since the
    dial channel was built (Phase 5) -- deterministic, not a re-derivation of
    the musical effect: checks sax_generator's exposed critic_weights directly
    (same "reach into state when there's no dedicated accessor" convention as
    RehearsalMemory._phrases above), matching this codebase's own established
    lesson to verify via the same computation or direct state, not statistics."""
    from gesture.vocabulary import Gesture

    def toggle_on_bar_zero(song, bar_index, timeline):
        gesture = Gesture("toggle_singability") if bar_index == 0 else None
        return DirectorSignal(intensity=0.5, gesture=gesture)

    director = Director(id="teacher", source="ai", signal_source=toggle_on_bar_zero)
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", seed=1)
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)

    assert sax_gen.critic_weights["singability"] != 0.0  # default, before any gesture arrives

    Session(song=load_blues(), voices=[bass, sax], directors=[director]).generate()

    assert sax_gen.critic_weights["singability"] == 0.0  # flipped off by the bar-0 gesture

    # Toggling again (a second Session sharing the same sax_gen closure) flips it back on.
    director2 = Director(id="teacher", source="ai", signal_source=toggle_on_bar_zero)
    bass2 = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax2 = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
    Session(song=load_blues(), voices=[bass2, sax2], directors=[director2]).generate()
    assert sax_gen.critic_weights["singability"] != 0.0


def test_voice_order_does_not_affect_output():
    """Same content regardless of voice order — not the same tie-break order.
    bass and sax both start bar 0 at beat 0.0 (a genuine, expected tie: bass's
    stub plays on the downbeat, and sax's bar-0 seed is empty so its cursor
    also starts at bar_start), and Session.generate() only ever sorts by
    start_beat (ensemble/session.py) — ties break by voice iteration order,
    a documented property of the merge step, not something either generator
    should be expected to override. Comparing sorted-by-full-tuple sidesteps
    that expected tie-break difference and checks the thing this test is
    actually about: identical notes generated either way."""

    def make(reversed_order: bool) -> Session:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax = Voice(
            id="sax",
            instrument="sax",
            register=SAX_REGISTER,
            source="ai",
            generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=9),
        )
        voices = [sax, bass] if reversed_order else [bass, sax]
        return Session(song=load_blues(), voices=voices)

    def sort_key(e):
        return (e.start_beat, e.voice_id, e.pitch, e.velocity, e.duration_beats)

    forward = sorted(make(reversed_order=False).generate().events, key=sort_key)
    reversed_ = sorted(make(reversed_order=True).generate().events, key=sort_key)
    assert forward == reversed_
