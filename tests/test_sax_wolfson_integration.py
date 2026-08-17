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


def make_slow_session(memory=None, seed: int = 7, n_candidates: int = 1) -> Session:
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax = Voice(
        id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", memory=memory, seed=seed, n_candidates=n_candidates),
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


def test_search_makes_n_candidates_generate_calls_per_chunk():
    """Phase 14: n_candidates=5 should produce 5 PhraseGenerator.generate() calls
    for EACH of build_slow_song()'s 2 chunks (10 total), not 5 per bar dispensed
    from a chunk — same cadence as before, just more calls per chunk-build."""
    with spying_on_phrase_generator_calls() as calls:
        make_slow_session(n_candidates=5).generate()
    assert len(calls) == 10  # 2 chunks * 5 candidates


def test_search_picks_the_actual_highest_scoring_candidate():
    """Deterministic, not a re-derivation of the musical effect: a local spy
    captures every candidate's actual notes AND chord_idx/seed_phrase, then
    independently recomputes musicality_score for each and confirms both (a)
    generate.last_candidate_scores matches those recomputed scores exactly, and
    (b) the notes sax_generator actually dispensed correspond to the
    highest-scoring one -- verifying via the same computation, not trusting
    sax_generator's own bookkeeping blindly (the Phase 8 postmortem's lesson)."""
    from ensemble.critic import musicality_score

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []  # (kwargs, returned_notes) per call

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=5, seed=3)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    # generate.last_candidate_scores is OVERWRITTEN (not accumulated) on every
    # chunk-build, so after the full run it reflects only the LAST of
    # build_slow_song()'s 2 chunks (bars 4-7, DEFAULT_PLAN_BARS=4) -- comparing
    # against candidates[:5] (the first chunk) would silently compare the wrong
    # chunk's data. Use the last 5 calls to match what's actually still exposed.
    last_chunk = candidates[-5:]
    recomputed_scores = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase).overall
        for seed_phrase, kwargs, notes in last_chunk
    ]
    assert recomputed_scores == sax_gen.last_candidate_scores

    winner_notes = last_chunk[recomputed_scores.index(max(recomputed_scores))][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
    second_chunk_start = 4 * BEATS_PER_BAR  # DEFAULT_PLAN_BARS=4 -> chunk 2 starts at bar 4
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and second_chunk_start <= e.start_beat < second_chunk_start + BEATS_PER_BAR
    )
    # Dispensed pitches are a subset of the winning candidate's (clipping/register
    # backstop can drop some, per _split_phrase_into_bars) -- never pitches from
    # a losing candidate.
    assert set(dispensed_pitches).issubset(set(winner_pitches))


def test_search_with_one_candidate_matches_unset_behaviour():
    """n_candidates=1 (explicit) must reproduce n_candidates-unset behaviour
    exactly -- the concrete backward-compatibility check, same seed both ways."""
    explicit = make_slow_session(seed=9, n_candidates=1).generate()
    default = make_slow_session(seed=9).generate()
    assert explicit.events == default.events


def test_search_never_does_worse_than_a_single_draw():
    """A real quality comparison, reported honestly either way: search's best
    score for the *same final chunk* should never be WORSE than a single draw's,
    since search always keeps the max among what a single draw would have
    produced anyway. Scoped to build_slow_song()'s last chunk specifically --
    last_candidate_scores reflects only the most recently built chunk (it's
    overwritten, not accumulated, each time a new one is built), not an average
    across the whole run, said plainly rather than overclaimed."""
    single = make_slow_session(seed=11, n_candidates=1)
    single.generate()
    single_score = single.voices[1].generator.last_candidate_scores[0]

    searched = make_slow_session(seed=11, n_candidates=8)
    searched.generate()
    searched_best = max(searched.voices[1].generator.last_candidate_scores)

    assert searched_best >= single_score


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
