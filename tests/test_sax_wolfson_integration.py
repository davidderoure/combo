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


def test_phrasing_varies_across_real_candidates_and_reaches_selection():
    """Phase 23: musicality_score's new phrasing sub-score is a real, active
    ingredient in real selection, not just a correctly-computed pure function
    in isolation -- same spy-and-recompute technique as
    test_search_picks_the_actual_highest_scoring_candidate, but checking the
    phrasing sub-score specifically: real candidates in one chunk show genuine
    variance (not degenerate/constant across the batch -- the concrete proof
    this can actually differentiate candidates during search), and the
    winner's overall (recomputed via the real musicality_score, which now
    includes phrasing automatically) still matches what sax_generator picked
    -- no ensemble/sax.py changes were needed for this, since it only ever
    reads musicality_score(...).overall.

    Winner identification uses the REAL lexicographic selection key
    (-dissonance, motif_adherence, overall), not overall alone -- overall is
    only the final tie-break, and this seed/n_candidates combination does
    show real dissonance variance across the batch (unlike some other tests'
    seeds, where overall alone happens to coincide with the real winner)."""
    from ensemble.critic import dissonance, motif_adherence, musicality_score
    from ensemble.sax import _functional_tonic_scale

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        song = build_slow_song()
        timeline = Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    last_chunk = candidates[-8:]
    scored = [musicality_score(notes, kwargs["chord_idx"], seed_phrase) for seed_phrase, kwargs, notes in last_chunk]

    phrasing_values = [s.phrasing for s in scored]
    assert len(set(round(p, 6) for p in phrasing_values)) > 1  # genuine variance, not degenerate

    recomputed_overall = [s.overall for s in scored]
    assert recomputed_overall == sax_gen.last_candidate_scores

    second_chunk_start = 4 * BEATS_PER_BAR
    functional_scale = _functional_tonic_scale(song, second_chunk_start)
    keys = [
        (-dissonance(notes, kwargs["chord_idx"], extra_tolerated=functional_scale), motif_adherence(notes, []), score.overall)
        for (seed_phrase, kwargs, notes), score in zip(last_chunk, scored)
    ]
    winner_notes = last_chunk[keys.index(max(keys))][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and second_chunk_start <= e.start_beat < second_chunk_start + BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))


def test_search_with_one_candidate_matches_unset_behaviour():
    """n_candidates=1 (explicit) must reproduce n_candidates-unset behaviour
    exactly -- the concrete backward-compatibility check, same seed both ways."""
    explicit = make_slow_session(seed=9, n_candidates=1).generate()
    default = make_slow_session(seed=9).generate()
    assert explicit.events == default.events


def test_search_never_does_worse_than_a_single_draw():
    """A real quality comparison, reported honestly either way: search's best
    score for a chunk should never be WORSE than a single draw's, since search
    always keeps the max among what a single draw would have produced anyway.

    Scoped to the FIRST chunk specifically, captured via a spy rather than
    read off last_candidate_scores (which reflects only the most recently
    built chunk, i.e. build_slow_song()'s SECOND chunk) -- deliberately: by
    the second chunk, the two sessions have each already consumed a
    different number of candidate draws from torch's shared global RNG (1 vs
    8), so their RNG states have diverged and the "max of a superset"
    argument no longer rigorously holds there (found directly, not assumed,
    when Phase 23's phrasing sub-score -- itself sensitive to per-candidate
    rest placement, which is RNG-driven -- tipped a previously-lucky
    comparison the other way). It DOES hold for the first chunk, before any
    divergence: both sessions reseed identically at construction, so
    single's one draw is exactly searched's first draw."""
    from ensemble.critic import musicality_score

    def first_chunk_candidates(n_candidates):
        original = wolfson_phrase_generator.PhraseGenerator.generate
        candidates = []

        def recording_generate(self, seed_phrase, **kwargs):
            notes = original(self, seed_phrase, **kwargs)
            candidates.append((seed_phrase, kwargs, notes))
            return notes

        wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
        try:
            make_slow_session(seed=11, n_candidates=n_candidates).generate()
        finally:
            wolfson_phrase_generator.PhraseGenerator.generate = original
        return candidates[:n_candidates]  # first chunk only

    single_seed_phrase, single_kwargs, single_notes = first_chunk_candidates(1)[0]
    single_score = musicality_score(single_notes, single_kwargs["chord_idx"], single_seed_phrase).overall

    searched_best = max(
        musicality_score(notes, kwargs["chord_idx"], seed_phrase).overall
        for seed_phrase, kwargs, notes in first_chunk_candidates(8)
    )

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


def test_search_with_a_motif_target_prefers_higher_adherence_over_higher_overall():
    """Phase 17: selection uses (motif_adherence, overall) lexicographically, not
    overall alone. Verified the same way as test_search_picks_the_actual_highest_
    scoring_candidate above -- real inference, independently recompute both
    scores for every candidate, confirm the winner is the one with the highest
    adherence among all candidates generated that chunk (and, among ties on
    adherence, the highest overall). Holds regardless of how much real
    stochastic variety this particular run happens to produce. Phase 18 added
    a third, leading key term (dissonance, negated so lower is preferred) --
    this test recomputes the full 3-tuple, not just the Phase 17 2-tuple, so
    it still verifies the ACTUAL selection sax_generator performs rather than
    a stale approximation of it."""
    from ensemble.critic import dissonance, motif_adherence, musicality_score

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []  # (seed_phrase, kwargs, notes) per call

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        mem = RehearsalMemory()
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(
            SAX_REGISTER, target_voice_id="bass", memory=mem, seed=3,
            n_candidates=2, motif_recall_candidates=8,
        )
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    # build_slow_song() always produces exactly 2 chunks (DEFAULT_PLAN_BARS=4 over
    # an 8-bar hold): chunk 1 has nothing recalled yet (n_candidates=2 calls),
    # chunk 2 has memory.recall_motifs() supplying a target
    # (motif_recall_candidates=8 calls) -- 2 + 8 = 10 total.
    assert len(candidates) == 10
    last_chunk = candidates[-8:]
    motif_targets = last_chunk[0][1]["motif_targets"]
    assert motif_targets != []  # the scenario this test is actually about

    recomputed = [
        (
            -dissonance(notes, kwargs["chord_idx"]),
            motif_adherence(notes, motif_targets),
            musicality_score(notes, kwargs["chord_idx"], seed_phrase).overall,
        )
        for seed_phrase, kwargs, notes in last_chunk
    ]
    assert [overall for _dis, _adherence, overall in recomputed] == sax_gen.last_candidate_scores

    best_key = max(recomputed)
    winner_notes = last_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
    second_chunk_start = 4 * BEATS_PER_BAR
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and second_chunk_start <= e.start_beat < second_chunk_start + BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))
    assert sax_gen.dissonance_log[-1] == -best_key[0]
    assert sax_gen.motif_adherence_log[-1] == best_key[1]


def test_motif_recall_candidates_overrides_n_candidates_only_on_recall_chunks():
    """motif_recall_candidates should be used ONLY for a chunk that actually has
    a non-empty motif_targets -- the first of build_slow_song()'s 2 chunks never
    does (nothing recalled yet), the second always does once memory has stored
    the first chunk's motifs."""
    mem = RehearsalMemory()
    with spying_on_phrase_generator_calls() as calls:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax = Voice(
            id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
            generator=sax_generator(
                SAX_REGISTER, target_voice_id="bass", memory=mem, seed=3,
                n_candidates=2, motif_recall_candidates=7,
            ),
        )
        Session(song=build_slow_song(), voices=[bass, sax]).generate()
    assert len(calls) == 2 + 7
    assert calls[0]["motif_targets"] == []
    assert calls[1]["motif_targets"] == []
    assert all(c["motif_targets"] != [] for c in calls[2:])


def test_motif_recall_candidates_unset_reproduces_n_candidates_for_every_chunk():
    """Default motif_recall_candidates=None must reproduce n_candidates exactly
    for every chunk, even one with a non-empty motif_targets -- the concrete
    backward-compatibility check, same discipline as
    test_search_with_one_candidate_matches_unset_behaviour above."""
    mem = RehearsalMemory()
    with spying_on_phrase_generator_calls() as calls:
        make_slow_session(memory=mem, seed=3, n_candidates=2).generate()
    assert len(calls) == 4  # 2 chunks * 2 candidates each -- motif_recall_candidates never overrides


def test_search_prefers_lower_dissonance_even_without_a_motif_target():
    """Phase 18: dissonance-avoidance applies to every chunk, not just ones
    with a recalled motif -- no memory here at all. Real inference,
    independently recompute dissonance for every candidate, confirm the
    winner has the LOWEST dissonance among that chunk's candidates (ties
    broken by overall, matching the real selection key)."""
    from ensemble.critic import dissonance, musicality_score

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(candidates) == 16  # 2 chunks * 8 candidates, no motif_recall_candidates involved
    last_chunk = candidates[-8:]

    recomputed = [
        (-dissonance(notes, kwargs["chord_idx"]), musicality_score(notes, kwargs["chord_idx"], seed_phrase).overall)
        for seed_phrase, kwargs, notes in last_chunk
    ]
    best_key = max(recomputed)
    winner_notes = last_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
    second_chunk_start = 4 * BEATS_PER_BAR
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and second_chunk_start <= e.start_beat < second_chunk_start + BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))
    assert sax_gen.dissonance_log[-1] == -best_key[0]
    # The winner's dissonance is never worse than any candidate's -- the
    # concrete "what's bad matters" proof, not just "some candidate was picked".
    assert sax_gen.dissonance_log[-1] == min(-k[0] for k in recomputed)


def test_director_gesture_toggles_dissonance_mode():
    """Phase 20: mirrors test_director_gesture_toggles_singability_weight above
    exactly -- deterministic, checking sax_generator's exposed dissonance_mode
    directly rather than re-deriving the effect statistically."""
    from gesture.vocabulary import Gesture

    def toggle_on_bar_zero(song, bar_index, timeline):
        gesture = Gesture("toggle_dissonance_avoidance") if bar_index == 0 else None
        return DirectorSignal(intensity=0.5, gesture=gesture)

    director = Director(id="teacher", source="ai", signal_source=toggle_on_bar_zero)
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", seed=1)
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)

    assert sax_gen.dissonance_mode["enabled"] is True  # default, before any gesture arrives

    Session(song=load_blues(), voices=[bass, sax], directors=[director]).generate()

    assert sax_gen.dissonance_mode["enabled"] is False  # flipped off by the bar-0 gesture

    # Toggling again (a second Session sharing the same sax_gen closure) flips it back on.
    director2 = Director(id="teacher", source="ai", signal_source=toggle_on_bar_zero)
    bass2 = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax2 = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
    Session(song=load_blues(), voices=[bass2, sax2], directors=[director2]).generate()
    assert sax_gen.dissonance_mode["enabled"] is True


def test_dissonance_mode_disabled_reverts_to_overall_only_selection():
    """The concrete proof the toggle changes real selection outcomes, not just
    a flag nobody reads: with dissonance_mode explicitly disabled, the winning
    candidate is chosen by (motif_adherence, overall) alone -- same
    spy-and-recompute technique as test_search_prefers_lower_dissonance_even_
    without_a_motif_target, but asserting the OPPOSITE outcome now that the
    gate is off."""
    from ensemble.critic import dissonance, motif_adherence, musicality_score

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=5)
        sax_gen.dissonance_mode["enabled"] = False
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(candidates) == 16  # 2 chunks * 8 candidates
    last_chunk = candidates[-8:]

    overall_only = [
        (motif_adherence(notes, []), musicality_score(notes, kwargs["chord_idx"], seed_phrase).overall)
        for seed_phrase, kwargs, notes in last_chunk
    ]
    best_overall_only = max(overall_only)
    winner_notes = last_chunk[overall_only.index(best_overall_only)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)

    second_chunk_start = 4 * BEATS_PER_BAR
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and second_chunk_start <= e.start_beat < second_chunk_start + BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))

    # dissonance_log still reflects the real value even though it wasn't the
    # deciding factor -- Phase 20's "always logged" guarantee, checked
    # directly rather than assumed.
    recomputed_dissonances = [dissonance(notes, kwargs["chord_idx"]) for _sp, kwargs, notes in last_chunk]
    assert sax_gen.dissonance_log[-1] == recomputed_dissonances[overall_only.index(best_overall_only)]


def test_credit_resolved_tension_reaches_real_selection():
    """Phase 22: with credit_resolved_tension=True, the winning candidate is
    the one selection actually picks under dissonance(..., credit_resolved_
    tension=True) -- verified by recomputing the EXACT same key sax_generator
    uses on real candidates (same spy-and-recompute technique as
    test_search_prefers_lower_dissonance_even_without_a_motif_target), not
    trusting sax_generator's own bookkeeping. Deterministic regardless of
    whether any one candidate happens to contain a genuine resolved-tension
    shape (Phase 11's own honest precedent: a targeted melodic device can be
    real but rare in stochastic output) -- this proves the True path is
    actually wired into real selection, not that a specific shape occurs."""
    from ensemble.critic import dissonance, musicality_score
    from ensemble.sax import _functional_tonic_scale

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=5, credit_resolved_tension=True)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        song = build_slow_song()
        timeline = Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(candidates) == 16  # 2 chunks * 8 candidates
    last_chunk = candidates[-8:]
    second_chunk_start = 4 * BEATS_PER_BAR
    functional_scale = _functional_tonic_scale(song, second_chunk_start)
    assert functional_scale == frozenset()  # one chord the whole song -- no ii-V-I context here

    recomputed = [
        (
            -dissonance(notes, kwargs["chord_idx"], extra_tolerated=functional_scale, credit_resolved_tension=True),
            musicality_score(notes, kwargs["chord_idx"], seed_phrase).overall,
        )
        for seed_phrase, kwargs, notes in last_chunk
    ]
    best_key = max(recomputed)
    winner_notes = last_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and second_chunk_start <= e.start_beat < second_chunk_start + BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))
    assert sax_gen.dissonance_log[-1] == -best_key[0]

    # The True-crediting formula genuinely differs from the plain (uncredited)
    # one for at least one real candidate in this chunk -- proof the flag
    # isn't a no-op on real generated data, not just a correctly-wired but
    # vacuous parameter.
    plain = [dissonance(notes, kwargs["chord_idx"], extra_tolerated=functional_scale) for _sp, kwargs, notes in last_chunk]
    credited = [-k[0] for k in recomputed]
    assert any(c < p for c, p in zip(credited, plain))


def build_ii_v_i_song() -> Song:
    """A genuine 1-bar-per-chord Dm7-G7-Cmaj7 turnaround -- bar-granular,
    matching _ii_v_i_target's own bar-level lookup exactly, so bar 0 (the ii)
    is unambiguously recognised."""
    return Song(
        title="ii-V-I", changes=Changes([
            ChangesEvent(Chord.parse("Dm7"), BEATS_PER_BAR),
            ChangesEvent(Chord.parse("G7"), BEATS_PER_BAR),
            ChangesEvent(Chord.parse("Cmaj7"), BEATS_PER_BAR),
        ]),
        form=[Section("A", 4)], tempo_bpm=120,
    )


def test_functional_context_reaches_real_selection_over_the_ii_chord():
    """Phase 21, Lever E: a real, non-trivial extra tolerance is added at the
    ii chord (checked directly: functional_scale is a STRICT superset of
    dissonance_scale(Dm7's own chord_idx) alone -- D-dorian is missing
    exactly the b6 pc 8 that C-major-widened has, matching the hand
    computation in dissonance_scale's own docstring), and the actual
    dispensed candidate is the one selection -- using functional context --
    actually picks, verified by recomputing the exact same key sax_generator
    uses, not trusting its own bookkeeping (the Phase 8 postmortem's lesson,
    applied again here)."""
    from ensemble.critic import dissonance, dissonance_scale as critic_dissonance_scale, motif_adherence, musicality_score
    from ensemble.sax import _functional_tonic_scale, chord_to_wolfson_index

    song = build_ii_v_i_song()
    dm7_idx = chord_to_wolfson_index(Chord.parse("Dm7"))
    functional_scale = _functional_tonic_scale(song, 0.0)

    # The real, non-trivial extra tolerance this phase claims to add.
    assert functional_scale > critic_dissonance_scale(dm7_idx)
    assert functional_scale - critic_dissonance_scale(dm7_idx) == {8}  # the b6, Ab

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=11)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    # bar 0 (the ii, Dm7) is its own chunk: chord changes every bar on this
    # chart, so _bars_until_chord_change caps it at span_bars=1.
    first_chunk = [c for c in candidates if c[1]["chord_idx"] == dm7_idx][:8]
    assert len(first_chunk) == 8

    recomputed = [
        (
            -dissonance(notes, dm7_idx, extra_tolerated=functional_scale),
            motif_adherence(notes, []),
            musicality_score(notes, dm7_idx, seed_phrase).overall,
        )
        for seed_phrase, kwargs, notes in first_chunk
    ]
    best_key = max(recomputed)
    winner_notes = first_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)

    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and 0.0 <= e.start_beat < BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))
