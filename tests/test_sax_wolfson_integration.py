"""Integration tests for ensemble/sax.py against the real Wolfson-adapted model —
DESIGN.md §12, Phase 8. First test file in this codebase needing a binary artifact
absent from a fresh clone: skips cleanly if ensemble/wolfson/models/sax_best.pt
hasn't been copied in (gitignored — see README). Runs real inference throughout,
matching this codebase's no-mocking norm — measured ~11ms/call on CPU, fast enough
that faking it would be needless."""

from contextlib import contextmanager
from pathlib import Path

import pytest

import wjd_corpus
import ensemble.wolfson.phrase_generator as wolfson_phrase_generator
from ensemble.corpus_motifs import CorpusMotifs
from ensemble.director import Director, DirectorSignal, constant_director_source
from ensemble.generators import chord_tone_generator
from ensemble.memory import RehearsalMemory
from ensemble.sax import (
    DEFAULT_MOTIF_STRENGTH,
    PHRASE_BOUNDARY_REST_BEATS,
    REGISTER_BALANCE_HALF_LIFE_BEATS,
    _decay_pitch_weighted,
    sax_generator,
)
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

# Phase 29: a SECOND, additional skip condition -- only the corpus_familiarity
# real-consumer tests need the WJD motif cache built (python wjd_corpus.py
# --build), not every test in this file. Stacked on top of the module-level
# pytestmark above (either skip reason applies independently).
needs_corpus_cache = pytest.mark.skipif(
    not wjd_corpus.CACHE_PATH.exists(),
    reason=f"WJD corpus cache not present at {wjd_corpus.CACHE_PATH} — run 'python wjd_corpus.py --build' first",
)


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


def _dispensed_pitch_range(timeline, voice_id: str, until_beat: float):
    """Phase 32: the real (low, high) pitch bounds a voice actually dispensed
    strictly before until_beat -- used to recompute the real prior_range a
    later chunk's candidates were scored against, the same value
    ensemble/sax.py's own_pitch_range tracks internally. None if nothing was
    dispensed yet (mirrors register_usage's own prior_range=None case)."""
    pitches = [e.pitch for e in timeline if e.voice_id == voice_id and e.start_beat < until_beat]
    return (min(pitches), max(pitches)) if pitches else None


def _dispensed_pitch_mean_beats(timeline, voice_id: str, until_beat: float):
    """Phase 36: the real (weighted_pitch_sum, total_beats) a voice actually
    dispensed strictly before until_beat -- used to recompute the real
    prior_mean_beats a later chunk's candidates were scored against, the same
    value ensemble/sax.py's own_pitch_weighted tracks internally. None if
    nothing was dispensed yet (mirrors register_balance's own
    prior_mean_beats=None case), same style as _dispensed_pitch_range above."""
    events = [e for e in timeline if e.voice_id == voice_id and e.start_beat < until_beat]
    total_beats = sum(e.duration_beats for e in events)
    if total_beats <= 0:
        return None
    return (sum(e.pitch * e.duration_beats for e in events), total_beats)


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


def test_own_voice_id_adds_self_history_to_the_seed_on_the_second_chunk():
    """Phase 37: build_slow_song() always produces exactly 2 chunks
    (DEFAULT_PLAN_BARS=4 over an 8-bar hold). Chunk 1 (bar 0) has nothing
    dispensed by anyone yet, so its seed_phrase should be empty, same as
    today (own_voice_id changes nothing when there's no history to add).
    Chunk 2's seed_phrase should be the bass's recent notes FOLLOWED BY the
    sax's own real dispensed notes from bars 0-3 -- verified via the same
    computation (_build_seed_phrase, already trusted), not re-derived by
    hand, the established discipline throughout this file."""
    from ensemble.sax import _build_seed_phrase

    original = wolfson_phrase_generator.PhraseGenerator.generate
    seed_phrases = []

    def recording_generate(self, seed_phrase, **kwargs):
        seed_phrases.append(seed_phrase)
        return original(self, seed_phrase, **kwargs)

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", own_voice_id="sax", n_candidates=1, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(seed_phrases) == 2  # n_candidates=1, 2 chunks
    first_seed, second_seed = seed_phrases
    assert first_seed == []  # bar 0 -- nothing dispensed by anyone yet

    second_chunk_start = 4 * BEATS_PER_BAR  # DEFAULT_PLAN_BARS=4
    since_beat = 2 * BEATS_PER_BAR  # lookback_bars=2 default
    expected_bass = _build_seed_phrase(timeline, "bass", since_beat, second_chunk_start)
    expected_own = _build_seed_phrase(timeline, "sax", since_beat, second_chunk_start)
    assert expected_own  # the real proof self-history exists to add
    assert second_seed == expected_bass + expected_own


def test_phrase_boundary_rest_creates_a_real_gap_at_the_second_chunk():
    """Phase 39: build_slow_song() always produces exactly 2 chunks. The
    first chunk's earliest dispensed sax event starts at beat 0.0 -- no
    boundary rest before the very first phrase of the performance. The
    second chunk's earliest dispensed sax event must start strictly after
    the chunk's own start beat, by at least min(PHRASE_BOUNDARY_REST_BEATS)
    -- a real, structural gap."""
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5)
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
    timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()

    sax_events = sorted([e for e in timeline if e.voice_id == "sax"], key=lambda e: e.start_beat)
    assert sax_events[0].start_beat == 0.0  # no boundary rest before the very first phrase

    second_chunk_start = 4 * BEATS_PER_BAR  # DEFAULT_PLAN_BARS=4
    second_chunk_events = [e for e in sax_events if e.start_beat >= second_chunk_start]
    assert second_chunk_events  # a real, non-empty second chunk
    gap = second_chunk_events[0].start_beat - second_chunk_start
    assert gap >= min(PHRASE_BOUNDARY_REST_BEATS)


def test_chunks_dispensed_counts_real_chunk_builds():
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5)
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
    Session(song=build_slow_song(), voices=[bass, sax]).generate()
    assert sax_gen.chunks_dispensed["count"] == 2  # build_slow_song() always produces exactly 2 chunks


def _drive_bars(sax_gen, song, n_bars: int, force_wait_at: int = None):
    """Phase 43 test helper: manually drives sax_gen bar-by-bar (mirroring
    Session.generate()'s own per-bar mechanics: a prior-bars-only Timeline
    snapshot, voice_id stamped onto returned events) so a specific bar can
    force sax_gen.laying_out active -- Session.generate() itself has no hook
    for that. Returns the accumulated Timeline."""
    from dataclasses import replace

    from ensemble.timeline import Timeline

    timeline = Timeline()
    director_signal = DirectorSignal()
    for bar_index in range(n_bars):
        prior = Timeline(list(timeline.events))
        if bar_index == force_wait_at:
            sax_gen.laying_out["active"] = True
            sax_gen.laying_out["bars_waited"] = 0
        for event in sax_gen(song, bar_index, prior, director_signal):
            timeline.add(replace(event, voice_id="sax"))
    return timeline


def test_lay_out_for_cue_lands_precisely_with_no_boundary_rest_when_resuming():
    """Phase 43: force a wait right before a real no-change bar (bar 3 on
    blues_in_f.chart), confirm genuine silence there, and confirm the FIRST
    note of the next real chord-change bar (bar 4) lands exactly on its own
    downbeat -- no Phase 39 boundary rest offset, the concrete
    resumed_from_wait suppression proof."""
    song = load_blues()
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5, plan_bars=1)
    timeline = _drive_bars(sax_gen, song, n_bars=5, force_wait_at=3)

    bar3_events = [e for e in timeline if 3 * BEATS_PER_BAR <= e.start_beat < 4 * BEATS_PER_BAR]
    assert bar3_events == []  # genuine silence -- still waiting

    bar4_events = sorted([e for e in timeline if e.start_beat >= 4 * BEATS_PER_BAR], key=lambda e: e.start_beat)
    assert bar4_events  # a real chord change (F7->Bb7) -- resumed here
    assert bar4_events[0].start_beat == 4 * BEATS_PER_BAR  # exactly on the downbeat, no rest offset


def test_lay_out_for_cue_safety_cap_resumes_after_max_bars():
    """Phase 43: build_slow_song() never changes chord, so a forced wait can
    never find a real cue -- must resume via MAX_LAY_OUT_BARS regardless,
    not wait indefinitely."""
    from ensemble.sax import MAX_LAY_OUT_BARS

    song = build_slow_song()
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5, plan_bars=1)
    _drive_bars(sax_gen, song, n_bars=MAX_LAY_OUT_BARS + 3, force_wait_at=1)

    assert sax_gen.laying_out["active"] is False
    assert sax_gen.laying_out["bars_waited"] == MAX_LAY_OUT_BARS


def test_lay_out_for_cue_produces_real_silence_over_blues_without_forcing_state():
    """Phase 43: with lay_out_for_cue_probability=1.0 (deterministic entry
    whenever eligible, no seed-hunting needed), a REAL, unforced
    Session.generate() run over blues_in_f.chart shows genuine silence at
    every real no-change bar (3, 5, 7) and real playing resumes at the
    following real chord-change bars -- the concrete proof this engages for
    real, not only when manually forced."""
    song = load_blues()
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax_gen = sax_generator(
        SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5, plan_bars=1, lay_out_for_cue_probability=1.0,
    )
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
    timeline = Session(song=song, voices=[bass, sax]).generate()
    sax_events = [e for e in timeline if e.voice_id == "sax"]

    for no_change_bar in (3, 5, 7):
        lo, hi = no_change_bar * BEATS_PER_BAR, (no_change_bar + 1) * BEATS_PER_BAR
        assert [e for e in sax_events if lo <= e.start_beat < hi] == []

    for resume_bar in (4, 6, 8):
        lo, hi = resume_bar * BEATS_PER_BAR, (resume_bar + 1) * BEATS_PER_BAR
        assert [e for e in sax_events if lo <= e.start_beat < hi] != []


def test_lay_out_for_cue_probability_zero_reproduces_existing_behaviour():
    """Phase 43: the default (0.0) must reproduce exactly what sax_generator
    already did before this phase -- the real backward-compatibility check."""
    song = load_blues()

    def run(lay_out_kwargs):
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5, plan_bars=1, **lay_out_kwargs)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate()
        return [e for e in timeline if e.voice_id == "sax"]

    explicit_zero = run({"lay_out_for_cue_probability": 0.0})
    unset = run({})
    assert explicit_zero == unset


def test_phrase_boundary_rest_duration_is_deterministic_given_a_seed():
    def dispensed_gap_at_second_chunk():
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
        sax_events = sorted([e for e in timeline if e.voice_id == "sax"], key=lambda e: e.start_beat)
        second_chunk_start = 4 * BEATS_PER_BAR
        first_second_chunk_event = [e for e in sax_events if e.start_beat >= second_chunk_start][0]
        return first_second_chunk_event.start_beat - second_chunk_start

    assert dispensed_gap_at_second_chunk() == dispensed_gap_at_second_chunk()


def test_chord_change_landing_inactive_without_a_real_chord_change():
    """Phase 41: build_slow_song() is a single 8-bar F7 hold -- chunk 2 has
    the SAME chord_idx as chunk 1 (just capped by plan_bars, not a real
    harmonic change), so generate.landing_log's second entry must stay 0.0
    -- the concrete "gate correctly stays off" proof."""
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=3, seed=5)
    sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
    Session(song=build_slow_song(), voices=[bass, sax]).generate()

    assert len(sax_gen.landing_log) == 2  # 2 chunks
    assert sax_gen.landing_log == [0.0, 0.0]


def test_chord_change_landing_reaches_real_selection_on_a_real_chord_change():
    """Phase 41: build_ii_v_i_song() changes chord every bar, so chunk 2
    (Dm7 -> G7) follows a genuine chord change -- verified directly via
    song.chord_at, not assumed. Real candidates show genuine landing
    variance, and the winner identified via the real 5-term lexicographic
    key matches what sax_generator actually dispensed."""
    from ensemble.critic import chord_change_landing, dissonance, motif_adherence, musicality_score
    from ensemble.sax import _functional_tonic_scale, chord_to_wolfson_index

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        song = build_ii_v_i_song()
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    # Real chord change, verified directly -- chunk 2 starts at bar 1.
    chunk1_chord = chord_to_wolfson_index(song.chord_at(0.0))
    chunk2_start = BEATS_PER_BAR
    chunk2_chord = chord_to_wolfson_index(song.chord_at(chunk2_start))
    assert chunk1_chord != chunk2_chord  # Dm7 -> G7

    last_chunk = candidates[-8:]
    landing_values = [chord_change_landing(notes, kwargs["chord_idx"]) for _sp, kwargs, notes in last_chunk]
    assert len(set(landing_values)) > 1  # genuine variance, not degenerate

    functional_scale = _functional_tonic_scale(song, chunk2_start)
    scored = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, extra_tolerated=functional_scale)
        for seed_phrase, kwargs, notes in last_chunk
    ]
    keys = [
        (
            -dissonance(notes, kwargs["chord_idx"], extra_tolerated=functional_scale),
            motif_adherence(notes, []),
            landing,
            0.0,  # corpus_score -- no corpus configured
            score.overall,
        )
        for (seed_phrase, kwargs, notes), landing, score in zip(last_chunk, landing_values, scored)
    ]
    winner_notes = last_chunk[keys.index(max(keys))][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and chunk2_start <= e.start_beat < chunk2_start + BEATS_PER_BAR
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))


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
    second_chunk_start = 4 * BEATS_PER_BAR  # DEFAULT_PLAN_BARS=4 -> chunk 2 starts at bar 4
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)  # Phase 32
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)  # Phase 36
    last_chunk = candidates[-5:]
    recomputed_scores = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats).overall
        for seed_phrase, kwargs, notes in last_chunk
    ]
    assert recomputed_scores == sax_gen.last_candidate_scores

    winner_notes = last_chunk[recomputed_scores.index(max(recomputed_scores))][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
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

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)  # Phase 32
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)  # Phase 36
    last_chunk = candidates[-8:]
    scored = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats)
        for seed_phrase, kwargs, notes in last_chunk
    ]

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


def test_register_usage_varies_across_real_candidates_and_reaches_selection():
    """Phase 24: same proof as test_phrasing_varies_across_real_candidates_and_
    reaches_selection above, for the new register_usage sub-score -- real
    candidates in one chunk show genuine variance (not degenerate/constant),
    and the winner identified via the REAL lexicographic selection key
    (-dissonance, motif_adherence, overall) matches what sax_generator
    actually dispensed."""
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

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)  # Phase 32
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)  # Phase 36
    last_chunk = candidates[-8:]
    scored = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats)
        for seed_phrase, kwargs, notes in last_chunk
    ]

    register_usage_values = [s.register_usage for s in scored]
    assert len(set(round(r, 6) for r in register_usage_values)) > 1  # genuine variance, not degenerate

    recomputed_overall = [s.overall for s in scored]
    assert recomputed_overall == sax_gen.last_candidate_scores

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


def test_sustain_quality_varies_across_real_candidates_and_reaches_selection():
    """Phase 40: same proof as test_register_usage_varies_across_real_candidates_
    and_reaches_selection above, for the new sustain_quality sub-score -- real
    candidates in one chunk show genuine variance, and the winner identified via
    the REAL lexicographic selection key matches what sax_generator actually
    dispensed."""
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

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)
    last_chunk = candidates[-8:]
    scored = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats)
        for seed_phrase, kwargs, notes in last_chunk
    ]

    sustain_quality_values = [s.sustain_quality for s in scored]
    assert len(set(round(v, 6) for v in sustain_quality_values)) > 1  # genuine variance, not degenerate

    recomputed_overall = [s.overall for s in scored]
    assert recomputed_overall == sax_gen.last_candidate_scores

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


def test_directional_naturalness_varies_across_real_candidates_and_reaches_selection():
    """Phase 42: same proof as test_sustain_quality_varies_across_real_candidates_
    and_reaches_selection above, for the new directional_naturalness sub-score --
    real candidates in one chunk show genuine variance, and the winner identified
    via the REAL lexicographic selection key matches what sax_generator actually
    dispensed."""
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

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)
    last_chunk = candidates[-8:]
    scored = [
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats)
        for seed_phrase, kwargs, notes in last_chunk
    ]

    directional_values = [s.directional_naturalness for s in scored]
    assert len(set(round(v, 6) for v in directional_values)) > 1  # genuine variance, not degenerate

    recomputed_overall = [s.overall for s in scored]
    assert recomputed_overall == sax_gen.last_candidate_scores

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


def test_sustain_quality_uses_quartal_not_tertian_tones_on_a_modal_chart():
    """Phase 40: song.modal=True reaches sustain_quality's own modal parameter
    in real selection -- confirmed two ways: quartal and tertian scoring
    genuinely differ for at least one real candidate (not a no-op), and the
    real sax_generator's own bookkeeping (last_candidate_scores) matches
    independent recomputation using modal=True (quartal), not modal=False."""
    from ensemble.critic import musicality_score, sustain_quality

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        song = build_slow_song()
        song.modal = True
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    last_chunk = candidates[-8:]
    quartal_values = [sustain_quality(notes, kwargs["chord_idx"], modal=True) for _sp, kwargs, notes in last_chunk]
    tertian_values = [sustain_quality(notes, kwargs["chord_idx"], modal=False) for _sp, kwargs, notes in last_chunk]
    assert quartal_values != tertian_values  # genuinely differs for at least one real candidate

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)
    recomputed_quartal_overall = [
        musicality_score(
            notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, modal=True,
            prior_range=prior_range, prior_mean_beats=prior_mean_beats,
        ).overall
        for seed_phrase, kwargs, notes in last_chunk
    ]
    assert recomputed_quartal_overall == sax_gen.last_candidate_scores


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
    single_score = musicality_score(single_notes, single_kwargs["chord_idx"], single_seed_phrase, SAX_REGISTER).overall

    searched_best = max(
        musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER).overall
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

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)  # Phase 32
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)  # Phase 36
    recomputed = [
        (
            -dissonance(notes, kwargs["chord_idx"]),
            motif_adherence(notes, motif_targets),
            musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats).overall,
        )
        for seed_phrase, kwargs, notes in last_chunk
    ]
    assert [overall for _dis, _adherence, overall in recomputed] == sax_gen.last_candidate_scores

    best_key = max(recomputed)
    winner_notes = last_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)
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
        (-dissonance(notes, kwargs["chord_idx"]), musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER).overall)
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

    second_chunk_start = 4 * BEATS_PER_BAR
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)  # Phase 32
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)  # Phase 36
    overall_only = [
        (
            motif_adherence(notes, []),
            musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER, prior_range=prior_range, prior_mean_beats=prior_mean_beats).overall,
        )
        for seed_phrase, kwargs, notes in last_chunk
    ]
    best_overall_only = max(overall_only)
    winner_notes = last_chunk[overall_only.index(best_overall_only)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)

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
    prior_range = _dispensed_pitch_range(timeline, "sax", second_chunk_start)  # Phase 32
    prior_mean_beats = _dispensed_pitch_mean_beats(timeline, "sax", second_chunk_start)  # Phase 36

    recomputed = [
        (
            -dissonance(notes, kwargs["chord_idx"], extra_tolerated=functional_scale, credit_resolved_tension=True),
            musicality_score(
                notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER,
                prior_range=prior_range, prior_mean_beats=prior_mean_beats,
            ).overall,
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
    ii chord (checked directly: functional_scale adds exactly the b6 pc 8
    that C-major-widened has and D-dorian-widened doesn't, matching the hand
    computation in dissonance_scale's own docstring), and the actual
    dispensed candidate is the one selection -- using functional context --
    actually picks, verified by recomputing the exact same key sax_generator
    uses, not trusting its own bookkeeping (the Phase 8 postmortem's lesson,
    applied again here).

    Note (Phase 36): functional_scale is no longer a STRICT SUPERSET of
    dissonance_scale(dm7_idx) alone -- Phase 36 gave minor chords their own
    always-on extra pitch class (pc 1, a chromatic approach tone into the
    root) that functional_scale (built from the I chord's own widened scale,
    Cmaj7, which has no reason to include a tone specific to D-dorian) never
    had. The set-DIFFERENCE check below is what actually matters for this
    test's claim (a real, non-trivial NEW tolerance from functional context)
    and is unaffected by that -- pc 8 was never in D-dorian's scale, widened
    or not, so it stays a genuine, functional-context-only addition."""
    from ensemble.critic import dissonance, dissonance_scale as critic_dissonance_scale, motif_adherence, musicality_score
    from ensemble.sax import _functional_tonic_scale, chord_to_wolfson_index

    song = build_ii_v_i_song()
    dm7_idx = chord_to_wolfson_index(Chord.parse("Dm7"))
    functional_scale = _functional_tonic_scale(song, 0.0)

    # The real, non-trivial extra tolerance this phase claims to add.
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
            musicality_score(notes, dm7_idx, seed_phrase, SAX_REGISTER).overall,
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


def test_minor_chromatic_approach_tone_reaches_real_selection_over_dm7():
    """Phase 36: real proof the minor-chord scale widening (a chromatic
    approach tone into the root -- pc 1 for Dm7, see _widened_mode_scale's
    own docstring) isn't a no-op on real generated data, mirroring
    test_credit_resolved_tension_reaches_real_selection's own "genuinely
    differs for at least one real candidate" structure. There's no public
    on/off toggle for this lever (unlike credit_resolved_tension), so this
    reimplements dissonance()'s exact clash-counting loop -- reusing its own
    private helpers directly, not reinventing them -- against the OLD,
    pre-Phase-36 scale reference (plain dorian, no pc 1) to compute what
    dissonance WOULD have reported for each real candidate, then compares
    against the real, current dissonance()."""
    from ensemble.critic import (
        DISSONANT_SEMITONE_DISTANCE, _is_passing_tone, _real_notes, _semitones_to_scale,
        chord_root, chord_to_mode, dissonance, dissonance_scale as critic_dissonance_scale, scale_pitch_classes,
    )
    from ensemble.sax import chord_to_wolfson_index

    def old_style_dissonance(notes, plain_scale):
        real = _real_notes(notes)
        if not real:
            return 0.0
        clashes = 0
        for i, n in enumerate(real):
            pc = n["pitch"] % 12
            if pc in plain_scale:
                continue
            if _semitones_to_scale(pc, plain_scale) != DISSONANT_SEMITONE_DISTANCE:
                continue
            if _is_passing_tone(real, i):
                continue
            clashes += 1
        return clashes / len(real)

    song = build_ii_v_i_song()
    dm7_idx = chord_to_wolfson_index(Chord.parse("Dm7"))
    plain_dorian = scale_pitch_classes(chord_root(dm7_idx), chord_to_mode(dm7_idx))
    assert 1 not in plain_dorian
    assert 1 in critic_dissonance_scale(dm7_idx)  # the real, active new tolerance

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
        Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    dm7_bar_candidates = [notes for _sp, kwargs, notes in candidates if kwargs["chord_idx"] == dm7_idx]
    assert dm7_bar_candidates  # a real, non-empty batch over the Dm7 bar

    new_scores = [dissonance(notes, dm7_idx) for notes in dm7_bar_candidates]
    old_scores = [old_style_dissonance(notes, plain_dorian) for notes in dm7_bar_candidates]
    # Genuinely differs for at least one real candidate -- proof the widening
    # isn't a no-op on real generated data.
    assert any(new < old for new, old in zip(new_scores, old_scores))
    assert all(new <= old for new, old in zip(new_scores, old_scores))  # never worse under the new scale


def test_chord_tagged_recall_reaches_real_selection_over_a_multi_quality_chart():
    """Phase 25: build_ii_v_i_song() cycles Dm7(minor)-G7(dominant)-Cmaj7(major)
    every bar across its 4 choruses (12 bars total; chord changes every bar so
    each chunk spans exactly span_bars=1, one PhraseGenerator.generate() call
    each with n_candidates=1 default) -- a real, multi-quality chart, unlike
    blues_in_f.chart (checked directly in the plan: every chord there is a
    dominant 7th, so chord-tagged recall has nothing to differentiate on it).
    Same spy-and-recompute discipline as every other real-consumer test here:
    for every chunk that got a non-empty motif_target, independently replay
    what recall_motifs(chord_quality=...) would have returned using ONLY that
    quality's phrases stored so far (a fresh RehearsalMemory populated from a
    slice of the real mem._phrases, not re-derived by a different method), and
    confirm the target sax_generator actually used matches -- not just that
    _some_ target was picked."""
    from ensemble.sax import _pick_achievable_motif, chord_to_wolfson_index
    from ensemble.wolfson.chords import QUAL_DOM, QUAL_MAJOR, QUAL_MINOR

    original = wolfson_phrase_generator.PhraseGenerator.generate
    calls = []  # (chord_idx, motif_targets) per generate() call, in order

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        calls.append((kwargs["chord_idx"], kwargs.get("motif_targets") or []))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        mem = RehearsalMemory()
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", memory=mem, seed=4)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        Session(song=build_ii_v_i_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(calls) == 12  # one chunk per bar -- chord changes every bar, n_candidates=1
    assert len(mem._phrases) == 12

    dm7_idx = chord_to_wolfson_index(Chord.parse("Dm7"))
    g7_idx = chord_to_wolfson_index(Chord.parse("G7"))
    cmaj7_idx = chord_to_wolfson_index(Chord.parse("Cmaj7"))
    quality_by_chord_idx = {dm7_idx: QUAL_MINOR, g7_idx: QUAL_DOM, cmaj7_idx: QUAL_MAJOR}

    # Stored chord_quality matches the actual chord each chunk was generated over.
    for i, (chord_idx, _targets) in enumerate(calls):
        assert mem._phrases[i]["chord_quality"] == quality_by_chord_idx[chord_idx]

    # The real proof: recompute, per chunk, exactly what recall_motifs(
    # chord_quality=...) would have returned from ONLY that quality's
    # phrases stored so far, and confirm the real motif_targets match.
    for i, (chord_idx, targets) in enumerate(calls):
        quality = quality_by_chord_idx[chord_idx]
        replay = RehearsalMemory()
        replay._phrases = list(mem._phrases[:i])
        expected_pick = _pick_achievable_motif(replay.recall_motifs(chord_quality=quality))
        assert targets == ([expected_pick] if expected_pick is not None else [])

    # Not vacuous: at least one chunk actually got a real, non-empty target
    # from same-quality history.
    assert any(targets for _chord_idx, targets in calls)


def test_disk_persistence_crosses_a_real_process_boundary(tmp_path):
    """Phase 26: the genuine 'crosses a process boundary' proof no other test
    provides -- two separate, independent RehearsalMemory OBJECTS (not the
    same one reused, simulating two separate self_test.py --persist
    invocations on separate days) sharing one JSON file on disk. Rehearsal 2's
    first chunk getting a real motif_target is only possible if rehearsal 1's
    save and rehearsal 2's load both actually worked end-to-end."""
    path = tmp_path / "blues_in_f.json"

    memory_1 = RehearsalMemory(persist_path=path)
    make_slow_session(memory=memory_1, seed=9).generate()
    assert path.exists()  # rehearsal 1 actually wrote something

    original = wolfson_phrase_generator.PhraseGenerator.generate
    first_chunk_targets = []

    def recording_generate(self, seed_phrase, **kwargs):
        if not first_chunk_targets:
            first_chunk_targets.append(kwargs.get("motif_targets") or [])
        return original(self, seed_phrase, **kwargs)

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        memory_2 = RehearsalMemory(persist_path=path)  # a brand-new object -- simulates a fresh process
        make_slow_session(memory=memory_2, seed=9).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert first_chunk_targets[0] != []  # rehearsal 2's first chunk recalled something from disk


def test_credit_resolved_tension_reaches_tonal_conformity_in_real_selection():
    """Phase 27: credit_resolved_tension no longer only helps a candidate
    survive the dissonance gate -- it now also raises tonal_conformity for
    the exact same note, verified on real generated candidates (seed=5,
    already established in Phase 22's own test to produce a real
    resolved-tension shape), not just the hand-built unit-test example."""
    from ensemble.critic import tonal_conformity

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
        Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    last_chunk = candidates[-8:]
    without = [tonal_conformity(notes, kwargs["chord_idx"]) for _sp, kwargs, notes in last_chunk]
    credited = [
        tonal_conformity(notes, kwargs["chord_idx"], credit_resolved_tension=True)
        for _sp, kwargs, notes in last_chunk
    ]
    assert any(c > w for c, w in zip(credited, without))


def test_modal_chart_passes_modal_strength_into_real_generation():
    """Phase 27: song.modal reaches PhraseGenerator.generate()'s real
    modal_strength kwarg -- MODAL_STRENGTH_WHEN_ACTIVE when True."""
    from ensemble.sax import MODAL_STRENGTH_WHEN_ACTIVE

    original = wolfson_phrase_generator.PhraseGenerator.generate
    modal_strengths = []

    def recording_generate(self, seed_phrase, **kwargs):
        modal_strengths.append(kwargs.get("modal_strength"))
        return original(self, seed_phrase, **kwargs)

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        song = build_slow_song()
        song.modal = True
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
                    generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=7))
        Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert modal_strengths
    assert all(ms == MODAL_STRENGTH_WHEN_ACTIVE for ms in modal_strengths)


def test_non_modal_chart_passes_zero_modal_strength():
    """Phase 27: today's unchanged default -- a chart without modal: true
    passes modal_strength=0.0, matching Wolfson's own generate() default."""
    original = wolfson_phrase_generator.PhraseGenerator.generate
    modal_strengths = []

    def recording_generate(self, seed_phrase, **kwargs):
        modal_strengths.append(kwargs.get("modal_strength"))
        return original(self, seed_phrase, **kwargs)

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        song = build_slow_song()  # modal=False, the default
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
                    generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=7))
        Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert modal_strengths
    assert all(ms == 0.0 for ms in modal_strengths)


def test_corpus_none_reproduces_selection_without_it():
    """Phase 29: corpus=None (the default) must be provably a no-op --
    recompute the pre-Phase-29 3-term key (-dissonance, adherence, overall)
    directly from real candidates and confirm it still picks the same winner
    sax_generator actually dispensed, the same backward-compatibility proof
    every earlier selection-key extension in this file has made. Needs no
    corpus cache -- corpus=None never touches CorpusMotifs at all."""
    from ensemble.critic import dissonance, motif_adherence, musicality_score
    from ensemble.sax import chord_to_wolfson_index

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=9, corpus=None)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    f7_idx = chord_to_wolfson_index(Chord.parse("F7"))
    first_chunk = candidates[:8]
    assert len(first_chunk) == 8
    assert all(kwargs["chord_idx"] == f7_idx for _sp, kwargs, _notes in first_chunk)

    recomputed = [
        (-dissonance(notes, f7_idx), motif_adherence(notes, []), musicality_score(notes, f7_idx, seed_phrase, SAX_REGISTER).overall)
        for seed_phrase, kwargs, notes in first_chunk
    ]
    best_key = max(recomputed)
    winner_notes = first_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)

    first_chunk_span = 4 * BEATS_PER_BAR  # DEFAULT_PLAN_BARS=4
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and 0.0 <= e.start_beat < first_chunk_span
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))


@needs_corpus_cache
def test_corpus_score_is_a_noop_when_chunk_is_not_bias_distorted():
    """Phase 29: corpus IS provided here, but build_slow_song()'s single held
    F7 chord with no RehearsalMemory (motif_targets always []) and modal=False
    (the default) means is_bias_distorted is False for every chunk -- so
    corpus_familiarity should never be computed/contribute, and selection
    should match the plain 3-term key exactly, proving the "only bias-
    distorted chunks" gate actually holds on real data, not just that it
    reads correctly in ensemble/sax.py's source."""
    from ensemble.critic import dissonance, motif_adherence, musicality_score
    from ensemble.sax import chord_to_wolfson_index

    corpus = CorpusMotifs(wjd_corpus.CACHE_PATH)
    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=9, corpus=corpus)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    f7_idx = chord_to_wolfson_index(Chord.parse("F7"))
    first_chunk = candidates[:8]
    assert len(first_chunk) == 8

    recomputed = [
        (-dissonance(notes, f7_idx), motif_adherence(notes, []), musicality_score(notes, f7_idx, seed_phrase, SAX_REGISTER).overall)
        for seed_phrase, kwargs, notes in first_chunk
    ]
    best_key = max(recomputed)
    winner_notes = first_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)

    first_chunk_span = 4 * BEATS_PER_BAR
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and 0.0 <= e.start_beat < first_chunk_span
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))


@needs_corpus_cache
def test_corpus_familiarity_reaches_real_selection_on_a_recalled_motif_chunk():
    """Phase 29: build_slow_song()'s SECOND chunk (DEFAULT_PLAN_BARS=4 over an
    8-bar hold) gets a real, non-empty motif_targets from RehearsalMemory --
    the same bias-distortion source as test_search_with_a_motif_target_
    prefers_higher_adherence_over_higher_overall above -- so is_bias_distorted
    is True and corpus_familiarity should actually enter the selection key.
    Recomputes the FULL 4-term key (-dissonance, adherence, corpus_familiarity,
    overall) using the real CorpusMotifs built from wjazzd.db, confirming the
    dispensed candidate matches -- not just that corpus_familiarity computes a
    sensible number in isolation (already covered in tests/test_critic.py),
    but that it actually reaches real selection here."""
    from ensemble.critic import corpus_familiarity, dissonance, motif_adherence, musicality_score
    from ensemble.wolfson.chords import QUAL_DOM

    corpus = CorpusMotifs(wjd_corpus.CACHE_PATH)
    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

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
            n_candidates=2, motif_recall_candidates=8, corpus=corpus,
        )
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(candidates) == 10  # chunk 1: 2 candidates, chunk 2 (recall): 8
    last_chunk = candidates[-8:]
    motif_targets = last_chunk[0][1]["motif_targets"]
    assert motif_targets != []  # the bias-distortion this test is actually about

    recomputed = [
        (
            -dissonance(notes, kwargs["chord_idx"]),
            motif_adherence(notes, motif_targets),
            corpus_familiarity(notes, QUAL_DOM, corpus),
            musicality_score(notes, kwargs["chord_idx"], seed_phrase, SAX_REGISTER).overall,
        )
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


@needs_corpus_cache
def test_corpus_familiarity_reaches_real_selection_on_a_modal_chart():
    """Phase 29: the second half of is_bias_distorted's OR — song.modal=True
    (Phase 27's modal_strength bias) also makes corpus_familiarity enter the
    key, with no RehearsalMemory involved at all this time (motif_targets is
    always [] here, isolating the modal half of the condition specifically)."""
    from ensemble.critic import corpus_familiarity, dissonance, motif_adherence, musicality_score
    from ensemble.sax import chord_to_wolfson_index
    from ensemble.wolfson.chords import QUAL_DOM

    corpus = CorpusMotifs(wjd_corpus.CACHE_PATH)
    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        song = build_slow_song()
        song.modal = True
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=8, seed=13, corpus=corpus)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=song, voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    f7_idx = chord_to_wolfson_index(Chord.parse("F7"))
    first_chunk = candidates[:8]
    assert len(first_chunk) == 8

    recomputed = [
        (
            -dissonance(notes, f7_idx),
            motif_adherence(notes, []),
            corpus_familiarity(notes, QUAL_DOM, corpus),
            musicality_score(notes, f7_idx, seed_phrase, SAX_REGISTER, modal=True).overall,
        )
        for seed_phrase, kwargs, notes in first_chunk
    ]
    best_key = max(recomputed)
    winner_notes = first_chunk[recomputed.index(best_key)][2]
    winner_pitches = sorted(n["pitch"] for n in winner_notes if n["pitch"] != REST_PITCH)

    first_chunk_span = 4 * BEATS_PER_BAR
    dispensed_pitches = sorted(
        e.pitch for e in timeline
        if e.voice_id == "sax" and 0.0 <= e.start_beat < first_chunk_span
    )
    assert set(dispensed_pitches).issubset(set(winner_pitches))


def test_winning_score_log_matches_the_real_selected_winner():
    """Phase 30: winning_score_log holds the actual WINNING candidate's full
    MusicalityScore per chunk-build -- one entry per chunk (build_slow_song()
    always produces exactly 2). Verified via the same spy-and-recompute
    discipline as every other real-consumer test in this file, NOT by
    comparing against max(last_candidate_scores) -- selection is a
    lexicographic (-dissonance, adherence, corpus_score, overall) key, so
    the winner isn't necessarily the candidate with the highest .overall
    (the whole point of Phase 18's dissonance gate)."""
    from ensemble.critic import musicality_score
    from ensemble.sax import chord_to_wolfson_index

    original = wolfson_phrase_generator.PhraseGenerator.generate
    candidates = []

    def recording_generate(self, seed_phrase, **kwargs):
        notes = original(self, seed_phrase, **kwargs)
        candidates.append((seed_phrase, kwargs, notes))
        return notes

    wolfson_phrase_generator.PhraseGenerator.generate = recording_generate
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=5, seed=7)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        wolfson_phrase_generator.PhraseGenerator.generate = original

    assert len(sax_gen.winning_score_log) == 2  # build_slow_song() always produces exactly 2 chunks
    assert len(candidates) == 10  # 5 candidates x 2 chunks

    f7_idx = chord_to_wolfson_index(Chord.parse("F7"))
    first_chunk = candidates[:5]
    recomputed_scores = [
        musicality_score(notes, f7_idx, seed_phrase, SAX_REGISTER) for seed_phrase, kwargs, notes in first_chunk
    ]
    # winning_score_log[0] must be EXACTLY one of the 5 real candidates'
    # independently-recomputed scores (frozen-dataclass equality) -- proof
    # it holds a real winner from this batch, not a stale or fabricated value.
    assert sax_gen.winning_score_log[0] in recomputed_scores
    # ...and specifically the one with the lowest dissonance (Phase 18's own
    # gate, checked first in the selection key) -- dissonance_log[0] is that
    # winner's own real dissonance, recomputed here as a cross-check.
    from ensemble.critic import dissonance
    recomputed_dissonances = [dissonance(notes, f7_idx) for _sp, _kw, notes in first_chunk]
    assert sax_gen.dissonance_log[0] == min(recomputed_dissonances)


def test_prior_range_reaches_real_selection_on_the_second_chunk():
    """Phase 32: build_slow_song() always produces exactly 2 chunks
    (DEFAULT_PLAN_BARS=4 over an 8-bar hold) -- chunk 1 has no prior range
    yet (own_pitch_range starts empty), chunk 2 should be scored with
    prior_range equal to chunk 1's own real dispensed pitch bounds. Spies on
    ensemble.sax's own bound name for musicality_score (NOT
    ensemble.critic.musicality_score -- sax.py did `from .critic import
    musicality_score`, its own local reference), the same technique
    rehearsal_ab_test.py already uses, delegating to the real implementation
    throughout."""
    import ensemble.sax as sax_module

    original_score = sax_module.musicality_score
    calls = []  # (register, prior_range) per musicality_score call

    def score_wrapper(notes, chord_idx, seed_phrase, register, **kwargs):
        calls.append((register, kwargs.get("prior_range")))
        return original_score(notes, chord_idx, seed_phrase, register, **kwargs)

    sax_module.musicality_score = score_wrapper
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=3, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        sax_module.musicality_score = original_score

    assert len(calls) == 6  # 3 candidates x 2 chunks

    first_chunk_calls = calls[:3]
    second_chunk_calls = calls[3:]
    assert all(prior_range is None for _reg, prior_range in first_chunk_calls)

    second_chunk_start = 4 * BEATS_PER_BAR
    chunk1_pitches = [
        e.pitch for e in timeline
        if e.voice_id == "sax" and 0.0 <= e.start_beat < second_chunk_start
    ]
    assert chunk1_pitches  # a real, non-empty dispensed chunk
    expected_prior_range = (min(chunk1_pitches), max(chunk1_pitches))
    assert all(prior_range == expected_prior_range for _reg, prior_range in second_chunk_calls)

    all_sax_pitches = [e.pitch for e in timeline if e.voice_id == "sax"]
    assert sax_gen.own_pitch_range == {"low": min(all_sax_pitches), "high": max(all_sax_pitches)}


def test_prior_mean_beats_reaches_real_selection_on_the_second_chunk():
    """Phase 36: the register_balance counterpart to
    test_prior_range_reaches_real_selection_on_the_second_chunk above --
    build_slow_song() always produces exactly 2 chunks; chunk 1 has no prior
    mean yet (own_pitch_weighted starts empty), chunk 2 should be scored with
    prior_mean_beats equal to chunk 1's own real dispensed, duration-weighted
    pitch mean. Same spy-on-ensemble.sax's-own-bound-name technique."""
    import ensemble.sax as sax_module

    original_score = sax_module.musicality_score
    calls = []  # (register, prior_mean_beats) per musicality_score call

    def score_wrapper(notes, chord_idx, seed_phrase, register, **kwargs):
        calls.append((register, kwargs.get("prior_mean_beats")))
        return original_score(notes, chord_idx, seed_phrase, register, **kwargs)

    sax_module.musicality_score = score_wrapper
    try:
        bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=3, seed=5)
        sax = Voice(id="sax", instrument="sax", register=SAX_REGISTER, source="ai", generator=sax_gen)
        timeline = Session(song=build_slow_song(), voices=[bass, sax]).generate()
    finally:
        sax_module.musicality_score = original_score

    assert len(calls) == 6  # 3 candidates x 2 chunks

    first_chunk_calls = calls[:3]
    second_chunk_calls = calls[3:]
    assert all(prior_mean_beats is None for _reg, prior_mean_beats in first_chunk_calls)

    second_chunk_start = 4 * BEATS_PER_BAR
    chunk1_events = [
        e for e in timeline
        if e.voice_id == "sax" and 0.0 <= e.start_beat < second_chunk_start
    ]
    assert chunk1_events  # a real, non-empty dispensed chunk
    expected_sum = sum(e.pitch * e.duration_beats for e in chunk1_events)
    expected_beats = sum(e.duration_beats for e in chunk1_events)
    for _reg, prior_mean_beats in second_chunk_calls:
        assert prior_mean_beats[0] == pytest.approx(expected_sum)
        assert prior_mean_beats[1] == pytest.approx(expected_beats)

    # Phase 44: own_pitch_weighted's FINAL state (after chunk 2's own update)
    # now decays chunk 1's raw contribution by the real elapsed beats between
    # the two updates (second_chunk_start - 0.0 = 16 beats = exactly one
    # REGISTER_BALANCE_HALF_LIFE_BEATS) before adding chunk 2's own
    # contribution -- no longer the plain undecayed sum of every dispensed
    # event. Recomputed independently via _decay_pitch_weighted, not assumed.
    chunk2_events = [e for e in timeline if e.voice_id == "sax" and e.start_beat >= second_chunk_start]
    decayed_sum, decayed_beats = _decay_pitch_weighted(
        expected_sum, expected_beats, second_chunk_start - 0.0, REGISTER_BALANCE_HALF_LIFE_BEATS,
    )
    expected_final_sum = decayed_sum + sum(e.pitch * e.duration_beats for e in chunk2_events)
    expected_final_beats = decayed_beats + sum(e.duration_beats for e in chunk2_events)
    assert sax_gen.own_pitch_weighted["sum"] == pytest.approx(expected_final_sum)
    assert sax_gen.own_pitch_weighted["beats"] == pytest.approx(expected_final_beats)


def test_own_pitch_weighted_decays_more_over_a_larger_gap():
    """Phase 44: not every chunk's own real content lands in-register (a
    real, pre-existing, seed-dependent property this test doesn't want to
    depend on), so the gap is created deterministically by backdating
    own_pitch_weighted_last_beat directly -- the same "manipulate exposed
    closure state directly for testing" convention already used for
    laying_out["active"] (Phase 43). Spies on ensemble.sax's own
    _decay_pitch_weighted to confirm elapsed_beats genuinely reflects
    whatever real gap is set up, rather than being a fixed per-update
    factor -- a much larger, artificially backdated gap here than the
    single-half-life gap test_prior_mean_beats_reaches_real_selection_on_
    the_second_chunk's own fix exercises."""
    from dataclasses import replace

    from ensemble.timeline import Timeline

    import ensemble.sax as sax_module

    original_decay = sax_module._decay_pitch_weighted
    elapsed_calls = []

    def decay_wrapper(pitch_sum, pitch_beats, elapsed_beats, half_life_beats):
        elapsed_calls.append(elapsed_beats)
        return original_decay(pitch_sum, pitch_beats, elapsed_beats, half_life_beats)

    sax_module._decay_pitch_weighted = decay_wrapper
    try:
        song = build_slow_song()
        sax_gen = sax_generator(SAX_REGISTER, target_voice_id="bass", n_candidates=1, seed=5, plan_bars=1)
        timeline = _drive_bars(sax_gen, song, n_bars=1)  # bar 0: a real first update, elapsed_beats == 0
        assert sax_gen.own_pitch_weighted_last_beat["value"] == 0.0  # confirms bar 0 really updated it

        # bar 1's own chunk (this seed, this song) happens not to land any
        # real in-register content -- a real, pre-existing, seed-dependent
        # property this test doesn't want to depend on -- so the gap is
        # created deterministically by backdating own_pitch_weighted_last_beat
        # directly instead, whichever bar ends up updating it next.
        backdated_last_beat = -1000.0
        sax_gen.own_pitch_weighted_last_beat["value"] = backdated_last_beat
        director_signal = DirectorSignal()
        for bar_index in (1, 2):
            prior = Timeline(list(timeline.events))
            for event in sax_gen(song, bar_index, prior, director_signal):
                timeline.add(replace(event, voice_id="sax"))
    finally:
        sax_module._decay_pitch_weighted = original_decay

    assert elapsed_calls[0] == 0.0  # bar 0's own update
    assert len(elapsed_calls) >= 2  # a real second update happened
    resume_bar_start = 2 * BEATS_PER_BAR  # bar 2 is where this seed's second real update lands
    real_gap = resume_bar_start - backdated_last_beat
    assert elapsed_calls[-1] == pytest.approx(real_gap)
    assert elapsed_calls[-1] > 10 * REGISTER_BALANCE_HALF_LIFE_BEATS  # far more than one half-life
