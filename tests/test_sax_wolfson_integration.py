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
from ensemble.director import Director, constant_director_source
from ensemble.generators import chord_tone_generator
from ensemble.sax import sax_generator
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
def counting_phrase_generator_calls():
    """Spy on PhraseGenerator.generate's call count — wraps the real method,
    delegates to it, counts invocations, restores the original after. Not a
    mock: the real model still runs every call, matching this codebase's
    no-mocking-framework norm and its own "verify via a spy, not independent
    re-derivation" lesson (see the Phase 8 postmortem)."""
    original = wolfson_phrase_generator.PhraseGenerator.generate
    counter = {"calls": 0}

    def counting_generate(self, *args, **kwargs):
        counter["calls"] += 1
        return original(self, *args, **kwargs)

    wolfson_phrase_generator.PhraseGenerator.generate = counting_generate
    try:
        yield counter
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
    with counting_phrase_generator_calls() as counter:
        Session(song=song, voices=[bass, sax]).generate()
    assert counter["calls"] < n_bars


def test_plan_buffer_makes_far_fewer_calls_on_a_slow_harmonic_rhythm_chart():
    """A directly-constructed Song with one chord held for 8 bars (32 beats) —
    same Song-construction pattern as tests/test_transitions.py's own edge-case
    tests. DEFAULT_PLAN_BARS=4 caps each chunk, so 8 bars should collapse into
    at most 2 generate() calls — a clear demonstration of the mechanism's real
    effect on a chart with a realistic-for-many-tunes harmonic rhythm, unlike
    blues's fast changes above."""
    song = Song(
        title="slow changes",
        changes=Changes([ChangesEvent(Chord.parse("F7"), 32.0)]),
        form=[Section("A", 1)],
        tempo_bpm=120,
    )
    n_bars = 8
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax = Voice(
        id="sax", instrument="sax", register=SAX_REGISTER, source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=7),
    )
    with counting_phrase_generator_calls() as counter:
        Session(song=song, voices=[bass, sax]).generate()
    assert counter["calls"] == 2  # ceil(8 bars / plan_bars=4) -- exact, not just "fewer"
    assert counter["calls"] < n_bars


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
