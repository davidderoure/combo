"""Integration tests for ensemble/sax.py against the real Wolfson-adapted model —
DESIGN.md §12, Phase 8. First test file in this codebase needing a binary artifact
absent from a fresh clone: skips cleanly if ensemble/wolfson/models/sax_best.pt
hasn't been copied in (gitignored — see README). Runs real inference throughout,
matching this codebase's no-mocking norm — measured ~11ms/call on CPU, fast enough
that faking it would be needless."""

from pathlib import Path

import pytest

from ensemble.generators import chord_tone_generator
from ensemble.sax import sax_generator
from ensemble.session import Session
from ensemble.timeline import BEATS_PER_BAR
from ensemble.voice import Voice
from ensemble.wolfson.phrase_generator import REST_PITCH
from song import parse_chart

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


def make_session(seed: int) -> Session:
    bass = Voice(id="bass", instrument="bass", register=BASS_REGISTER, source="ai", generator=chord_tone_generator(BASS_REGISTER))
    sax = Voice(
        id="sax",
        instrument="sax",
        register=SAX_REGISTER,
        source="ai",
        generator=sax_generator(SAX_REGISTER, target_voice_id="bass", seed=seed),
    )
    return Session(song=load_blues(), voices=[bass, sax])


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
