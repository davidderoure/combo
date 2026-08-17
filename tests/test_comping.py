"""Tests for ensemble/comping.py — no MIDI/audio needed."""

from pathlib import Path

from ensemble.comping import comping_generator
from ensemble.director import DirectorSignal
from ensemble.listening import synthetic_varying_density_generator
from ensemble.roles import default_accompanist_roles
from ensemble.session import Session
from ensemble.timeline import BEATS_PER_BAR, NoteEvent, Timeline
from ensemble.voice import Voice
from song import parse_chart

CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"
KEYS_REGISTER = (48, 72)
LOOKBACK_BARS = 2


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


def lookback_window(bar_index: int) -> tuple:
    since = max(0, bar_index - LOOKBACK_BARS) * BEATS_PER_BAR
    until = bar_index * BEATS_PER_BAR
    return since, until


def test_busy_target_density_ducks():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    since, _until = lookback_window(5)
    tl = Timeline([NoteEvent("sax", 60, 80, since + i * 0.5, 0.1) for i in range(16)])  # 2.0 notes/beat
    assert gen(song, 5, tl, DirectorSignal()) == []


def test_sparse_target_density_fills_with_two_stabs():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    since, _until = lookback_window(5)
    tl = Timeline([NoteEvent("sax", 60, 80, since, 0.1)])  # 0.125 notes/beat
    events = gen(song, 5, tl, DirectorSignal())
    assert len(events) == 4  # two stabs, root+fifth each
    chord = song.chord_at(5 * BEATS_PER_BAR)
    for e in events:
        assert e.pitch % 12 in {chord.root, (chord.root + 7) % 12}


def test_moderate_target_density_plays_one_stab():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    since, _until = lookback_window(5)
    tl = Timeline([NoteEvent("sax", 60, 80, since + i * 2.0, 0.1) for i in range(4)])  # 0.5 notes/beat
    events = gen(song, 5, tl, DirectorSignal())
    assert len(events) == 2  # one stab, root+fifth


def test_empty_history_fills_by_default():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    events = gen(song, 0, Timeline(), DirectorSignal())  # bar_index < lookback_bars -> empty window
    assert len(events) == 4


def test_low_director_intensity_pushes_a_moderate_density_to_duck():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    since, _until = lookback_window(5)
    # 11 notes / 8 beats = 1.375 notes/beat: moderate at neutral intensity (< 1.5)
    tl = Timeline([NoteEvent("sax", 60, 80, since + i * (8 / 11), 0.1) for i in range(11)])

    assert len(gen(song, 5, tl, DirectorSignal(intensity=0.5))) == 2  # neutral -> moderate
    assert gen(song, 5, tl, DirectorSignal(intensity=0.0)) == []  # low intensity -> duck


def test_high_director_intensity_pushes_a_moderate_density_to_fill():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    since, _until = lookback_window(5)
    # 3 notes / 8 beats = 0.375 notes/beat: moderate at neutral intensity (> 0.25)
    tl = Timeline([NoteEvent("sax", 60, 80, since + i * (8 / 3), 0.1) for i in range(3)])

    assert len(gen(song, 5, tl, DirectorSignal(intensity=0.5))) == 2  # neutral -> moderate
    assert len(gen(song, 5, tl, DirectorSignal(intensity=1.0))) == 4  # high intensity -> fill


def test_voice_order_does_not_affect_output():
    song = load_blues()

    def make_session(reversed_order: bool) -> Session:
        soloist = Voice(
            id="sax",
            instrument="sax",
            register=(55, 79),
            source="ai",
            generator=synthetic_varying_density_generator(seed=1),
        )
        comper = Voice(
            id="keys",
            instrument="keys",
            register=KEYS_REGISTER,
            source="ai",
            generator=comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=2),
        )
        voices = [comper, soloist] if reversed_order else [soloist, comper]
        return Session(song=song, voices=voices)

    forward = make_session(reversed_order=False).generate()
    reversed_ = make_session(reversed_order=True).generate()
    assert forward.events == reversed_.events


def test_lay_out_produces_far_fewer_notes_than_full_over_many_bars():
    song = load_blues()
    full = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1)
    laying_out = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1, lay_out=True)
    since, _until = lookback_window(5)
    tl = Timeline([NoteEvent("sax", 60, 80, since, 0.1)])  # sparse -> full would fill

    full_count = sum(len(full(song, bar, tl, DirectorSignal())) for bar in range(5, 30))
    lay_out_count = sum(len(laying_out(song, bar, tl, DirectorSignal())) for bar in range(5, 30))
    assert lay_out_count < full_count / 4


def test_lay_out_ignores_target_density_entirely():
    song = load_blues()
    gen = comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=1, lay_out=True)
    since, _until = lookback_window(5)
    # Very sparse target density — would normally trigger a two-stab fill.
    tl = Timeline([NoteEvent("sax", 60, 80, since, 0.1)])
    for bar in range(5, 25):
        events = gen(song, bar, tl, DirectorSignal())
        assert len(events) <= 2  # never the two-stab fill pattern


def test_two_accompanists_with_overlapping_registers_split_role_in_session():
    song = load_blues()
    soloist = Voice(
        id="sax",
        instrument="sax",
        register=(55, 79),
        source="ai",
        generator=synthetic_varying_density_generator(seed=1),
    )
    roles = default_accompanist_roles([("keys", KEYS_REGISTER), ("guitar", KEYS_REGISTER)])
    assert roles == {"keys": True, "guitar": False}
    keys = Voice(
        id="keys",
        instrument="keys",
        register=KEYS_REGISTER,
        source="ai",
        generator=comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=2, lay_out=not roles["keys"]),
    )
    guitar = Voice(
        id="guitar",
        instrument="guitar",
        register=KEYS_REGISTER,
        source="ai",
        generator=comping_generator(KEYS_REGISTER, target_voice_id="sax", seed=3, lay_out=not roles["guitar"]),
    )
    result = Session(song=song, voices=[soloist, keys, guitar]).generate()
    keys_count = sum(1 for e in result if e.voice_id == "keys")
    guitar_count = sum(1 for e in result if e.voice_id == "guitar")
    assert guitar_count < keys_count / 2


def test_generator_cannot_corrupt_the_session_timeline():
    song = load_blues()

    def naughty_generator(song, bar_index, timeline, director_signal):
        timeline.add(NoteEvent("intruder", 999, 999, -1.0, 0.1))
        return []

    naughty = Voice(id="naughty", instrument="test", register=(0, 127), source="ai", generator=naughty_generator)
    session = Session(song=song, voices=[naughty])
    result = session.generate()
    assert all(e.voice_id != "intruder" for e in result)
