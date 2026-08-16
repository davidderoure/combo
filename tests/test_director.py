"""Tests for ensemble/director.py — no MIDI/audio needed."""

from pathlib import Path

from gesture.vocabulary import Gesture
from ensemble.director import (
    REFERENCE_MAX_DENSITY,
    Director,
    DirectorSignal,
    aggregate_director_signals,
    constant_director_source,
    ensemble_intensity_critic,
)
from ensemble.session import Session
from ensemble.timeline import NoteEvent, Timeline
from ensemble.voice import Voice
from song import parse_chart

CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


def test_aggregate_of_no_signals_is_neutral():
    assert aggregate_director_signals([]) == DirectorSignal()


def test_aggregate_averages_intensity():
    result = aggregate_director_signals([DirectorSignal(intensity=0.2), DirectorSignal(intensity=0.8)])
    assert result.intensity == 0.5


def test_aggregate_carries_through_a_lone_gesture():
    g = Gesture("reset_tempo")
    result = aggregate_director_signals([DirectorSignal(intensity=0.5), DirectorSignal(intensity=0.5, gesture=g)])
    assert result.gesture == g


def test_aggregate_with_no_gestures_has_none():
    result = aggregate_director_signals([DirectorSignal(intensity=0.3), DirectorSignal(intensity=0.7)])
    assert result.gesture is None


def test_constant_director_source_ignores_its_inputs():
    source = constant_director_source(0.9)
    song = load_blues()
    assert source(song, 0, Timeline()).intensity == 0.9
    assert source(song, 47, Timeline([NoteEvent("x", 60, 80, 0.0, 1.0)])).intensity == 0.9


def test_ensemble_intensity_critic_reads_combined_density():
    critic = ensemble_intensity_critic(voice_ids=["sax", "drums"], lookback_bars=2)
    song = load_blues()
    since = 0.0  # bar_index=2, lookback_bars=2 -> window [0, 8)
    # sax: 4 notes, drums: 4 notes, over 8 beats -> combined density 1.0 notes/beat
    tl = Timeline(
        [NoteEvent("sax", 60, 80, since + i * 2.0, 0.1) for i in range(4)]
        + [NoteEvent("drums", 42, 80, since + i * 2.0, 0.1) for i in range(4)]
    )
    signal = critic(song, 2, tl)
    assert signal.intensity == 1.0 / REFERENCE_MAX_DENSITY


def test_ensemble_intensity_critic_clamps_to_one():
    critic = ensemble_intensity_critic(voice_ids=["sax"], lookback_bars=2)
    song = load_blues()
    tl = Timeline([NoteEvent("sax", 60, 80, i * 0.1, 0.05) for i in range(200)])  # absurdly dense
    assert critic(song, 2, tl).intensity == 1.0


def test_director_validation_mirrors_voice():
    try:
        Director(id="bad", source="robot")
        assert False, "expected a ValueError for an invalid source"
    except ValueError:
        pass

    try:
        Director(id="ai-no-source", source="ai")
        assert False, "expected a ValueError for a missing signal_source"
    except ValueError:
        pass

    Director(id="human", source="human")  # valid without a signal_source, like Voice


def test_session_passes_the_aggregated_signal_to_generators():
    """Proves Session actually computes and wires the aggregated signal through, by
    having a generator record exactly what it was handed — deliberately not testing
    this via comping's note *output*, which would need fragile beat-boundary counting
    across a multi-bar run (comping.py's own tests already cover its intensity
    sensitivity precisely, via direct calls with a controlled Timeline)."""
    song = load_blues()
    received_signals = []

    def spy_generator(song, bar_index, timeline, director_signal):
        received_signals.append(director_signal)
        return []

    spy = Voice(id="spy", instrument="test", register=(0, 127), source="ai", generator=spy_generator)

    Session(song=song, voices=[spy]).generate()
    assert all(s == DirectorSignal() for s in received_signals)  # no directors -> neutral every bar

    received_signals.clear()
    high_intensity_director = Director(id="d", source="ai", signal_source=constant_director_source(0.9))
    Session(song=song, voices=[spy], directors=[high_intensity_director]).generate()
    assert all(s.intensity == 0.9 for s in received_signals)
