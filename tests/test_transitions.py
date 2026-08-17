"""Tests for ensemble/transitions.py — no MIDI/audio needed."""

import threading
import time
from pathlib import Path

from ensemble.drums import CLOSED_HI_HAT, drum_generator
from ensemble.session import Session
from ensemble.timeline import BEATS_PER_BAR
from ensemble.transitions import LiveGestureQueue, TransitionController, scripted_gesture_source
from ensemble.voice import Voice
from gesture.vocabulary import Gesture
from song import Section, Song, parse_chart

CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


def test_effective_song_with_no_gestures_is_unchanged():
    song = load_blues()
    tc = TransitionController()
    assert tc.effective_song(song) == song


def test_handover_truncates_the_current_section():
    song = load_blues()  # Head x1, Solos x3, Out x1
    tc = TransitionController()
    # bar 24 = start of Solos chorus_index=1 (the 2nd of 3 choruses)
    tc.on_gesture(Gesture("handover"), song, 24 * BEATS_PER_BAR)
    effective = tc.effective_song(song)
    assert effective.form == [
        Section(name="Head", repeats=1),
        Section(name="Solos", repeats=2),
        Section(name="Out", repeats=1),
    ]
    assert song.form[1].repeats == 3  # original untouched


def test_handover_during_open_ended_section_makes_it_finite():
    changes = load_blues().changes
    song = Song(title="t", changes=changes, form=[Section("Solos", None)], tempo_bpm=120)
    tc = TransitionController()
    tc.on_gesture(Gesture("handover"), song, 2 * changes.total_beats)  # partway through chorus_index=2
    effective = tc.effective_song(song)
    assert effective.form == [Section("Solos", 3)]


def test_second_later_handover_does_not_extend_an_already_truncated_section():
    song = load_blues()
    tc = TransitionController()
    tc.on_gesture(Gesture("handover"), song, 24 * BEATS_PER_BAR)  # truncates Solos to 2
    tc.on_gesture(Gesture("handover"), song, 36 * BEATS_PER_BAR)  # later in the (already effectively past) section
    assert tc.overrides[1] == 2  # unchanged, not extended back up


def test_reset_tempo_gesture_is_ignored():
    song = load_blues()
    tc = TransitionController()
    tc.on_gesture(Gesture("reset_tempo"), song, 24 * BEATS_PER_BAR)
    assert tc.overrides == {}
    assert tc.effective_song(song) == song


def test_scripted_gesture_source_returns_only_scheduled_bars():
    g = Gesture("handover")
    source = scripted_gesture_source({5: [g]})
    assert source(5) == [g]
    assert source(6) == []


def test_live_gesture_queue_across_threads():
    queue = LiveGestureQueue()
    g1, g2 = Gesture("handover"), Gesture("reset_tempo")

    def producer():
        queue.append(g1)
        time.sleep(0.01)
        queue.append(g2)

    thread = threading.Thread(target=producer)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()

    drained = queue.drain(0)
    assert drained == [g1, g2]
    assert queue.drain(0) == []


def test_handover_shifts_a_real_consumer_drums_density():
    """Integration test: a scripted handover partway through Solos should make
    drums' section-aware density (§7, already built) shift to sparse earlier than
    the nominal bar count would place it -- proof the effective form actually
    reaches a real generator, not just that TransitionController computes the
    right dict in isolation."""
    song = load_blues()

    def make_session(gesture_source=None):
        drums = Voice(id="drums", instrument="drums", register=(35, 59), source="ai", generator=drum_generator(seed=1))
        return Session(song=song, voices=[drums], gesture_source=gesture_source)

    without_gesture = make_session().generate()
    bar36_without = {e.pitch for e in without_gesture if 36 * BEATS_PER_BAR <= e.start_beat < 37 * BEATS_PER_BAR}
    assert bar36_without != {CLOSED_HI_HAT}  # nominally busy (last Solos chorus), not sparse

    gesture_source = scripted_gesture_source({24: [Gesture("handover")]})
    with_gesture = make_session(gesture_source=gesture_source).generate()
    bar36_with = {e.pitch for e in with_gesture if 36 * BEATS_PER_BAR <= e.start_beat < 37 * BEATS_PER_BAR}
    assert bar36_with == {CLOSED_HI_HAT}  # shifted to sparse (Out-like) 12 bars early
