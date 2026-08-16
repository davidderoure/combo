"""Tests for ensemble/ — no real waiting, no MIDI hardware needed."""

import time
from pathlib import Path

from ensemble import FakeClock, MACHINE_SPEED, REAL_TIME, Session, Voice, chord_tone_generator
from song import Section, Song, parse_chart

CHARTS_DIR = Path(__file__).resolve().parent.parent / "songs"
SAX_REGISTER = (55, 79)  # G3-G5-ish, arbitrary for the stub


def load_blues():
    return parse_chart((CHARTS_DIR / "blues_in_f.chart").read_text())


def make_session(song):
    sax = Voice(
        id="sax",
        instrument="sax",
        register=SAX_REGISTER,
        source="ai",
        generator=chord_tone_generator(SAX_REGISTER),
    )
    return Session(song=song, voices=[sax])


def test_machine_speed_returns_complete_timeline_without_delay():
    song = load_blues()
    session = make_session(song)

    started = time.time()
    timeline = session.generate(mode=MACHINE_SPEED)
    elapsed = time.time() - started

    assert len(timeline) == 240  # 5 choruses * 12 bars * 4 events/bar
    assert elapsed < 0.5  # nowhere near the ~109s the song would take in real time


def test_real_time_paces_according_to_tempo_without_actually_waiting():
    song = load_blues()
    session = make_session(song)
    clock = FakeClock()

    started = time.time()
    timeline = session.generate(mode=REAL_TIME, clock=clock)
    elapsed = time.time() - started

    expected_seconds = song.total_beats * 60.0 / song.tempo_bpm
    assert abs(clock.total_slept - expected_seconds) < 1e-6
    assert elapsed < 0.5  # FakeClock.sleep() doesn't actually sleep
    assert len(timeline) == 240


def test_generated_notes_are_chord_tones_of_the_actual_changes():
    song = load_blues()
    session = make_session(song)
    timeline = session.generate(mode=MACHINE_SPEED)

    for event in timeline:
        chord = song.chord_at(event.start_beat)
        pitch_class = event.pitch % 12
        allowed = {chord.root, (chord.root + 7) % 12}
        assert pitch_class in allowed, (
            f"note {event.pitch} (pc {pitch_class}) at beat {event.start_beat} "
            f"isn't root or fifth of {chord}"
        )


def test_open_ended_form_rejected_in_machine_speed_mode():
    song = load_blues()
    song = Song(
        title=song.title,
        changes=song.changes,
        form=[Section(name="Solos", repeats=None)],
        tempo_bpm=song.tempo_bpm,
    )
    session = make_session(song)

    try:
        session.generate(mode=MACHINE_SPEED)
        assert False, "expected a ValueError for an open-ended form"
    except ValueError as e:
        assert "open-ended" in str(e)
