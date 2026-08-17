"""Tests for output/midi_output.py — no MIDI hardware needed.

build_schedule is pure. play_timeline is tested against a fake midi_out (a
plain class recording send_message calls, the same fake-object-not-mocking
technique test_sax_wolfson_integration.py's spies already use) and a
FakeClock (ensemble/session.py), so nothing here sleeps for real or touches
real hardware. Real port-opening/hardware I/O is explicitly not covered here,
same honest precedent as tests/test_midi_sources.py.
"""

from ensemble.session import FakeClock
from ensemble.timeline import NoteEvent, Timeline
from output.midi_output import NOTE_OFF, NOTE_ON, build_schedule, play_timeline


def test_single_event_produces_note_on_and_note_off_at_correct_times():
    tl = Timeline([NoteEvent("sax", 60, 90, start_beat=2.0, duration_beats=1.0)])
    schedule = build_schedule(tl, tempo_bpm=120.0, channels={"sax": 1})  # 0.5 sec/beat
    assert schedule == [
        (1.0, bytes([NOTE_ON | 0, 60, 90])),
        (1.5, bytes([NOTE_OFF | 0, 60, 0])),
    ]


def test_channel_is_1_indexed_input_0_indexed_nibble():
    tl = Timeline([NoteEvent("keys", 60, 80, start_beat=0.0, duration_beats=1.0)])
    schedule = build_schedule(tl, tempo_bpm=60.0, channels={"keys": 10})
    on_status = schedule[0][1][0]
    assert on_status == (NOTE_ON | 9)  # channel 10 -> nibble 9


def test_tempo_changes_the_seconds_per_beat_conversion():
    tl = Timeline([NoteEvent("sax", 60, 80, start_beat=1.0, duration_beats=1.0)])
    fast = build_schedule(tl, tempo_bpm=120.0, channels={"sax": 1})
    slow = build_schedule(tl, tempo_bpm=60.0, channels={"sax": 1})
    assert fast[0][0] == 0.5
    assert slow[0][0] == 1.0


def test_unrouted_voice_id_is_skipped_not_defaulted():
    tl = Timeline([NoteEvent("mystery", 60, 80, start_beat=0.0, duration_beats=1.0)])
    schedule = build_schedule(tl, tempo_bpm=120.0, channels={"sax": 1})
    assert schedule == []


def test_velocity_is_clamped_to_1_127():
    tl = Timeline(
        [
            NoteEvent("sax", 60, 0, start_beat=0.0, duration_beats=1.0),
            NoteEvent("sax", 61, 200, start_beat=0.0, duration_beats=1.0),
        ]
    )
    schedule = build_schedule(tl, tempo_bpm=120.0, channels={"sax": 1})
    velocities = sorted(msg[2] for _t, msg in schedule if msg[0] & 0xF0 == NOTE_ON)
    assert velocities == [1, 127]


def test_multiple_voices_interleave_sorted_by_time():
    tl = Timeline(
        [
            NoteEvent("sax", 60, 80, start_beat=1.0, duration_beats=0.5),
            NoteEvent("keys", 48, 70, start_beat=0.0, duration_beats=2.0),
        ]
    )
    schedule = build_schedule(tl, tempo_bpm=60.0, channels={"sax": 1, "keys": 2})
    times = [t for t, _msg in schedule]
    assert times == sorted(times)
    assert times[0] == 0.0  # keys note-on first


class FakeMidiOut:
    """fail_after: raise KeyboardInterrupt exactly once, on the call after this
    many messages have been sent — models a single Ctrl-C, not every send
    failing forever, so the cleanup calls that follow in play_timeline's
    finally block still succeed and get recorded."""

    def __init__(self, fail_after=None):
        self.sent = []
        self.fail_after = fail_after
        self._raised = False

    def send_message(self, message):
        if self.fail_after is not None and not self._raised and len(self.sent) >= self.fail_after:
            self._raised = True
            raise KeyboardInterrupt
        self.sent.append(tuple(message))


def test_play_timeline_sends_messages_in_schedule_order_with_fake_clock():
    tl = Timeline(
        [
            NoteEvent("sax", 60, 80, start_beat=0.0, duration_beats=1.0),
            NoteEvent("sax", 62, 80, start_beat=1.0, duration_beats=1.0),
        ]
    )
    fake_out = FakeMidiOut()
    clock = FakeClock()
    play_timeline(tl, tempo_bpm=120.0, channels={"sax": 1}, port_index=0, midi_out=fake_out, clock=clock)

    note_ons = [m for m in fake_out.sent if m[0] & 0xF0 == NOTE_ON]
    assert [m[1] for m in note_ons] == [60, 62]
    assert clock.total_slept > 0  # real pacing was requested, just not actually waited on


def test_play_timeline_sends_all_notes_off_after_normal_completion():
    tl = Timeline([NoteEvent("sax", 60, 80, start_beat=0.0, duration_beats=1.0)])
    fake_out = FakeMidiOut()
    play_timeline(tl, tempo_bpm=120.0, channels={"sax": 1, "drums": 10}, port_index=0, midi_out=fake_out, clock=FakeClock())

    cc_messages = [m for m in fake_out.sent if m[0] & 0xF0 == 0xB0]
    channels_silenced = {m[0] & 0x0F for m in cc_messages}
    assert channels_silenced == {0, 9}  # every configured channel, not just the one used


def test_play_timeline_sends_all_notes_off_even_when_interrupted():
    tl = Timeline(
        [
            NoteEvent("sax", 60, 80, start_beat=0.0, duration_beats=1.0),
            NoteEvent("sax", 62, 80, start_beat=1.0, duration_beats=1.0),
        ]
    )
    fake_out = FakeMidiOut(fail_after=1)  # raises KeyboardInterrupt after the first send
    try:
        play_timeline(tl, tempo_bpm=120.0, channels={"sax": 1}, port_index=0, midi_out=fake_out, clock=FakeClock())
        assert False, "expected KeyboardInterrupt to propagate"
    except KeyboardInterrupt:
        pass

    cc_messages = [m for m in fake_out.sent if m[0] & 0xF0 == 0xB0]
    assert len(cc_messages) == 2  # All-Notes-Off + All-Sound-Off still sent on channel 1


def test_play_timeline_sends_explicit_note_off_for_every_pitch_during_cleanup():
    """Some synths (Logic's software instruments, per Wolfson's own
    output/midi_output.py -- confirmed again by a real stuck note during
    testing) ignore CC 123/120 entirely and need an explicit note_off per
    pitch to guarantee nothing is left sounding."""
    tl = Timeline([NoteEvent("sax", 60, 80, start_beat=0.0, duration_beats=1.0)])
    fake_out = FakeMidiOut()
    play_timeline(tl, tempo_bpm=120.0, channels={"sax": 1}, port_index=0, midi_out=fake_out, clock=FakeClock())

    note_offs = [m for m in fake_out.sent if m[0] & 0xF0 == NOTE_OFF and m[0] & 0x0F == 0]
    pitches_turned_off = {m[1] for m in note_offs}
    assert pitches_turned_off == set(range(128))
