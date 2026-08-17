"""Plays a combo Timeline through a MIDI output port — the "separate playback
stage" DESIGN.md §4 names as the follow-on to symbolic generation.

combo generates a whole symbolic Timeline up front (machine_speed — see
ensemble/session.py), unlike wolfson's continuous phrase-by-phrase
generate-and-play interleaving (wolfson/output/midi_output.py). That
architectural difference makes this player much simpler than wolfson's: no
"latest wins" pending-queue, no dedicated output thread — the whole schedule
is known before playback starts, so a single loop that sleeps until each
scheduled event's time and sends it is enough. KeyboardInterrupt during
time.sleep() fires immediately, so a try/finally around that loop is enough
for all-notes-off cleanup too — no thread coordination needed for that either.
"""

from typing import Dict, List, Optional, Tuple

import rtmidi

from ensemble.session import Clock, SystemClock
from ensemble.timeline import Timeline

NOTE_ON = 0x90
NOTE_OFF = 0x80
CC = 0xB0
ALL_NOTES_OFF_CC = 123
ALL_SOUND_OFF_CC = 120


def list_output_ports() -> List[str]:
    return rtmidi.MidiOut().get_ports()


def build_schedule(timeline: Timeline, tempo_bpm: float, channels: Dict[str, int]) -> List[Tuple[float, bytes]]:
    """Timeline (beat-based) + tempo + voice_id->channel map -> a time-sorted list
    of (seconds_from_start, midi_message_bytes) note-on/note-off pairs. A
    voice_id with no entry in `channels` is silently skipped — an unrouted
    voice failing loud (an audibly incomplete performance) rather than
    colliding two voices onto a guessed fallback channel."""
    seconds_per_beat = 60.0 / tempo_bpm
    schedule: List[Tuple[float, bytes]] = []
    for event in timeline:
        channel = channels.get(event.voice_id)
        if channel is None:
            continue
        ch = channel - 1
        start = event.start_beat * seconds_per_beat
        end = start + event.duration_beats * seconds_per_beat
        velocity = max(1, min(127, event.velocity))
        schedule.append((start, bytes([NOTE_ON | ch, event.pitch, velocity])))
        schedule.append((end, bytes([NOTE_OFF | ch, event.pitch, 0])))
    schedule.sort(key=lambda item: item[0])
    return schedule


def play_timeline(
    timeline: Timeline,
    tempo_bpm: float,
    channels: Dict[str, int],
    port_index: int,
    midi_out: Optional["rtmidi.MidiOut"] = None,
    clock: Optional[Clock] = None,
) -> None:
    """Blocking. Opens port_index (unless midi_out is already provided — lets
    tests pass a fake object instead of touching real hardware), builds the
    schedule via build_schedule, then sleeps until each event's absolute
    scheduled time (via clock.now(), not summed per-event sleeps, to avoid
    cumulative drift) and sends it. finally: sends All-Notes-Off + All-Sound-
    Off on every channel actually used, whether playback completed normally
    or was interrupted — closes the port only if this function opened it."""
    schedule = build_schedule(timeline, tempo_bpm, channels)
    if clock is None:
        clock = SystemClock()

    owns_port = midi_out is None
    if midi_out is None:
        midi_out = rtmidi.MidiOut()
        ports = midi_out.get_ports()
        if not ports:
            raise RuntimeError("No MIDI output ports found.")
        if port_index >= len(ports):
            raise RuntimeError(f"MIDI output port {port_index} out of range ({len(ports)} available): {ports}")
        print(f"MIDI output: {ports[port_index]}")
        midi_out.open_port(port_index)

    start_time = clock.now()
    try:
        for scheduled_at, message in schedule:
            wait = start_time + scheduled_at - clock.now()
            if wait > 0:
                clock.sleep(wait)
            midi_out.send_message(list(message))
    finally:
        for channel in set(channels.values()):
            ch = channel - 1
            midi_out.send_message([CC | ch, ALL_NOTES_OFF_CC, 0])
            midi_out.send_message([CC | ch, ALL_SOUND_OFF_CC, 0])
            # Wolfson's own output/midi_output.py found (and documented) that
            # Logic's software instruments ignore both CC messages above and
            # need an explicit note_off per pitch -- the same fix, reused here
            # rather than rediscovered, after a real stuck note during testing.
            for pitch in range(128):
                midi_out.send_message([NOTE_OFF | ch, pitch, 0])
        if owns_port:
            midi_out.close_port()
