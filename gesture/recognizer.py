"""Real-time sub-gesture recognition from a monophonic pitch + amplitude stream.

Python port of AGRP's agrp.js (https://github.com/davidderoure/AGRP), David's 2022
rule-based gesture recogniser for George Lewis's Voyager/Forager work. One
SubGestureRecognizer instance handles one voice; the AGRP concert setup used one
recogniser per instrument (via separate browser tabs) — here that's just one instance
per voice in the same process.

The state machine, thresholds and detection logic are a faithful port of the original.
Changes made while porting:

- State lives on instances (`self.*`) instead of module-level globals, since this
  project needs several concurrent recognizers (one per voice), not one per browser tab.
- Time is real seconds (`now`, a float, passed in by the caller) instead of an integer
  "tick" counter driven by a fixed 20ms polling loop. `tick()` still needs to be called
  periodically (e.g. every 20-50ms) to detect long notes and rests, same as the
  original's `timeout()`, but detection thresholds are expressed in seconds and no
  longer depend on the poll interval.
- Bug fix: the original's `lastAmplitude` was assigned the note number, not the
  amplitude, so notes synthesised during a pitch-bend correction got the wrong
  amplitude. Fixed here (`midi_note_on` sets it from `velocity`).
- Bug fix: the original's pitch-bend handler read a stray `event` variable instead of
  its own `lsb`/`msb` parameters. Fixed here (`midi_pitch_bend` uses its own args).
- Bug fix: `trillStartTime` was declared but never assigned in the original, so it was
  always 0 — the trill duration threshold check was effectively comparing against
  elapsed-time-since-page-load rather than elapsed-time-since-trill-start. Fixed here by
  setting `trill_start_time` wherever `trill_length` restarts from zero.
- Bug fix: `slope()`'s backward scan started one index past the end of the sequence
  (comparing against `undefined` on the first iteration) — an off-by-one. Fixed here.
- Accuracy improvement: `slope()`'s early-exit detection used to estimate a duration as
  `note_count * tick_duration`, an approximation (notes aren't actually 20ms apart).
  Since note events are now timestamped anyway, `_slope()` returns the real timestamp of
  the matched run instead of an estimate.

One thing deliberately *not* changed: several of the state-transition blocks below can
both fire for the same note-on event (e.g. the X-to-U transition immediately followed by
the U-continuation check, since `state` is mutated partway through the original
function). This looks unintentional but affects the tuned thresholds, and there's no way
to be sure it's a bug rather than something the original's threshold values were tuned
around — it's preserved as-is here, pending testing against real/replayed gesture
recordings.

This module only classifies sub-gestures (R/X/U/D/S/T/L) from note events; composing
sequences of sub-gestures into the named higher-level gesture vocabulary (runs vs.
rips, up-down runs, forte-piano, etc. — never finished in AGRP) is the next layer to
build on top of this.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class SubGesture:
    label: str  # "R", "U", "D", "S", "T", or "L"
    start_time: float
    duration: float
    n: int


class SubGestureRecognizer:
    """Classifies one monophonic voice's note stream into sub-gestures.

    Feed it `note_on` / `note_off` (or the MIDI convenience wrappers
    `midi_note_on` / `midi_note_off` / `midi_pitch_bend`), and call `tick()`
    periodically (e.g. every 20-50ms) so long notes and rests get detected even
    when no new note-on/off event has arrived.
    """

    def __init__(
        self,
        on_subgesture: Optional[Callable[[SubGesture], None]] = None,
        *,
        min_up_duration: float = 0.2,
        min_down_duration: float = 0.2,
        min_same_duration: float = 0.2,
        min_trill_duration: float = 0.2,
        min_rest_duration: float = 0.1,
        min_long_duration: float = 2.0,
        rest_confirm_duration: float = 0.4,
        trill_tolerance: int = 6,
        gradient: int = 6,
        pitch_bend_range: float = 400,
    ):
        self.on_subgesture = on_subgesture

        self.min_up_duration = min_up_duration
        self.min_down_duration = min_down_duration
        self.min_same_duration = min_same_duration
        self.min_trill_duration = min_trill_duration
        self.min_rest_duration = min_rest_duration
        self.min_long_duration = min_long_duration
        self.rest_confirm_duration = rest_confirm_duration
        self.trill_tolerance = trill_tolerance
        self.gradient = gradient
        self.pitch_bend_range = pitch_bend_range

        # state is R, X, U, D, S = rest, detected one note, up, down, same
        self.state = "R"
        self.start_time = 0.0
        self.start_pitch = 0

        self.up_length = 0
        self.up_start_time = 0.0
        self.up_start_pitch = 0

        self.down_length = 0
        self.down_start_time = 0.0
        self.down_start_pitch = 0

        self.same_length = 0
        self.same_start_time = 0.0
        self.same_start_pitch = 0

        self.trill_length = 0
        self.trill_start_time = 0.0
        self.trill_start_pitch = 0

        self.rest_length = 0
        self.rest_start_time = 0.0
        # -inf, not 0.0: a fresh recognizer has never had a note-off, and 0.0
        # would collide with a session whose very first note starts at t=0.0
        self.last_note_off_time = float("-inf")

        self.active_notes: list[int] = []  # monophonic in practice; kept as a list
        self._note_sequence: list[tuple[int, float]] = []  # (quartertone, time)
        self._previous_note: Optional[int] = None

        self._history: list[SubGesture] = []

        # MIDI-facing state (quartertone note tracking, pitch bend)
        self.last_note: Optional[int] = None
        self.last_amplitude: Optional[int] = None
        self.pitch_bend_correction = 0

    # -- core note-level interface -----------------------------------------

    def note_on(self, note: int, amplitude: int, now: float) -> None:
        """`note` is a quartertone pitch number, `amplitude` a 0-100 percentage."""
        previous = self._previous_note

        # enforce monophonic — a new note ends the previous one if still sounding
        if len(self.active_notes) == 1:
            self.note_off(self.active_notes[0], now)

        if note not in self.active_notes:
            self.active_notes.append(note)

        self._note_sequence.append((note, now))
        self._previous_note = note
        self.rest_length = 0

        if self.state == "R":
            self._enter_first_note(note, now)
            return

        self._transition(note, previous, now)

    def note_off(self, note: int, now: float) -> None:
        if note in self.active_notes:
            self.active_notes.remove(note)
        self.last_note_off_time = now
        self.tick(now)

    def tick(self, now: float) -> None:
        """Call periodically to detect long notes and rests between note events."""
        if self.active_notes:
            if (
                self.last_note_off_time < self.start_time
                and now - self.start_time > self.min_long_duration
            ):
                self._emit("L", self.start_time, now - self.start_time, 1)
            return

        if self.state == "R":
            if now - self.rest_start_time > self.min_rest_duration:
                self._emit("R", self.rest_start_time, now - self.rest_start_time, self.rest_length)
            self.rest_length += 1
            return

        if self.rest_length == 0:
            self.rest_start_time = now
            self.rest_length = 1
            return

        self.rest_length += 1

        if now < self.rest_start_time + self.rest_confirm_duration:
            return

        # confirmed rest — flush whatever sub-gesture was pending and enter R

        self._note_sequence = []

        if self.state == "D" and self.down_length > 4 and now - self.down_start_time > self.min_down_duration:
            self._emit("D", self.down_start_time, now - self.down_start_time, self.down_length)

        if self.state == "U" and self.up_length > 4 and now - self.up_start_time > self.min_up_duration:
            self._emit("U", self.up_start_time, now - self.up_start_time, self.up_length)

        if self.state == "S" and self.same_length > 4 and now - self.same_start_time > self.min_same_duration:
            self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)

        if self.state == "X" and now - self.start_time > self.min_same_duration:
            self._emit("S", self.start_time, now - self.start_time, 1)

        if self.trill_length > 10 and now - self.start_time > self.min_trill_duration:
            self._emit("T", self.start_time, now - self.start_time, 1)

        self.state = "R"
        self._emit("R", self.rest_start_time, now - self.rest_start_time, self.rest_length)

        self.down_length = 0
        self.up_length = 0
        self.same_length = 0
        self.trill_length = 0
        self.rest_length = 0

    # -- MIDI convenience wrappers ------------------------------------------

    def midi_note_on(self, midi_note: int, velocity: int, now: float) -> None:
        amplitude = int(velocity * 100 / 127)
        quartertone = midi_note * 2 + self.pitch_bend_correction
        self.note_on(quartertone, amplitude, now)
        self.last_note = midi_note * 2
        self.last_amplitude = amplitude

    def midi_note_off(self, midi_note: int, now: float) -> None:
        if self.last_note is not None:
            self.note_off(self.last_note + self.pitch_bend_correction, now)

    def midi_pitch_bend(self, lsb: int, msb: int, now: float) -> None:
        if self.last_note is None or not self.active_notes:
            return

        bend = lsb + 128 * msb
        cents = (bend - 8192) * self.pitch_bend_range / 16384

        if cents >= 0:
            new_correction = int((25 + cents) // 50)
        else:
            new_correction = -int((25 - cents) // 50)

        if new_correction == self.pitch_bend_correction:
            return

        if (self.last_note + new_correction) not in self.active_notes:
            self.note_off(self.last_note + self.pitch_bend_correction, now)
            self.pitch_bend_correction = new_correction
            self.note_on(self.last_note + self.pitch_bend_correction, self.last_amplitude or 0, now)

    # -- internals ------------------------------------------------------------

    def _enter_first_note(self, note: int, now: float) -> None:
        self.state = "X"
        self.start_time = now
        self.up_length = 0
        self.down_length = 0
        self.same_length = 0
        self.trill_length = 1
        self.trill_start_time = now
        self.start_pitch = note

    def _transition(self, note: int, previous: int, now: float) -> None:
        g = self.gradient

        if self.state == "X" and note > previous and note - previous < g:
            self.state = "U"
            self.up_length = 1
            self.same_length = 0
            self.up_start_time = self.start_time
            self.up_start_pitch = previous
            if self.up_length > 4 and now - self.up_start_time > self.min_up_duration:
                self._emit("U", self.up_start_time, now - self.up_start_time, self.up_length)

        if self.state == "X" and note < previous and previous - note < g:
            self.state = "D"
            self.down_length = 1
            self.same_length = 0
            self.down_start_time = self.start_time
            self.down_start_pitch = previous
            if self.down_length > 4 and now - self.down_start_time > self.min_down_duration:
                self._emit("D", self.down_start_time, now - self.down_start_time, self.down_length)

        if self.state == "X" and note == previous:
            self.state = "S"
            self.same_length = 1
            self.same_start_time = now
            self.same_start_pitch = previous
            if self.same_length > 4 and now - self.same_start_time > self.min_same_duration:
                self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)

        if self.state == "U" and note > previous and note - previous < g:
            self.up_length += 1
            if self.up_length > 4 and now - self.up_start_time > self.min_up_duration:
                self._emit("U", self.up_start_time, now - self.up_start_time, self.up_length)
            else:
                c, c_start = self._slope(1)
                if c > 4 and c_start is not None:
                    self._emit("U", c_start, now - c_start, c)

        if self.state == "D" and note < previous and previous - note < g:
            self.down_length += 1
            if self.down_length > 4 and now - self.down_start_time > self.min_down_duration:
                self._emit("D", self.down_start_time, now - self.down_start_time, self.down_length)
            else:
                c, c_start = self._slope(-1)
                if c > 4 and c_start is not None:
                    self._emit("D", c_start, now - c_start, c)

        if self.state == "S" and note == previous:
            self.same_length += 1
            if self.same_length > 4 and now - self.same_start_time > self.min_same_duration:
                self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)

        # trill check — independent of state
        if self.trill_length > 0 and abs(note - self.start_pitch) < self.trill_tolerance:
            if note != previous:
                self.trill_length += 1
            if self.trill_length > 10 and now - self.trill_start_time > self.min_trill_duration:
                self._emit("T", self.trill_start_time, now - self.trill_start_time, self.trill_length)
                self.state = "X"
        else:
            self.trill_length = 0

        if self.state == "S" and note > previous:
            if self.same_length > 4 and now - self.same_start_time > self.min_same_duration:
                self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)
            self.state = "U"
            self.up_start_time = now
            self.up_start_pitch = previous
            self.up_length = 1
            self.down_length = 0
            self.same_length = 0

        if self.state == "S" and note < previous:
            if self.same_length > 4 and now - self.same_start_time > self.min_same_duration:
                self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)
            self.state = "D"
            self.down_start_time = now
            self.down_start_pitch = previous
            self.up_length = 0
            self.down_length = 1
            self.trill_length = 0
            self.same_length = 0

        if self.state == "D" and note > previous:
            if self.down_length > 4 and now - self.down_start_time > self.min_down_duration:
                self._emit("D", self.down_start_time, now - self.down_start_time, self.down_length)
            self.state = "U"
            self.up_start_time = now
            self.up_start_pitch = previous
            self.up_length = 1
            self.down_length = 0
            if self.trill_length == 0:
                self.trill_length = 1
                self.trill_start_time = now
            self.same_length = 0

        if self.state == "U" and note < previous:
            if self.up_length > 4 and now - self.up_start_time > self.min_up_duration:
                self._emit("U", self.up_start_time, now - self.up_start_time, self.up_length)
            self.state = "D"
            self.down_start_time = now
            self.down_start_pitch = previous
            self.up_length = 0
            self.down_length = 1
            if self.trill_length == 0:
                self.trill_length = 1
                self.trill_start_time = now
            self.same_length = 0

        if self.state == "U" and note == previous:
            if self.same_length == 0:
                self.same_length = 1
                self.same_start_time = now
                self.same_start_pitch = previous
            else:
                self.same_length += 1
            if self.same_length > 0 and now - self.same_start_time > self.min_same_duration:
                if now - self.up_start_time > self.min_up_duration:
                    self._emit("U", self.up_start_time, now - self.up_start_time, self.up_length)
                self.up_length = 0
                self.down_length = 0
                self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)
                self.state = "S"

        if self.state == "D" and note == previous:
            if self.same_length == 0:
                self.same_length = 1
                self.same_start_time = now
                self.same_start_pitch = previous
            else:
                self.same_length += 1
            if self.same_length > 0 and now - self.same_start_time > self.min_same_duration:
                if now - self.down_start_time > self.min_down_duration:
                    self._emit("D", self.down_start_time, now - self.down_start_time, self.down_length)
                self.up_length = 0
                self.down_length = 0
                self._emit("S", self.same_start_time, now - self.same_start_time, self.same_length)
                self.state = "S"

    def _slope(self, direction: int) -> tuple[int, Optional[float]]:
        """How many recent notes roughly follow this gradient direction, and
        the real timestamp at which that run started."""
        seq = self._note_sequence
        n = len(seq)
        if n < 2:
            return (0, None)

        u = d = 0
        for i in range(n - 1, 0, -1):
            note_i = seq[i][0]
            note_prev = seq[i - 1][0]
            if note_i == note_prev:
                u += 1
                d += 1
            if note_i > note_prev:
                u += 1
            else:
                d += 1

            if u + d > 6:
                if direction > 0 and u <= d:
                    return u, seq[i - 1][1]
                if direction < 0 and u <= d:
                    return d, seq[i - 1][1]

        return (u, seq[0][1]) if direction > 0 else (d, seq[0][1])

    def _emit(self, label: str, start_time: float, duration: float, n: int) -> None:
        if self._history:
            last = self._history[-1]
            if last.label == label and last.start_time == start_time:
                self._history[-1] = SubGesture(label, start_time, duration, n)
                return

        sg = SubGesture(label, start_time, duration, n)
        self._history.append(sg)
        self._history = self._history[-20:]
        if self.on_subgesture:
            self.on_subgesture(sg)
