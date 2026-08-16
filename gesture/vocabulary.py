"""Composes SubGestureRecognizer's sub-gesture stream into named Gestures.

See DESIGN.md §10 (10.1 seeded, 10.2 taught). Scope of this MVP, deliberately not the
whole of §10:

- Only argument-less gestures (`handover()`, `reset_tempo()`) are actually recognised
  from raw playing. DESIGN.md's parameterised examples (`handover(target=bass)`,
  `trade(unit=fours)`) need a value for `target`/`unit` from somewhere, and nothing in
  a raw sub-gesture stream supplies one yet — the `Gesture` type can *carry* params
  (so a parameterised gesture is representable the moment there's a source for its
  value, most plausibly the teaching mechanism below), but nothing here populates them
  automatically.
- Recording without a following alias (§10.2's "genuinely new meaning" case) is
  implemented up to *storing* the recorded pattern as unresolved (`pending`) — the
  automatic inference of what it means from repeated context is §10.3's harder,
  Steels-style emergent mechanism and isn't attempted here.

The seed rules and record markers below are placeholders, explicitly not meant to be
musically final any more than ensemble/generators.py's chord_tone_generator was —
they exist to prove the mechanism, and the real patterns are meant to be developed
empirically against real/replayed gesture recordings (DESIGN.md §9), not designed in
the abstract.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .recognizer import SubGesture, SubGestureRecognizer


@dataclass(frozen=True)
class Gesture:
    name: str
    params: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class GestureRule:
    pattern: Tuple[str, ...]  # exact-match sequence of sub-gesture labels
    gesture: Gesture


# Placeholder seed vocabulary (§10.1) — a single long held note, a single trill.
DEFAULT_RULES: List[GestureRule] = [
    GestureRule(pattern=("L",), gesture=Gesture("handover")),
    GestureRule(pattern=("T",), gesture=Gesture("reset_tempo")),
]

# Record markers (§10.2), chosen for a concrete reason, not arbitrarily: a single
# continuous run only ever emits once in SubGestureRecognizer (its start time doesn't
# reset until the run breaks — see recognizer.py's _transition), so two same-direction
# runs *in a row* can only happen via two genuinely separate, deliberate runs, not
# ordinary continuous phrasing. That's the concrete version of the "wake word"
# requirement §10.2 calls for.
DEFAULT_BEGIN_RECORD: Tuple[str, ...] = ("U", "U")
DEFAULT_END_RECORD: Tuple[str, ...] = ("D", "D")


def _matches_tail(sequence, pattern: Tuple[str, ...]) -> bool:
    n = len(pattern)
    if n == 0 or len(sequence) < n:
        return False
    return tuple(list(sequence)[-n:]) == pattern


class GestureRecognizer:
    """Wraps a SubGestureRecognizer and composes its output into named Gestures.

    Drop-in compatible with anywhere a bare SubGestureRecognizer is used (same
    note_on/note_off/tick/midi_* methods, delegated to the wrapped instance).
    """

    def __init__(
        self,
        on_gesture: Optional[Callable[[Gesture], None]] = None,
        *,
        rules: Optional[List[GestureRule]] = None,
        begin_record: Tuple[str, ...] = DEFAULT_BEGIN_RECORD,
        end_record: Tuple[str, ...] = DEFAULT_END_RECORD,
        history_size: int = 4,
        sub_recognizer: Optional[SubGestureRecognizer] = None,
    ):
        self.on_gesture = on_gesture
        self.rules: List[GestureRule] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self.begin_record = begin_record
        self.end_record = end_record

        self._window: Deque[str] = deque(maxlen=history_size)
        self._recording: Optional[List[str]] = None
        self.pending: List[Tuple[str, ...]] = []  # recorded but unresolved (§10.2/§10.3)

        self.sub_recognizer = sub_recognizer or SubGestureRecognizer()
        self.sub_recognizer.on_subgesture = self.feed_subgesture

    # -- pass-throughs, so this is a drop-in replacement for SubGestureRecognizer ---

    def note_on(self, note: int, amplitude: int, now: float) -> None:
        self.sub_recognizer.note_on(note, amplitude, now)

    def note_off(self, note: int, now: float) -> None:
        self.sub_recognizer.note_off(note, now)

    def tick(self, now: float) -> None:
        self.sub_recognizer.tick(now)

    def midi_note_on(self, midi_note: int, velocity: int, now: float) -> None:
        self.sub_recognizer.midi_note_on(midi_note, velocity, now)

    def midi_note_off(self, midi_note: int, now: float) -> None:
        self.sub_recognizer.midi_note_off(midi_note, now)

    def midi_pitch_bend(self, lsb: int, msb: int, now: float) -> None:
        self.sub_recognizer.midi_pitch_bend(lsb, msb, now)

    # -- composition ----------------------------------------------------------------

    def feed_subgesture(self, sg: SubGesture) -> None:
        """Public on purpose: lets the composition layer be tested directly against
        synthetic SubGesture sequences, without needing real note timing — continuing
        AGRP's own replayed-test-gesture approach (see its design notes)."""
        label = sg.label
        self._window.append(label)

        if self._recording is not None:
            self._recording.append(label)
            if _matches_tail(self._window, self.end_record):
                recorded = tuple(self._recording[: -len(self.end_record)])
                self._recording = None
                self._finish_recording(recorded)
            return

        if _matches_tail(self._window, self.begin_record):
            self._recording = []
            return

        for rule in self.rules:
            if _matches_tail(self._window, rule.pattern):
                if self.on_gesture:
                    self.on_gesture(rule.gesture)
                return

    def _finish_recording(self, recorded: Tuple[str, ...]) -> None:
        if not recorded:
            return  # nothing was actually played between the markers

        for rule in self.rules:
            if _matches_tail(list(recorded), rule.pattern):
                self.rules.append(GestureRule(pattern=recorded, gesture=rule.gesture))
                return

        self.pending.append(recorded)
