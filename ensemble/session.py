"""Runs a Song through a set of Voices to produce a Timeline — see DESIGN.md §4.

Generation always produces a symbolic (beat-based) timeline; the *mode* only controls
how the generation loop is paced, not what gets generated. "machine_speed" is the
one-shot-song-generator / batch case (§4.1): no pacing, runs as fast as inference
allows. "real_time" paces each bar to the song's tempo — the shared mechanism behind
both live radio-station self-play (§4.2) and interactive rehearsal (§4.3); which of
those it actually is depends on who's listening, not on anything this module does.
"""

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Optional, Protocol

from .director import Director, aggregate_director_signals
from .timeline import BEATS_PER_BAR, Timeline
from .voice import Voice

if TYPE_CHECKING:
    from song import Song

MACHINE_SPEED = "machine_speed"
REAL_TIME = "real_time"


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


@dataclass
class FakeClock:
    """Advances instantly on sleep(), but records how much virtual time was
    requested — lets tests verify real-time pacing without actually waiting."""

    virtual_time: float = 0.0
    total_slept: float = 0.0

    def now(self) -> float:
        return self.virtual_time

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.virtual_time += seconds
            self.total_slept += seconds


@dataclass
class Session:
    song: "Song"
    voices: List[Voice] = field(default_factory=list)
    directors: List[Director] = field(default_factory=list)

    def generate(self, mode: str = MACHINE_SPEED, clock: Optional[Clock] = None) -> Timeline:
        if mode not in (MACHINE_SPEED, REAL_TIME):
            raise ValueError(f"mode must be {MACHINE_SPEED!r} or {REAL_TIME!r}, got {mode!r}")

        total_beats = self.song.total_beats
        if total_beats is None:
            if mode == MACHINE_SPEED:
                raise ValueError(
                    "Song has an open-ended form (a section with no repeat count) — "
                    "machine-speed generation needs a defined endpoint to generate to."
                )
            raise NotImplementedError(
                "Open-ended real-time self-play isn't built yet — that's Phase 8+ "
                "territory (an indefinitely-running session), not this MVP."
            )

        if clock is None:
            clock = SystemClock()
        seconds_per_beat = 60.0 / self.song.tempo_bpm

        timeline = Timeline()
        bar_index = 0
        while bar_index * BEATS_PER_BAR < total_beats:
            # Every voice generating for this bar sees the same snapshot of prior
            # bars only (never the current bar, never affected by iteration order
            # over self.voices) — a copy, not the live timeline, so a generator can't
            # corrupt the loop by calling .add() on what it's handed (DESIGN.md §5).
            prior_bars = Timeline(list(timeline.events))

            # One aggregated signal per bar, shared by every voice generating that
            # bar (DESIGN.md §11). No directors configured -> aggregating [] ->
            # the neutral default -> existing generators are unaffected.
            signals = [
                director.signal_source(self.song, bar_index, prior_bars)
                for director in self.directors
                if director.source == "ai"
            ]
            director_signal = aggregate_director_signals(signals)

            bar_events = []
            for voice in self.voices:
                if voice.source != "ai":
                    continue
                for event in voice.generator(self.song, bar_index, prior_bars, director_signal):
                    bar_events.append(replace(event, voice_id=voice.id))
            for event in bar_events:
                timeline.add(event)

            if mode == REAL_TIME:
                clock.sleep(BEATS_PER_BAR * seconds_per_beat)

            bar_index += 1

        timeline.events.sort(key=lambda e: e.start_beat)
        return timeline
