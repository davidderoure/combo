# combo

See [DESIGN.md](DESIGN.md) for the full design notes.

Design work-in-progress for a self-playing multi-instrument improvising ensemble
(bass, sax, keys, drums) that can perform on its own against a pulse, or interactively
with a live human for rehearsal. Successor/sibling to
**wolfson**, David's earlier system (not yet public), which proved trading licks, accompaniment, and
AI-vs-AI self-test all work for one human + one AI voice; this project generalises to
N voices with a notion of song structure, and explores whether the system can develop
recurring "tunes" of its own.

## Current status

Early build-out, six pieces in so far. First: a **gesture recognition** layer for
monophonic note streams (bass, sax, or any instrument via a pitch-to-MIDI tracker, e.g.
a Sonuus i2M — but equally an AI voice's own generated output), so any performer, human
or AI, can cue the ensemble — handovers, dialogue — using a small vocabulary of
playable gestures rather than only note-level features. As in Voyager/Forager, gesture
recognition is meant to be mutual: any performer's gesture can prompt a response from
any other, not just human-to-AI.

This is a Python port of [AGRP](https://github.com/davidderoure/AGRP), David's 2022
rule-based gesture recogniser built for George Lewis's Voyager/Forager work at PRiSM
(see also the ML-based
[PRiSM-MusicGestureRecognition](https://github.com/rncm-prism/PRiSM-MusicGestureRecognition),
which took a different, audio/ML-based approach and was adopted for later performances).

`gesture/recognizer.py` ports AGRP's real-time sub-gesture state machine (rest / up /
down / same / trill / long-note detection from a stream of pitch + amplitude events).
It's a faithful port with a few fixes and one accuracy improvement — see comments at
the top of the file for what changed and why. It emits **sub-gestures**
(`R`, `X`, `U`, `D`, `S`, `T`, `L`) — composing those into named higher-level gestures
was never finished in AGRP; `gesture/vocabulary.py` picks that up (DESIGN.md §10).

`gesture/vocabulary.py`'s `GestureRecognizer` wraps a `SubGestureRecognizer` and
composes its sub-gesture stream into named `Gesture`s, via two mechanisms (§10.1/§10.2,
both with passing tests): a small **seeded** vocabulary (`handover()`, `reset_tempo()`
— placeholder patterns, not meant to be musically final any more than the ensemble's
stub generator was), and a **taught** mechanism — a reserved begin/end sub-gesture
marker pair, chosen for a checkable reason (two same-direction runs in a row can't
happen from ordinary continuous phrasing, per `SubGestureRecognizer`'s own state
machine — see the module docstring), that lets a human teach a new gesture entirely by
playing: alias an existing meaning to new material, or record something whose meaning
isn't resolved yet (stored, not guessed at). Recognising a gesture's *parameters*
(`handover(target=bass)`, not just `handover()`) and automatically working out a
genuinely new, unaliased meaning are both explicitly not attempted here — see DESIGN.md
§10 for why those are separate, harder problems.

Unlike AGRP (one browser tab per instrument channel), `SubGestureRecognizer` is a
plain class — one instance per voice, so multiple concurrent instruments can each get
their own recognizer within a single process. This mirrors how the AGRP concert setup
actually worked (one mic, one Sonuus, one recogniser per instrument) but makes that
modularity a first-class part of the code rather than a browser-tab convention.

Its core interface (`note_on`/`note_off`, and the `midi_note_on`/`midi_note_off`
convenience wrappers) is already source-agnostic — it doesn't care whether note events
arrive from a live MIDI port or are generated in-process by an AI voice, so an AI
voice's own output can be fed through the same recogniser without any extra bridging
code, just by calling those methods directly instead of going through `MidiListener`.

The second piece is the **ensemble skeleton** (DESIGN.md §2/§4): a
thin `Voice` (id, instrument, register, human-or-AI source) driving a `Session` that
steps a `Song` bar-by-bar and merges each AI-sourced voice's output into one symbolic
`Timeline` — notes tagged by beat position, not wall-clock time. Generation mode only
controls *pacing* of that same loop, not what gets generated: `machine_speed` runs
unpaced (the one-shot-song-generator case, §4.1), `real_time` paces each bar to the
song's tempo (self-play or interactive rehearsal, §4.2/§4.3 — which one depends on who's
listening, not on this code). `ensemble/generators.py`'s `chord_tone_generator` is a
deliberately dumb placeholder (root + fifth on beats 1 and 3) whose only job is proving
the pipeline end-to-end; real generation (an adapted Wolfson model, DYCI2/Dicy2-python,
or something new — DESIGN.md §12) replaces it once the rest of the skeleton (the
director) is built and proven against it.

The third piece is **drums** (DESIGN.md §7, `ensemble/drums.py`), a rule-based pattern
engine rather than a trained model (WJazzD has no drum data) — no ML, just another
`Voice`'s generator, reusing `Song.section_at` for section-aware density (sparse under
the head/out, busier toward the end of a section, standing in for "approaching a peak"
until there's a real `ArcController` to ask). Worth being explicit that "brushes"
(DESIGN.md §7's word) is approximated here as discrete hi-hat/ride/snare hits — a real
brush pattern is continuous sweeping texture that `NoteEvent` genuinely can't
represent, not just something this stub plays badly. `ensemble/demo.py` now runs the
sax stub and drums together, the first real multi-voice run of the ensemble.

The fourth piece is **accompaniment-listening** (DESIGN.md §5, `ensemble/listening.py`
+ `ensemble/comping.py`) — voices reacting to what other voices have actually played,
not just to the song/section state drums uses. This needed a real architectural change,
not just a new file: every `Generator` now receives a defensive-copy snapshot of prior
bars alongside `(song, bar_index)` (`ensemble/session.py` collects each bar's new
events separately and only merges them in after every voice has generated for that
bar, so listening is never affected by voice order — see
`tests/test_comping.py::test_voice_order_does_not_affect_output`).
`ensemble/listening.py` extracts all four features DESIGN.md §5 names — density,
register (as pitch range), dynamics (as average velocity), space/rests (as beats of
silence) — though only density is actually consumed by the accompanist built here; the
rest are extracted and tested for future consumers, said plainly rather than left
implicit. `comping_generator` ducks when its target voice is busy, fills with chord
stabs when it leaves space, plays one stab otherwise — demonstrated in
`ensemble/demo.py` against a synthetic varying-density fixture rather than the sax
stub, since `chord_tone_generator`'s output never varies in density at all.

The fifth piece is the **musical director**'s dial channel (DESIGN.md §11,
`ensemble/director.py`) — `Director`/`DirectorSource` deliberately mirror
`Voice`/`Generator`'s shape rather than a new pattern, and `Session` now aggregates one
`DirectorSignal` per bar (mean intensity across however many directors are configured,
neutral by default) that every voice's generator receives as a fourth argument.
`comping_generator` is the real consumer — its duck/fill thresholds shift with
intensity, byte-identical to Phase 4's behaviour at the neutral default, so this was
safe to bolt on rather than rewrite. `ensemble_intensity_critic` is a genuine, if
simple, "AI critic" (DESIGN.md §11's own phrase): it derives intensity by listening to
the ensemble's own combined density, not just accepting a manually-supplied constant —
demonstrated in `ensemble/demo.py` reading back the sax+drums session from the first
demo. The gesture channel's data model and aggregation are built (`DirectorSignal` can
carry a `Gesture`) but have no consumer yet — a director-emitted `reset_tempo()` has
nowhere to act until §4.1's runtime tempo and §8's handover triggers exist as code, said
plainly in the module docstring and in the demo's own output rather than left implicit.
Also unbuilt: batch-mode scoring.

The sixth piece is **multi-role MIDI input** (DESIGN.md §6, `input/sources.py`) —
the live-human-input gap the director and voices both had is now closed for the
performer and director roles. `config.MIDI_SOURCES` lists role-tagged sources;
`start_midi_sources` dispatches each to a `MidiListener`/`GestureRecognizer` pair
(performer) or a new `DirectorMidiListener` (director — reads one MIDI Control
Change into a live intensity value, `as_source()` returning a `DirectorSource` that
always reflects the *current* value, not a frozen snapshot, so a real fader genuinely
drives `Session.generate`'s per-bar aggregation as it moves). This environment has no
physical MIDI hardware, but does have macOS's virtual IAC Driver ports — used to
verify the whole pipeline for real rather than only in the abstract: an actual note-on
sent through a real (if virtual) port produced `Gesture("handover")` end-to-end, and
an actual CC message updated a live intensity value correctly. The automated test
suite stays hardware-independent regardless (`tests/test_midi_sources.py`, feeding
synthetic MIDI byte tuples directly, the same technique `gesture/recognizer.py`'s own
tests use), since the IAC-based check depends on macOS-specific infrastructure not
guaranteed present elsewhere. Not built: the audience/room-mic path, and anything
verified against real physical hardware.

## Layout

- `gesture/recognizer.py` — the ported sub-gesture state machine (no MIDI/IO deps)
- `gesture/vocabulary.py` — composes sub-gestures into named `Gesture`s (DESIGN.md
  §10.1/§10.2): `Gesture`, `GestureRule`, `GestureRecognizer` (wraps a
  `SubGestureRecognizer`, drop-in compatible with anywhere it's used today)
- `input/midi_listener.py` — `python-rtmidi`-based MIDI input, converts note/pitch-bend
  events into calls on a `SubGestureRecognizer`/`GestureRecognizer` (style follows
  wolfson's `input/midi_listener.py`)
- `input/sources.py` — role-based MIDI dispatch (DESIGN.md §6): `MidiSourceConfig`,
  `DirectorMidiListener` (CC -> live intensity), `MidiSources`, `start_midi_sources`
- `song/` — the song/scenario data model (DESIGN.md §3): `Chord`, `Changes` (one
  chorus' worth of chord changes), `Section`/`Song` (form = ordered sections, each a
  number of cycles through the changes), and `chart.py`, a plain-text chart format for
  authoring songs by hand — see `songs/blues_in_f.chart` for an example.
- `ensemble/` — the ensemble skeleton (DESIGN.md §2/§4): `timeline.py` (`NoteEvent`,
  `Timeline`), `voice.py` (`Voice`), `generators.py` (the stub `chord_tone_generator`,
  and `place_in_register`, shared with `comping.py`), `session.py` (`Session`,
  generation-mode dispatch, an injectable `Clock` so real-time pacing is testable
  without actually waiting), `drums.py` (DESIGN.md §7 — rule-based, section-aware
  `drum_generator`, no ML), `listening.py` (DESIGN.md §5 — feature extraction:
  `density`, `pitch_range`, `average_velocity`, `beats_of_silence`, plus the
  `synthetic_varying_density_generator` test/demo fixture), `comping.py` (DESIGN.md
  §5 — the concrete accompanist, `comping_generator`), `director.py` (DESIGN.md §11
  — `DirectorSignal`, `Director`, `aggregate_director_signals`,
  `constant_director_source`, `ensemble_intensity_critic`).
- `tests/test_recognizer.py`, `tests/test_gesture_vocabulary.py`, `tests/test_song.py`,
  `tests/test_session.py`, `tests/test_drums.py`, `tests/test_listening.py`,
  `tests/test_comping.py`, `tests/test_director.py`, `tests/test_midi_sources.py` —
  no MIDI hardware needed
- `listen.py` — small runnable script: starts every source in `config.MIDI_SOURCES`,
  prints recognised gestures (performers) and live intensity (directors)
- `gesture/demo.py` — small runnable script: replays synthetic gesture sequences
  (seed gestures, record + alias teaching) and prints what's recognised
  (`python -m gesture.demo`)
- `ensemble/demo.py` — small runnable script: generates a chart's worth of stub sax +
  drums output and prints it, then separate demonstrations of comping's duck/fill
  behaviour and the director's intensity dial (low vs. high, plus the AI critic
  reading the sax+drums session) (`python -m ensemble.demo`)

## Running

```
pip install -r requirements.txt
python listen.py --list          # show available MIDI ports
python listen.py                 # start every source in config.MIDI_SOURCES
python -m gesture.demo           # no MIDI needed
python -m ensemble.demo          # no MIDI needed
pytest
```
