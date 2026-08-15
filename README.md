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

Early build-out. First piece under construction: a **gesture recognition** layer for
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
(`R`, `X`, `U`, `D`, `S`, `T`, `L`), not yet the higher-level named gestures (runs vs.
rips, up-down runs, forte-piano, etc.) — composing those from sequences of sub-gestures
was never finished in AGRP and is the next piece of work here, to be developed against
real/replayed gesture recordings rather than designed in the abstract.

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

## Layout

- `gesture/recognizer.py` — the ported sub-gesture state machine (no MIDI/IO deps)
- `input/midi_listener.py` — `python-rtmidi`-based MIDI input, converts note/pitch-bend
  events into calls on a `SubGestureRecognizer` (style follows wolfson's
  `input/midi_listener.py`)
- `song/` — the song/scenario data model (DESIGN.md §3): `Chord`, `Changes` (one
  chorus' worth of chord changes), `Section`/`Song` (form = ordered sections, each a
  number of cycles through the changes), and `chart.py`, a plain-text chart format for
  authoring songs by hand — see `songs/blues_in_f.chart` for an example.
- `tests/test_recognizer.py`, `tests/test_song.py` — no MIDI hardware needed
- `listen.py` — small runnable script: live MIDI in, prints detected sub-gestures

## Running

```
pip install -r requirements.txt
python listen.py --port 0   # list ports with --list
pytest
```
