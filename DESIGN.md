# combo — design notes

Status: design consolidated 2026-08-06. This document is the single source of truth
for the design; individual decisions below supersede any earlier scattered notes.

**Built so far**: the gesture sub-gesture layer (§9, ported from AGRP) and the
song/scenario data model (§3) — both with passing tests, see [README.md](README.md).
**Designed but not yet built**: everything else below — voice/role architecture (§2),
generation modes (§4), accompaniment-listening (§5), drums (§7), the gesture
composition layer and its possible emergence (§9-10), the musical director (§11).
**Open research questions, not yet answered**: can sub-gesture sequences compose into
a genuine gesture grammar rather than a hand-authored one (§10); can the system
develop and recall recurring "tunes" of its own (§4, §9).

## 1. Vision

A self-playing multi-instrument improvising ensemble (bass, sax, keys, drums, roles
TBD) that can perform entirely on its own against a pulse — "like a radio station" —
or interactively with a live human for rehearsal. Successor/sibling to **wolfson**,
David's earlier system (not yet public), which proved trading licks, accompaniment
(emergent, beyond its original spec), and AI-vs-AI self-test all work for one human +
one AI voice trained on WJazzD. combo generalises that to N voices, adds a notion of
structure so the ensemble doesn't just noodle, and asks a genuine research question:
can the system develop — and later recall — recurring "tunes" and even a communicative
"grammar" of its own, rather than only reproducing what it was trained on?

## 2. Voices, not fixed chairs

The architecture is organised around a flexible set of **voices**, not instrument-named
chairs. Each voice has an instrument/register profile and a **source**: human-live
(monophonic pitch-tracked MIDI — see §6) or AI-generated. Any voice can take any role
(solo / accompany / lay out / trade) in any section, assigned by a tune-level form
controller (successor to Wolfson's `ArcController`, promoted from phrase-level to
tune-level).

- **Any instrument can accompany, not just solo** — a human can sit in on any chair,
  including an accompanying one (e.g. play bass while an AI voice solos), and the
  system's listening/response logic must be symmetric to that.
- **Same-instrument doubling is supported**: two saxes, two basses (human+AI, or AI+AI
  as a generalisation of Wolfson's self-test mode). When two same-register voices are
  both in an accompanying role at once, the **default is to split the role** — one
  plays full accompaniment, the other lays out or plays sparse punctuation — rather
  than both playing independently at full density and colliding.

## 3. Song as a persistent object

A **song** is changes + form: chord changes, and a form template (chorus length,
section order — e.g. "12-bar blues" or "AABA-32"), plus tempo/feel. MVP is changes+form
only, no composed head melody (a possible later layer). The system can be asked to
"play a song" — recall a stored one, or generate-and-store a new one under a name. This
is deliberately close to IRCAM ImproteK's "scenario" concept (see §11) — a chord chart
as an explicit, reusable, structural object — and is the piece of combo's design with
the most direct published prior art.

## 4. Three generation modes

Decoupled by what real-time constraint (if any) applies:

1. **Batch/offline self-generation** — no real-time pacing at all; the generator runs
   as fast as inference allows. For research-corpus generation (the emergent-tunes
   work, §9) and for quickly iterating the arc/harmony/role logic.
2. **Live radio-station self-play** — no human, but real-time paced because a live
   listener is tuning in.
3. **Interactive rehearsal** — real-time, reactive to a live human.

Architecturally this means generation should produce a symbolic timeline (notes tagged
with beat position/duration, not wall-clock time), with playback/scheduling as a
separate stage — only that later stage needs to know which mode it's in (write
immediately to a file, or pace to real time).

## 5. Accompaniment-listening

Accompanists extract live features (density, register, dynamics, space/rests) from
whichever voice(s) they're accompanying, and modulate around the planned arc
trajectory — mostly **complementary** (duck density when the soloist is busy, fill in
when they leave space), with occasional **mirrored** builds near arc peaks. Applies
symmetrically regardless of whether the accompanied voice is human or AI. The
same-register role-split default (§2) is this same complementary logic applied
laterally, between two accompanists, not just between accompanist and soloist.

## 6. Human input: monophonic pitch-tracked MIDI only

Scoped deliberately to a monophonic pitch-to-MIDI tracker (e.g. a Sonuus i2M) on bass,
sax, or any monophonic instrument — polyphonic/keyboard chord input is explicitly out
of scope, a much harder and separate problem. Any human-occupied voice uses this same
input path regardless of which instrument they're playing.

## 7. Drums

Start as a rule-based pattern engine (brushes, humanised timing/velocity,
section-aware density changes) rather than a trained model, since WJazzD has no drum
data. Generative/soloing drums are an explicit later phase, reusing the trading-licks
engine but substituting kit-voice + density contour for pitch contour as the
"melodic" analogue — a drum solo needs its own stand-in for phrase shape since drums
don't have pitch contour in the same sense a horn line does.

## 8. Handover / transition triggers

Fixed bar counts alone are good for rehearsal legibility, but listening-driven
transitions were more musically effective in Wolfson's own live-tested experience. The
resolution: bar counts are a **nominal scaffold** (a target section length), and a
**trained gesture vocabulary** is the concrete, learnable trigger that can pull a
transition early or late — closer to how a real band uses a nominal form as a shared
map while letting a cue move the actual boundary.

## 9. Gesture recognition and vocabulary

Built on [AGRP](https://github.com/davidderoure/AGRP), David's 2022 concert-tested,
rule-based gesture recogniser for George Lewis's Voyager/Forager work at PRiSM — proof
that a real, usable gesture vocabulary (runs, rips, staccato calls, trills, flutter
tongue, cresc-dim) can be recognised in real time purely from a monophonic pitch
tracker's event stream (pitch onsets + amplitude), no raw audio/timbral analysis
needed. That directly matches combo's input scope (§6).

- **Status**: the sub-gesture layer (rest / up / down / same / trill / long-note
  detection) is ported to Python — `gesture/recognizer.py`, `SubGestureRecognizer`,
  one instance per voice, tests passing. See [README.md](README.md) for what changed
  in porting (two confirmed bugs fixed, two more found and fixed, one accuracy
  improvement, one preserved-but-flagged behavioural quirk).
- **Not yet built**: composing sequences of sub-gestures into the named higher-level
  gesture vocabulary (runs vs. rips, up-down runs, forte-piano, etc.) — never finished
  in AGRP, and now understood to be better approached empirically (§10) than by
  hand-designing rules up front.
- These gestures were general call-and-response dialogue gestures in Voyager, not
  handover-specific — combo still needs to decide which gesture(s), or gesture+rest
  patterns, are handover triggers specifically vs. general responsive cues.
- **Symmetric across human and AI voices, matching §2**: gesture recognition isn't a
  human-only input feature. `SubGestureRecognizer`'s core interface is source-agnostic
  (quartertone note + amplitude events in, sub-gesture events out) — an AI voice's own
  generated output can be fed through the same recogniser as a live human's, so any
  voice can cue any other, exactly as Voyager's mutual gesture recognition intended.
- **PRiSM's alternative ML-based recogniser**
  ([PRiSM-MusicGestureRecognition](https://github.com/rncm-prism/PRiSM-MusicGestureRecognition))
  took a different approach — raw audio, spectrogram features, trained classifier per
  gesture — adopted for later performances after AGRP. It's a poorer fit for combo's
  monophonic-MIDI-only scope than AGRP's approach, which needs no raw audio pipeline.

## 10. Could the gesture vocabulary itself emerge, rather than being authored?

Prompted by Luc Steels' language-game work (Steels 1998, 2000 — robot populations
bootstrapping a shared lexicon and then rudimentary grammar through repeated
interaction, no central design). Concrete mapping onto combo, not yet built:

- **AGRP's fixed sub-gesture thresholds are already, literally, a hand-built
  discrimination game** (Steels' term for carving a continuous feature space into
  discrete categories, refined only when existing distinctions fail to disambiguate).
  A contained place to introduce real adaptivity later: let thresholds refine based on
  what's proving distinctive in use, instead of staying fixed constants.
- **The unsolved composition-rule layer (§9) maps onto naming games + chunking**:
  start with atomic sub-gestures as meaningful units; log
  (context, sub-gesture(s) played, ensemble response, outcome); let combinations that
  reliably succeed get **chunked** into new reusable named gestures (Steels' term —
  "the backbone of the grammar"). Steels' sharpest transferable idea: grammar/
  composition should only emerge once a flat vocabulary becomes genuinely ambiguous
  relative to what needs expressing — not designed in up front.
- **The hard open problem**: Steels' games work because there's a cheap, checkable
  ground truth (two robot heads watching the same scene). Music has no free equivalent
  for "did this gesture communicate what was intended." This is where the musical
  director (§11) comes in directly — see below.
- **Theoretical framing, useful if this is ever written up**: Steels frames this as a
  "Complex Adaptive Systems" account of language (meaning emerges from repeated
  interaction, never in a steady state), explicitly against Chomskyan Universal Grammar
  (fixed, innate). combo's gesture vocabulary — and its emergent-tunes idea (§9 in the
  memory, effectively "the same principle applied to whole songs") — both sit in this
  camp. It's a third lineage distinct from the Voyager/Lewis tradition (fixed,
  hand-scored vocabulary) and the IRCAM/DYCI2 tradition (memory-navigation, §12).
  George Lewis himself is independently interested in how gestures might form — worth
  keeping in mind alongside the Steels connection.

## 11. The musical director

A continuously-running critic/director that listens to the **overall ensemble** (not
one voice) and shares the same evaluation logic across two very different consumption
modes: in real-time modes it **nudges** generation as it happens (like a bandleader);
in batch mode the same signal, accumulated over a whole song, becomes a **score** for
curation (generate many candidate songs, pick the best — with automatic filters for
clear rule violations plus human curation on top, rather than trying to fully automate
aesthetic judgment). This generalises three previously-separate ideas into one
component: the tune-level form controller (a planned trajectory), listening-driven
transitions (nudging that plan from live cues), and batch song evaluation.

- **Nudge mechanism, decided**: dial-based, not directive. The director adjusts a
  small set of shared parameters (e.g. an intensity/density target, closeness to arc
  resolution) that voice-generators read and interpret individually — composes with
  the per-voice accompaniment-listening model (§5) rather than overriding it, and
  avoids the director becoming a single point of micromanagement.
- **Also the answer to §10's hard problem**: a live director's real-time input is a
  genuine ground-truth preference signal for reinforcing or abandoning gesture-meaning
  pairings — stronger than any proxy we'd otherwise have to invent.
- **Director is itself multi-instance and source-agnostic**, exactly like voices: N
  director "slots," each producing the same signal type, whether the source is a human
  at a control surface or an AI critic computing it from listening to the ensemble. An
  **aggregation stage** combines however many active director signals are present into
  the single effective dial-state voice-generators read — start simple (e.g. a
  weighted average), designed as a swappable piece so it can grow into something
  richer (e.g. clustering, if this ever scales toward many simultaneous listeners).
- **No app**: a human director is driven through the same MIDI infrastructure combo
  already uses (`python-rtmidi`) — a small physical control surface (fader/knob box,
  or a couple of MIDI CC controls) mapped to the shared dial parameters, not a web/
  phone layer. A human director and an AI director produce the same signal type
  through different sources; no separate code path.
- **Real prior art David performed with live**: Mood Conductor (Fazekas, Barthet,
  Sandler, QMUL, ACII 2013) — an audience, via a phone web app, marks a target mood on
  a 2D arousal-valence plane; responses are clustered in real time into "emotion blobs"
  shown to performers as live guidance. "Conductor" is explicitly metaphorical there
  too. Open Symphony (same group) is a related/successor project — audiences vote for
  discrete musical "modes" instead of a continuous position. Worth a citation nod later
  — combo isn't reusing their tech (no app, and a small number of directors rather than
  a crowd, at least to start) but the shape (a listening entity → continuous signal →
  live nudge) is the same lineage.

## 12. Positioning against prior art — IRCAM's OMax/ImproteK/Somax2/Dicy2 family

Researched in depth (papers read in full, not just abstracts) because a colleague
flagged them. **They don't supersede combo.** All four are memory/corpus-navigation
generators (factor oracle, concatenative synthesis, n-gram matching) driven by
continuous machine listening of audio/MIDI features — mechanically and philosophically
different from a trained generative model. Per the Somax2 paper's own framing, that
whole family "does not generate novel material ex nihilo, but recombines and
transforms what it has learned" (cliché re-injection, in ImproteK's own words) — combo
aims at *origination* (its own tunes, its own gesture grammar), not recombination.
None of the four use a discrete trained gesture vocabulary as a communication channel,
and none have combo's self-generation/persistent-tune-memory or musical-director ideas.

That said, real overlap is worth taking seriously, not dismissing:

- **ImproteK's "scenario"** is, in real jazz use, often literally a chord chart (one
  documented session used "Blue in Green"'s chart as the scenario while the memory was
  a recording of a *different* tune, "Autumn Leaves," transposed to fit) — the closest
  published prior art to combo's song-as-changes+form idea (§3), including running
  multiple simultaneous voices (accompaniment/solo) against one scenario. Cite
  directly if §3 is ever written up.
- **ImproteK's real-time architecture** — an offline scenario-guided generation module
  wrapped in a reactive layer that revises its own precomputed anticipations as live
  control events arrive — is a real working precedent for the musical director's
  plan-then-revise shape (§11), just scoped to one voice's harmonic continuity rather
  than an ensemble-level critic.
- **Somax2** is confirmed to run genuine multi-agent networks (documented live use with
  multiple simultaneous interacting instances) — the best existing precedent for
  combo's multi-voice coordination, independent of the differing generative mechanism.

**Decided**: happy to build on IRCAM/DYCI2 work where useful rather than reimplementing
everything from scratch — in particular, `DYCI2/Dicy2-python`'s generative core is a
candidate reusable substrate for some voices, with combo's actual novel contribution
being the layers on top (gesture vocabulary and its possible emergence, ensemble
form/song identity, self-play + emergent tunes, the musical director). Not yet decided
which voices, if any, would actually use it — a sketching-phase question.

## 13. Explicitly out of scope (for now)

- **Lookahead/search generation** ("chess-move" idea: generate a few bars, evaluate,
  branch/backtrack). Shares the same core dependency as batch curation (an evaluator),
  just far more often and expensively. Natural fit for batch/offline mode later; a
  poor fit for live/interactive performance, which can't pause to search before
  committing to the next notes.
- **Polyphonic/keyboard chord input** (§6).
- **Composed head melodies** as part of a song object (§3) — changes+form only for now.
- **Crowd-of-listeners director UI** (Mood-Conductor-style web app) — architecture
  should support N directors (§11), but the human-facing input starts as a small MIDI
  control surface, not an app.

## 14. Nods owed, not yet written

Small acknowledgments to fold in once there's something being written up, not
structural influences on the design itself:

- SHRDLU (Winograd) — David visited MIT in the 80s; a nod, not a design input.
- George Lewis's Voyager/Forager — direct lineage for §9, not just a nod.
- Steels' language games — direct design input for §10, not just a nod.
- Mood Conductor / Open Symphony — direct design input for §11 (§12 above).
