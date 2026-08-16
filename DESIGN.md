# combo — design notes

Status: design consolidated 2026-08-06. This document is the single source of truth
for the design; individual decisions below supersede any earlier scattered notes.

**Built so far**: the gesture sub-gesture layer (§9, ported from AGRP), the
song/scenario data model (§3), and an ensemble skeleton MVP (§2/§4 — see
[README.md](README.md)) — a thin `Voice`, a `Session` that steps a `Song` bar-by-bar
into a symbolic `Timeline`, and machine-speed-vs-real-time pacing behind a single
generation loop, proven end-to-end with a deliberately dumb `chord_tone_generator`
stub. All with passing tests. **Not yet built even within §2/§4**: role assignment,
same-instrument doubling and the register-split default, multi-human/multi-voice
sessions, and tempo elasticity (§4.1/§4.2) — the MVP is one song, any number of
AI-sourced stub voices, no roles, and a single fixed tempo for the whole song.
**Designed but not yet built at all**: accompaniment-listening (§5), drums (§7), the
gesture vocabulary-establishment mechanisms (§10) and composition layer, the musical
director and its two channels (§11), and the unified MIDI-only human input covering
performer, director, and audience (§6). Note that `input/midi_listener.py` currently
only wires up a single MIDI port (`config.MIDI_INPUT_PORT`) — a concrete gap against
§2's multi-human principle, not a design decision; it needs to grow into one
listener/recogniser pair per human voice, and eventually a room-mic path for the
audience case (§6). See `/Users/davidderoure/.claude/plans/modular-dazzling-emerson.md`
for the full build-order plan across all remaining subsystems.
**Open research questions, not yet answered**: can sub-gesture sequences compose into
a genuine gesture grammar rather than a hand-authored one (§10.3); can the system
develop and recall recurring "tunes" of its own (§1, §4).

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

**Cross-cutting principle (applies to voices here and to directors in §11)**: every
role in combo — performer/voice, director/listener, and any future role — is a set of
N independently-sourced slots, each either human or AI, with no assumption that only
one slot in the whole system is ever human. Multiple simultaneous humans in the same or
different roles is a real prior-tested case, not a hypothetical one: the AGRP concert
setup (§9) already ran five human performers at once, each on their own Sonuus tracker
and MIDI channel. Only the human-facing input plumbing needs to grow to match this —
see the note on `MidiListener` below.

- **Any instrument can accompany, not just solo** — a human can sit in on any chair,
  including an accompanying one (e.g. play bass while an AI voice solos), and the
  system's listening/response logic must be symmetric to that.
- **Same-instrument doubling is supported**: two saxes, two basses — any combination of
  sources (human+AI, human+human, or AI+AI, the last a generalisation of Wolfson's
  self-test mode). When two same-register voices are
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

## 4. Generation modes, paced by whether a human occupies any role

The real-time constraint on generation is a direct consequence of §2/§11's principle,
not a separately-chosen "mode": it follows from whether a human occupies *any* role at
all — performer or director/listener — not from a fixed enumeration. Three cases fall
out of that rule:

1. **Machine speed — no human in any role.** No real-time pacing at all; the generator
   runs as fast as inference allows. This covers both the **one-shot song generator**
   (generate a single song right now, as fast as possible, then return it — nobody is
   listening live) and generating a large research corpus (the emergent-tunes work,
   §9) or quickly iterating the arc/harmony/role logic — same mechanism, different call
   pattern, not different modes.
2. **Live radio-station self-play** — no human performer, but a human occupies the
   director/listener role (even just "an audience is listening," per Mood Conductor,
   §11) — must pace to real time, because a human ear is receiving it live even though
   nothing is playing back *to* the ensemble.
3. **Interactive rehearsal** — a human performer is present — real-time, reactive.

Architecturally this means generation should produce a symbolic timeline (notes tagged
with beat position/duration, not wall-clock time), with playback/scheduling as a
separate stage — only that later stage needs to know whether a human occupies any role
right now (return the result immediately, or pace to real time).

### 4.1 Tempo is a runtime value, not a fixed constant

The current MVP (`ensemble/session.py`'s `Session.generate`) paces real-time mode off a
single `song.tempo_bpm` fixed for the whole song — deliberately rigid, since Phase 1
was only about proving the pacing mechanism worked at all, not about musical
elasticity. The intended direction, prompted by David asking how rigid the pulse
should be (combo has a deliberate pulse; Wolfson never did — its timing was always
derived from phrases, never clock-driven): tempo, and tempo nudges (a push, a rit., a
snap back to nominal), should be one of the director's dial parameters (§11), read live
each bar rather than computed once at the start of generation. A tempo gesture —
`reset_tempo()`, alongside §10.1's `handover`/`trade` frames — is then just another
producer writing to that same dial; no new mechanism needed beyond what's already
planned for the director and the seeded vocabulary.

**Three sources can feed the tempo dial**, none needing new architecture beyond what's
already planned: the song's nominal tempo (the default); an explicit gesture
nudge/reset (above); and — prompted by David's rehearsal/teaching question below —
**tempo tracked live from a performer's actual note-onset timing**, extending §5's
accompaniment-listening feature extraction (which already reads live density/register/
dynamics) to also read timing. This is really Wolfson's original beat-sync mechanism
(aligning a proactive phrase to the bassist's actual beat) generalised from a one-off
alignment into a continuously tracked tempo. It's smaller and nearer-term than §4.2's
full free-time stretch goal, and covers a lot of the same practical ground for this
specific case: a learner deliberately playing slowly doesn't need the ensemble to
abandon the clock entirely, just to have the clock's rate genuinely follow them instead
of sitting fixed at the song's nominal tempo.

### 4.2 Free time (stretch goal, captured now so it shapes nearer-term decisions)

A sharper version of the same question: can a stretch of playing come off the pulse
*entirely*, not just have its rate nudged — genuinely phrase-derived timing for a
passage, the way Wolfson always worked, rejoining the pulse later? Not being built now,
but worth capturing because it's a different axis from §4.1 above (that's "is the
clock's rate steady"; this is "is there a clock running at all"), and conflating them
risks shaping the tempo-dial mechanism in a way that can't later accommodate a
genuinely clock-free stretch.

- **Real precedent this isn't unusually ambitious**: ImproteK's real-time architecture
  (§12) already syncs its audio rendering "with a non-metronomic beat" — a live,
  elastic pulse, not a fixed click. This is a solved problem elsewhere, not
  speculative.
- **Likely shape**: a per-section property in the form (§3) — "free time" vs. "in
  time," the same way real charts already notate a rubato intro. A free section
  advances on phrase/gesture completion (closer to how Wolfson actually worked) rather
  than fixed beat-interval pacing; a resolving gesture or structural landmark is what
  snaps the ensemble back onto the pulse — the same handover-style gesture mechanism
  as §8, just triggering a return to clock time instead of a role change.

## 5. Accompaniment-listening

Accompanists extract live features (density, register, dynamics, space/rests) from
whichever voice(s) they're accompanying, and modulate around the planned arc
trajectory — mostly **complementary** (duck density when the soloist is busy, fill in
when they leave space), with occasional **mirrored** builds near arc peaks. Applies
symmetrically regardless of whether the accompanied voice is human or AI. The
same-register role-split default (§2) is this same complementary logic applied
laterally, between two accompanists, not just between accompanist and soloist.

## 6. Human input: everything through MIDI, no control panels or apps

One principle covers every role: all human interaction with combo — performer,
director, audience — goes through MIDI (a pitch-tracked voice/instrument, or a MIDI
keyboard/controller), never a bespoke control panel or app.

- **Performer**: a monophonic pitch-to-MIDI tracker (e.g. a Sonuus i2M) on bass, sax,
  voice, or any monophonic instrument. Polyphonic/keyboard **chord** input is
  explicitly out of scope here — extracting a musical line from chords is a much
  harder, separate problem (§13). Any human-occupied voice uses this same input path
  regardless of which instrument they're playing.
- **Director**: a MIDI keyboard or controller, read as discrete note/CC values mapped
  to the shared dial parameters (§11). This is a trivial use of keyboard MIDI — reading
  control values, not extracting a melodic line — so it doesn't reopen the chord-input
  exclusion above; the two are different problems that happen to share hardware. No
  dedicated fader/knob box needed.
- **Audience**: no phone app (supersedes the earlier Mood-Conductor-style deferral,
  §13) — a room/ambient microphone through the same pitch-tracker-to-gesture pipeline
  already built for a single performer (§9), treated as one aggregate voice feeding the
  director's aggregation stage (§11). Reuses the existing architecture entirely; the
  audience isn't a new subsystem, just another source for machinery that already
  exists.

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

## 10. Establishing the gesture vocabulary

Three complementary mechanisms, not competing ones — build roughly in this order,
since each rests on the previous one existing rather than starting from nothing.

### 10.1 Seeded

A small starter set of gestures with designer-assigned meanings, present from day one
(as Lewis's Voyager score, or AGRP's hard-coded eight, were). Worth having from the
start, and worth noticing these span two different shapes:

- Plain signals, no argument — e.g. "I'm finishing now, handing over": the form
  controller's existing role-assignment logic picks who receives it.
- **Parameterised frames**, not flat symbols — e.g. "...handing over to a bass
  solo..." is really `handover(target=bass)`; "let's play fours" is
  `trade(unit=fours)` (a proposal to change the *form*, §3/§8, not a handover at all,
  with a parameter on the trading unit — fours, vs. twos, vs. eights). Seeding with a
  handful of small two-slot frames (action + target/parameter) rather than a longer
  flat list gives the emergent-grammar work (§10.3) real compositional structure to
  extend later — directly analogous to the "groups" with roles and an order constraint
  that Steels' agents built up from repeated two-word combinations.
- A **tempo gesture** — `reset_tempo()` at minimum, possibly a push/pull pair — belongs
  in this same seed set (§4.1): it's a plain signal like the handover example above,
  just writing to the director's tempo dial instead of triggering a role change.

### 10.2 Taught

The thing AGRP could have done but never built: a human teaches a new gesture, ahead of
a session or live, entirely through playing — no separate control panel, per §6.

- **Mechanism**: a reserved "begin-record" gesture, followed by the new material,
  followed by a reserved "end-record" gesture — implemented entirely on top of the
  existing `SubGestureRecognizer` pipeline, no new subsystem. Works identically whether
  it happens in a pre-concert teaching session or mid-performance.
- **Teaching by alias**: following end-record immediately with an *existing* known
  gesture means "this new thing means the same as that" — no naming or labelling step
  needed at all, entirely musical.
- **Teaching a genuinely new meaning** (not an alias): recording without a following
  known gesture gives a tightly scoped window ("a new definition is happening right
  now") rather than a label — the meaning still has to be inferred from what happens
  structurally right after, across a few repetitions. Much more tractable than the
  fully unscoped version of this problem (§10.3), because the record markers narrow
  down *when* a new definition is being attempted, not just *that* one might be.
- **Risk to design around**: the begin/end markers need to be gestures unlikely to
  occur by accident during normal playing — the same "wake word" problem voice
  assistants have. Choose something deliberately distinctive for those two, not
  anything already in the working vocabulary.
- The director can teach and cue this way too, not just through the dial channel —
  see §11's note on the two-channel director.

### 10.3 Emergent

Prompted by Luc Steels' language-game work (Steels 1998, 2000 — robot populations
bootstrapping a shared lexicon and then rudimentary grammar through repeated
interaction, no central design, no explicit teaching). The harder, longest-run
mechanism of the three — rests on 10.1 and 10.2 already existing:

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
  (fixed, innate). combo's gesture vocabulary — and its emergent-tunes idea (§1/§4,
  effectively "the same principle applied to whole songs") — both sit in this
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

- **Teacher is a purpose this role can serve, not a new role**: in rehearsal (§4, case
  3), a director slot can be occupied by, or configured for, teaching. Live tempo-
  tracking (§4.1) is the first concrete mechanism this motivates — the ensemble
  adapting to a learner playing slowly, rather than the learner having to keep up with
  a fixed nominal tempo — and it needs no new architecture beyond what §11 already has.
  Other teaching-specific behaviour (patience, simplified accompaniment, practice
  loops) is a later question, deliberately not specified now.
- **Two channels, not one**: the **dial** (continuous, ambient — intensity, closeness
  to arc resolution) sits alongside the **same gesture vocabulary** every performer
  shares (§9/§10) for discrete, structural decisions — an explicit "let's trade fours"
  or handover cue, not just a gradual nudge. Mirrors how a real bandleader mostly
  shapes energy continuously but occasionally gives an explicit cue. The director is
  just another participant that can emit and recognise gesture events, same channel as
  everyone else (§6) — it isn't a separate mechanism bolted on.
- **Dial nudge mechanism, decided**: dial-based, not directive, for the continuous
  channel specifically. The director adjusts a small set of shared parameters (e.g. an
  intensity/density target, or **tempo** — §4.1) that voice-generators read and
  interpret individually — composes with the per-voice accompaniment-listening model
  (§5) rather than overriding it, and avoids the director becoming a single point of
  micromanagement. (The gesture channel above is deliberately more directive — that's
  fine, it's playing the same role an explicit human cue does.)
- **Also the answer to §10.3's hard problem**: a live director's real-time input is a
  genuine ground-truth preference signal for reinforcing or abandoning gesture-meaning
  pairings — stronger than any proxy we'd otherwise have to invent.
- **Director is itself multi-instance and source-agnostic**, exactly like voices: N
  director "slots," each producing the same signal type, whether the source is a human
  (via §6's MIDI keyboard/controller or room mic) or an AI critic computing it from
  listening to the ensemble. An **aggregation stage** combines however many active
  director signals are present into the single effective dial-state voice-generators
  read — start simple (e.g. a weighted average), designed as a swappable piece so it
  can grow into something richer (e.g. clustering) if this scales toward many
  simultaneous listeners.
- **Real prior art David performed with live**: Mood Conductor (Fazekas, Barthet,
  Sandler, QMUL, ACII 2013) — an audience, via a phone web app, marks a target mood on
  a 2D arousal-valence plane; responses are clustered in real time into "emotion blobs"
  shown to performers as live guidance. "Conductor" is explicitly metaphorical there
  too. Open Symphony (same group) is a related/successor project — audiences vote for
  discrete musical "modes" instead of a continuous position. Worth a citation nod later
  — combo isn't reusing their tech (§6's room-mic mechanism replaces the app entirely)
  but the shape (a listening entity → continuous signal → live nudge) is the same
  lineage.

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
- **Polyphonic/keyboard chord input**, for the performer role specifically (§6) — the
  director's use of a MIDI keyboard is a different, much simpler problem and is in
  scope.
- **Composed head melodies** as part of a song object (§3) — changes+form only for now.

## 14. Nods owed, not yet written

Small acknowledgments to fold in once there's something being written up, not
structural influences on the design itself:

- SHRDLU (Winograd) — David visited MIT in the 80s; a nod, not a design input.
- George Lewis's Voyager/Forager — direct lineage for §9, not just a nod.
- Steels' language games — direct design input for §10, not just a nod.
- Mood Conductor / Open Symphony — direct design input for §11 (§12 above).
