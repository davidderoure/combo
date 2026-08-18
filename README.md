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

Early build-out, eight pieces in so far. First: a **gesture recognition** layer for
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
**Updated in Phase 15** (DESIGN.md §2): the same-register role-split default for
two accompanists is now built (`ensemble/roles.py`'s `default_accompanist_roles`,
a greedy register-overlap rule; `comping_generator`'s new `lay_out` parameter).
Deliberately decided once at ensemble-construction time, not live per bar —
computing it live would mean a voice's generator needing to know what another
voice is *about* to play this same bar, which conflicts with the tested
voice-order-independence guarantee above. The larger tune-level solo/accompany/
lay-out/trade assignment (any voice, any role, any section — needs the rest of
`ArcController`) is still entirely unbuilt.

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
carry a `Gesture`) but isn't wired to anything that acts on it — §8's handover
triggers now exist as code (below), but they consume gestures from a `Session`'s
`gesture_source`, not from `DirectorSignal.gesture`, so a director-emitted `handover()`
still has nowhere to act, said plainly in the module docstring and in the demo's own
output rather than left implicit. Also unbuilt: batch-mode scoring.

The sixth piece is **multi-role MIDI input** (DESIGN.md §6, `input/sources.py`) —
the live-human-input gap the director and voices both had is now closed for the
performer and director roles. `config.MIDI_SOURCES` lists role-tagged sources;
`start_midi_sources` dispatches each to a `MidiListener`/`GestureRecognizer` pair.
This environment has no physical MIDI hardware, but does have macOS's virtual IAC
Driver ports — used to verify the whole pipeline for real rather than only in the
abstract: an actual note-on sent through a real (if virtual) port produced
`Gesture("handover")` end-to-end, and an actual CC message updated a live intensity
value correctly. The automated test suite stays hardware-independent regardless
(`tests/test_midi_sources.py`, feeding synthetic MIDI byte tuples directly, the same
technique `gesture/recognizer.py`'s own tests use), since the IAC-based check
depends on macOS-specific infrastructure not guaranteed present elsewhere.

**Updated in Phase 13** (DESIGN.md §11): checking a specific request — a director
should be able to use the *same* interface a performer does, not a crippled dial
("dual control car") — surfaced a general principle: **role determines
destination, not recognition capability**. `DirectorMidiListener`, checked
directly, really did only handle Control Change and silently ignored notes
entirely; but `GestureRecognizer`/`MidiListener` never cared who was playing.
`DirectorMidiListener` is retired — `MidiListener` (`input/midi_listener.py`) is
now the one listener type for every source, with an optional `cc_number`
alongside its existing recognizer, so recognition is uniform and only *routing*
varies by role (`input/sources.py`'s `_director_source` latches a director's
recognised gesture and returns it, consumed once, alongside live intensity — the
same "current value, not a frozen snapshot" pattern the old dial channel already
used, extended to a discrete event). Not built: a source feeding more than one
destination at once (representable now, not wired up), live human note-capture
into the `Timeline` for a `source="human"` `Voice`, `listen.py` becoming a full
live-performance driver (found along the way: even a performer's live gestures
don't reach a running `Session` today, `listen.py` only prints them), the
audience/room-mic path, and anything verified against real physical hardware.

The seventh piece is **handover/transition triggers** (DESIGN.md §8,
`ensemble/transitions.py`) — the first real slice of the long-referenced
`ArcController` (still missing everywhere else it's cited: §5, §7, §11), specifically
the transition-timing piece. A recognised `handover()` (the only seeded gesture whose
meaning maps onto a transition — `reset_tempo()` is untouched, that's §4.1's still-
unbuilt tempo dial) shortens the current section to end after its current chorus.
`TransitionController.effective_song()` returns a form-truncated `Song`, and
`Session` now passes that *effective* song through the exact same parameter slot
every generator already reads `section_at`/`chord_at` from — `chord_tone_generator`,
`drum_generator`, and `comping_generator` all pick this up with **zero code
changes**, confirmed by this phase touching none of those three files. Proven against
a real consumer, not just checked in isolation: `ensemble/demo.py` shows
`drum_generator`'s section-aware density shifting 12 bars early after a scripted
handover. `LiveGestureQueue` is the **first thread-safety primitive in this
codebase** — genuinely needed now that a live MIDI callback thread (§6) and
`Session.generate`'s real-time loop run concurrently, not a precaution added out of
habit. Not built: "pulling late" (no gesture exists for it), genuine shortening of
the *total* performance length (a handover reallocates which section plays when,
within the same nominal duration), and wiring the director's gesture channel to this
mechanism.

The eighth piece is **real generation for the sax voice** (DESIGN.md §12,
`ensemble/wolfson/` + `ensemble/sax.py`) — the first voice to move off
`chord_tone_generator`. `ensemble/wolfson/` is a near-verbatim port of the separable
generative core of **wolfson** (David's earlier system): the trained LSTM
(`PhraseModel`), its token vocabulary, and its full inference-time bias-layer
pipeline — see the module's own docstring for exactly what changed mechanically
(relative imports, inlined constants) versus what's an unmodified port. `IRCAM`'s
`DYCI2/Dicy2-python` was considered and set aside for this phase on practical
grounds (not pip-installable, macOS+Python-3.9-only, and GPLv3-licensed — a real
concern for a public repo with no LICENSE file), and kept as a documented, deferred
candidate for a future *comping* voice specifically — see DESIGN.md §12. Real
inference (not a mock) runs in tests: `chord_to_wolfson_index` translates combo's
`Chord` into Wolfson's chord vocabulary (a total, exhaustively-tested mapping);
`sax_generator` builds a seed phrase from a target voice's recent notes (mirroring
`comping_generator`'s lookback-window pattern). The director's aggregated
intensity (DESIGN.md §11) reaches this generation too: `rhythmic_density` is the
one bias parameter the model itself already frames as a general busyness dial
("0=lyrical/slow, 1=bebop/fast"), so `sax_generator` passes
`director_signal.intensity` straight through — no translation function needed, and
grounded in a real probe (0.723 vs. 0.419 beats average note duration at
`rhythmic_density` 0.0 vs. 1.0) before the test threshold was picked.

The ninth piece is **MIDI output** (DESIGN.md §4, `output/midi_output.py`) — the
"playback/scheduling as a separate stage" §4 already named as designed-but-not-built.
Until now every generator's output only ever got printed as text; nothing in combo
could actually be heard. `build_schedule` (pure — Timeline + tempo + a
voice_id-to-channel map -> a time-sorted list of MIDI note-on/note-off messages) and
`play_timeline` (a single loop that sleeps until each scheduled message's time and
sends it, all-notes-off in a `finally` block regardless of how playback ends) are
deliberately much simpler than wolfson's own `output/midi_output.py`: wolfson
continuously interleaves generation and playback one phrase at a time, needing a
dedicated output thread and a "latest wins" pending-queue; combo's `Session` already
generates a whole multi-voice `Timeline` up front (`machine_speed` mode), so the
entire schedule is known before playback starts and no thread coordination is needed.
`play_timeline` reuses `ensemble/session.py`'s `Clock`/`FakeClock` directly rather
than inventing a second pacing abstraction — the same mechanism `test_session.py`
already uses to verify real-time pacing without actually waiting, applied here to
verify playback scheduling the same way. `self_test.py` (new, top-level, sibling to
`listen.py`) is the first thing that generalises Wolfson's AI-vs-AI self-play to
combo's whole ensemble: bass, sax (real generation, skipped with a clear
message if `sax_best.pt` isn't present), two role-split comping voices (Phase 15,
now audible rather than only measured), and drums — generated once, then played
through a real MIDI output port. `--loop N` plays the chart N times sharing one
`RehearsalMemory`, making Phase 11's rehearsal idea audible rather than only tested.
**Updated in Phase 17**: heard for real (through Logic Pro, not assumed), the
original bass stand-in — `chord_tone_generator`'s simultaneous root+fifth
double-stop, only on beats 1 and 3 with a full beat of silence after each hit —
sounded thuddy and staccato, the double-stop and the gaps, not the sample or note
duration alone. `self_test.py` now has its own `walking_bass_stub`: a single note
per beat, alternating root/fifth, sustained for nearly the full beat — local to
this script, the shared/tested `chord_tone_generator` untouched.
**Updated in Phase 18**: a real stuck note during testing traced to
`play_timeline`'s cleanup only ever sending CC 123/120 (All Notes Off/All Sound
Off) — wolfson's own `output/midi_output.py` already documented that Logic's
software instruments ignore both and need an explicit `note_off` per pitch;
the same fix is now reused here rather than rediscovered from scratch.

Sax also plans several bars ahead now (Phase 10), prompted by David asking directly
whether bar-by-bar generation loses a soloist's "conscious planning." What planning
means mechanically here: `sax_generator` generates a chord-consistent multi-bar span
— capped by `plan_bars` (default 4, matching Wolfson's own native
`MAX_PHRASE_BEATS`) or the next chord change, whichever is sooner, since
`PhraseGenerator.generate()` only accepts one chord broadcast across a whole call —
in ONE continuous call, so the model's own arc-position-driven bias layers (voice-
leading, contour) sweep across the real planned span instead of resetting every
bar, then dispenses one bar per `Session` call from an internal buffer, refilling
only when exhausted (`_bars_until_chord_change`, `_split_phrase_into_bars`, both in
`ensemble/sax.py`). No revision-on-mismatch mechanism exists: checked directly, a
handover only ever changes `Song.form`, never `Song.changes`, and `chord_at` only
reads `changes` — a plan's chord assumptions can never go stale, so building
revision logic would have been untestable, unreachable code. `blues_in_f.chart`'s
fast harmonic rhythm (chord changes almost every bar) limits the visible effect
there (45 `generate()` calls for 60 bars); a chart with an 8-bar chord hold shows it
clearly (2 calls for 8 bars) — both demonstrated in `ensemble/demo.py`, said
honestly rather than only showing the flattering case.

Sax also has **rehearsal memory** now (Phase 11, `ensemble/memory.py`'s
`RehearsalMemory`) — **the first thing in combo that persists across separate
`Session.generate()` calls**, everything else being deliberately fresh-per-
performance. Prompted by David distinguishing search-in-flight ("the chess
approach... something I'd like to try later," still deferred) from *rehearsal*:
play a piece more than once, carry what worked into the next run, so the actual
performance happens on the fly but is informed by practice. Real prior art existed
and was partly reused: wolfson's `input/phrase_analyzer.py::extract_interval_motifs`
(2/3/4-note transposition-invariant interval n-grams) ported near-verbatim into
`ensemble/wolfson/motifs.py`; wolfson's `memory/phrase_memory.py::PhraseMemory` — a
close match in shape (store/recall via a `Counter`) — was re-authored rather than
ported, since its reset policy (between `ArcController`'s arc loops, within one
live performance) doesn't fit persisting *across* performances. `sax_generator`
reads and writes `memory` at exactly the point a new plan chunk is built, which
gives two kinds of persistence from one mechanism: within-run (a later chunk can
draw on an earlier one, same `Session.generate()` call) and cross-run (a *new*
`Session.generate()` call can draw on a *previous* one's material, if the same
`RehearsalMemory` is passed to both — the rehearsal-informs-the-gig case). Tested
by spying on `PhraseGenerator.generate`'s actual call arguments (does `memory`'s
content reach `motif_targets`/`motif_strength`?) rather than the musical output —
a real empirical probe found the model only follows a fed-in motif rarely (2/40
trials over real inference), so a statistical pass/fail test on the musical effect
would have been unreliable; the demo section shows the qualitative effect and says
so plainly rather than implying a guaranteed audible callback.

Recall is **quality-weighted now, not pure frequency** (Phase 12,
`ensemble/critic.py`) — closing the gap `ensemble/memory.py`'s own docstring
named ("no evaluation of what's worth remembering... an actual critic... real,
separate work"). Grounded in two real sources, not first-principles guessing:
David's unrelated prior work measuring "musicality" for LSTM-generated piano
sight-reading pieces (a Colab notebook analysing real Grade 1 specimens via
**corpus-similarity**, not a hand-authored score — interval histograms, and
melodic contour reduced to a string of **U/D/S** compared by edit distance; a
striking, unplanned overlap with `gesture/recognizer.py`'s own sub-gesture
alphabet, which already contains that exact U/D/S vocabulary for an unrelated
purpose), and Wolfson's own ported bias layers, three of which are repurposed
from *generation-time sampling biases* into *retrospective scoring functions*
instead of new theory (singability's bell curve; voice-leading's chord-tone
resolution, via `chord_tones`/`chord_to_mode`/`scale_pitch_classes`,
`ensemble/wolfson/scales.py`). The corpus-similarity *method* doesn't transfer
directly — combo has no "truth set" of real jazz phrases to compare against, and
Grade 1 piano's interval norms are the wrong reference for jazz — but the
*technique* does: interval sequences, and U/D/S contour-string edit distance (a
small in-house Levenshtein function, not a new dependency for one ~15-line
algorithm). Seven metrics: `tonal_conformity`, `contour_smoothness`, `repetition`
(exact n-gram repeats, Phase 11's `extract_interval_motifs`, plus *near*-repeats
via contour edit distance — new here, catches a varied restatement exact matching
misses), `call_response_relatedness` (contour similarity between the seed and the
response — genuinely new, prompted directly by the notebook; documented honestly
as measuring *relatedness* not *goodness*, since real call-and-response sometimes
deliberately contrasts rather than mirrors), `singability`, `phrasing` (Phase 23,
below), and `register_usage` (Phase 24, below). Every one is a **pure,
deterministic function needing no model inference** — the first sax-adjacent
test file since Phase 8 that's fully testable without `sax_best.pt`. All weights
and thresholds (`DEFAULT_WEIGHTS`, the smoothness/near-repeat placeholders) are
explicit, unvalidated placeholders, same status as every other hand-picked
constant in this codebase — needing real tuning once there's a way to listen and
compare.

Deliberately still deferred, same scope-cut discipline as every earlier phase: all
~10 of the model's OTHER rule-based bias knobs (energy arc, contour, register
contrast, ...) stay at their defaults; hidden-state continuity still resets
*between* planned chunks.

**A phrasing/breath metric — "speaking in sentences"** (Phase 23). From the same
listening test as Phase 22: "the solos are not speaking in 'sentences' with gaps
in between, which is something that is taught by educators." `phrasing()` is the
only one of the seven metrics that looks at raw `notes` (including `REST_PITCH`
sentinels) rather than the `_real_notes()`-filtered view every other metric
uses — `phrase_generator.py`'s own `_inject_rests` already splices genuine rests
into generated output (bell-curve-weighted toward the phrase midpoint, capped at
`REST_MAX_PROBABILITY=0.15`, each 0.5 or 1.0 beats), but nothing had ever looked
at that structure before. Measures the fraction of a chunk's total duration spent
in a genuine breath (`MIN_BREATH_BEATS=0.5`, matching the model's own shortest
producible rest), scored via the same bell-curve treatment as `singability` —
too little breath reads as running on with no sentence structure, too much reads
as fragmented. `TARGET_BREATH_FRACTION`/`BREATH_FRACTION_WIDTH` are empirically
grounded, not guessed: 40 real 4-bar chunks had mean breath fraction 0.129,
median 0.124 — the target sits close to that natural centre, nudged slightly
higher to push selection toward a bit more breath than the unweighted default.
`DEFAULT_WEIGHTS` rebalanced (six keys at the time). **Needed no changes to
`ensemble/sax.py` at all** — the cleanest possible outcome: `sax_generator`'s
selection key already reads `musicality_score(...).overall` generically (never
enumerates `MusicalityScore`'s fields by name), so the new sub-score reaches
real selection automatically through the existing weighted-sum mechanism.
Verified this isn't vacuous with a real integration test: real candidates from
one chunk show genuine variance in `phrasing`, not a degenerate constant.
Deliberately not a separate "sentence count" measure on top of the breath
fraction — a possible future refinement, not attempted.

**A register-usage metric — "not much range... being used"** (Phase 24). From
the same listening test: "not much range of the instrument is being used. But
on reflection, this is correct for beginners." David asked specifically for
this to live in *the critic*, not just a wider `register` bound at the call
site (already possible) — mirroring Phase 23's own lesson that having a
mechanism available doesn't mean generation reliably uses it without a scored
incentive. `register_usage()` measures the fraction of the active `register`'s
width spanned by a chunk's IN-REGISTER real notes — checked directly before
designing around it: `musicality_score` scores the model's RAW candidate
output, before `_split_phrase_into_bars` clips out-of-register pitches for
playback, and 13 of 40 real chunks had at least one out-of-register note
(Wolfson's trained pitch vocabulary is MIDI 44-93, wider than any `register`
passed in) — a raw-span measurement would sometimes reward spread that never
actually sounds. Deliberately **not** a bell curve around a target the way
`phrasing`/`singability` are — reasoned through, not pattern-matched from
Phase 23: there's no clear "too much range" complaint the way there is for
breath, since `register` itself already caps what's appropriate for the
chosen skill level, and `contour_smoothness` already penalises erratic wide
leaps separately — so this is a monotonic "use more of what you're given"
reward instead. No separate beginner/advanced mode either: the skill-level
distinction lives entirely in which `register` the *caller* passes to
`sax_generator`, already a real parameter — a narrow register already caps how
much span is possible. Unlike `phrasing`, this one needed a small, real
`ensemble/sax.py` change: `musicality_score()` gained a required `register`
parameter (no sensible default), rippling through all 14 existing call sites —
expected, mechanical fallout, not a regression. `DEFAULT_WEIGHTS` rebalanced to
seven keys summing to 1.0.

**Recall is also chord-quality-aware, not just pooled globally** (Phase 25).
`RehearsalMemory.store`/`recall_motifs` take an optional `chord_quality`
(Wolfson's 4-class major/dominant/minor/diminished system), computed once per
chunk in `ensemble/sax.py` as `chord_idx % N_QUALITIES` and threaded through
both calls — recall becomes "what worked over a dominant chord", not "what
worked anywhere". Deliberately tagged by quality, not root or full `chord_idx`
— checked directly first: `extract_interval_motifs` is already
transposition-invariant, so a shape that worked over one root of a quality is
exactly as valid, transposed, over any other root of the same quality; tagging
by root too would only fragment the buffer for no musical reason. Strict
filtering, no cross-quality fallback (a quality with no history yet simply
recalls nothing) — a blended fallback is a real, separate future refinement.
A real finding worth stating plainly: `songs/blues_in_f.chart` is every chord
a dominant 7th (different roots, same quality), so this feature shows *zero*
observable difference there — the real effect is demonstrated instead over
`tests/test_sax_wolfson_integration.py`'s existing `build_ii_v_i_song()`
fixture (Dm7-G7-Cmaj7, three distinct qualities), verified by replaying the
actual stored history to confirm a later chorus's motif target came only from
same-quality phrases, never a different chord's.

Register, phrasing, and tension-and-resolution (Phases 22-24) are three pieces
of a larger "beginner vs advanced" idea from the same listening-test
discussion; side-slipping and an actual call-site register-narrowing *switch*
(vs. today's single `register` bound) remain explicitly open. A much bigger,
explicitly separate idea from the same discussion is a whole-solo
*narrative-arc* critic (intro/tension/resolution across many chunks) —
genuinely different in kind, since every metric here scores one chunk in
isolation, not a performance.

**The rehearsal effect is now real, not just wired** (Phase 17). A controlled A/B
test (`rehearsal_ab_test.py`, kept as a reusable tool rather than a throwaway
script) comparing one `RehearsalMemory` shared across `--loop` iterations against a
fresh one every loop found a genuine null result at first: no measurable
difference in `repetition`. Traced to two real causes, both fixed rather than
re-measured harder — `repetition()` checks whether a chunk repeats a pattern
*within itself*, with no reference to what memory actually recalled, so a chunk
that used the recalled motif exactly once (without repeating it again in that
same short chunk) scored `0.0` regardless; and `_apply_motif_bias`
(`phrase_generator.py`) only nudges the next token once the model has already
spontaneously started matching the target's prefix by chance, rare for a long
motif. Fixed entirely in combo-authored code, the ported model untouched: a new
`motif_adherence` metric (does a candidate's own output actually contain the
recalled target?, distinct from `repetition`'s self-similarity), a new
`_pick_achievable_motif` helper that prefers the shortest recalled motif —
grounded directly in `_apply_motif_bias`'s own prefix-matching logic, which needs
far fewer prior notes to coincide for a short motif than a long one — and
selection now uses `(motif_adherence, overall)` lexicographically (provably
identical to Phase 14's overall-only comparison when nothing's recalled). A new
`motif_recall_candidates` parameter spends extra search specifically on chunks
with a real target. Rerunning the A/B test afterward found a clean,
construction-clear effect exactly where it should live: the first plan-chunk of
every loop after loop 0 — the one place cross-loop persistence specifically
acts — scores `motif_adherence` 1.0 for the persistent condition and 0.0 for a
fresh-memory control, every time. `self_test.py` now prints a plain marker
("echoed a motif from an earlier rehearsal") whenever this fires.

**Dissonant notes are now actively avoided, not just tolerated** (Phase 18).
David's own reaction after listening for real: "even non-expert audiences can
hear when something is dissonant... the one semitone delta is as bad as it
gets in melodic playing, the dreaded minor 9th." A new `out_of_key_check.py`
(kept as a reusable tool, same lifecycle as `rehearsal_ab_test.py`) found
~16-22% of sax notes out of key, and — checked, not assumed — every single one
landed exactly 1 semitone from the scale, never further. `ensemble/critic.py`'s
new `dissonance` metric targets exactly that 1-semitone relationship
specifically (a note further from the scale isn't counted at all — landing
further out reads as deliberately "outside," David's own judgment, not a
clash); `ensemble/sax.py`'s selection now checks it FIRST, ahead of
`motif_adherence` and general quality — his framing: "we could have all sorts
of statistical measures of what's good, but what's bad matters a lot," so
dissonance is a gate candidates must clear, not one more positively-weighted
ingredient diluted into `overall`. `self_test.py`'s `n_candidates` went from 3
to 8 to give that gate real candidates to actually choose among. Result,
reproduced across two separate 5-loop runs: 2.1% out of key (31/1505, then
32/1525) — down from ~16-22%, an ~8-10x reduction, not zero (a batch can still
come up all-clashing, just increasingly rarely as `n_candidates` grows).

**A passing-tone exception, the first of three follow-on levers** (Phase 19).
Discussing that result, David raised a real nuance: plenty of legitimate jazz
vocabulary — 4ths, tritones, maj7 as a bebop passing tone over a dominant
chord — is technically dissonant by a plain scale-membership check, and
flatly penalising all of it risks blander soloing, not cleaner. Three levers
came out of that discussion: widen the per-quality scale reference
(`scales.py`'s `MODES` already defines `bebop_dom`/`altered`/`lydian_dom`,
just never wired to any chord quality); an anti-dissonance mode/strength
toggle, reusing Phase 13's `toggle_singability` director-gesture pattern; and
tolerating genuine passing tones. David asked to scope the last one first,
since it needs no new architecture — a direct refinement of `dissonance`
itself, not a new metric or a change to `ensemble/sax.py`'s selection key.
New `_is_passing_tone`: a flagged note approached AND left by step
(`PASSING_TONE_MAX_STEP`, a placeholder set to a major 2nd), continuing in
the SAME direction, is excused — the classical tonal-theory treatment of a
dissonance (David's own example: "a chromatically descending bass line is
strong in itself and justifies the one-semitone deltas"). A NEIGHBOUR tone
(approached and left in OPPOSITE directions, e.g. C-D-C) is a related,
distinct device, deliberately left uncovered, not an oversight — same as the
other two levers, both still open. `out_of_key_check.py` now breaks its
report down by passing-tone-vs-clash (reusing `_is_passing_tone` directly,
the same "verify via the same computation" precedent used everywhere else in
this codebase): the out-of-key rate that survives into final output rose to
3.2-4.2% across two runs (up from 2.1% — expected, not a regression, since
passing tones are now less penalised in selection), of which 51.5% and 62.5%
respectively were genuine passing tones, a real, repeatable majority rather
than unexplained clashes.

**The other two levers, built together** (Phase 20). Widening the scale
`dissonance` itself judges against: `dissonance_scale` (`ensemble/critic.py`)
unions the plain per-quality mode with a named jazz-standard "richer" variant
already sitting in `scales.py`'s `MODES` table but never wired to anything —
`mixolydian | bebop_dom` for dominant chords (the literal "E natural over F7"
case that started this), `ionian | bebop_major` for major chords (the b6).
Checked directly before building it, not assumed: `ensemble/sax.py` never
actually passes `scale_pitch_classes` into `PhraseGenerator.generate()` at
all, so this is purely a critic-side accuracy fix — it doesn't touch
generation-time bias or `scales.py` (the ported file). Minor/diminished have
no comparably-named richer variant and stay unwidened, a named scope-cut, not
an oversight. And an anti-dissonance toggle, reusing Phase 13's
`toggle_singability` pattern exactly: two separate rests in a row
(`gesture/vocabulary.py`'s new `("R","R")` rule) flips
`dissonance_mode["enabled"]` off/on, checked every bar. Picking the pattern
needed the same collision-avoidance care Phase 13 already established
(`"T"`/`"L"` unusable in any position of a new pattern; `("U","U")`/`("D","D")`
reserved as record-marker prefixes; `("S","S")` taken) — `("R","R")` was
verified empirically before being chosen, the same discipline that originally
caught the `("T","T")` collision: fires only on two genuinely separate rests
with nothing between them, never on ordinary playing. `dissonance()` is still
computed and logged every chunk regardless of the toggle — only whether it
drives selection is gated. `out_of_key_check.py` updated to use
`dissonance_scale` (not the plain scale) so its own report matches what
selection actually judges against: two more runs after the widened scale
landed — 4.4% (65/1485) then 1.2% (16/1374), down from 3.2-4.2% — and the
original "E natural over F7" example no longer appears at all, simply in-scale
now rather than merely excused.

**Functional context: ii-V-I simplification and tritone/b5 substitution**
(Phase 21) — the architectural gap named above, closed for two named,
teachable techniques. **Tritone/b5 substitution**: checked directly before
building it, not assumed — the first instinct (union the WHOLE scale of the
tritone-substitute dominant, mirroring Lever A) saturates the metric almost
completely: F7's own widened scale is 8 notes, its substitute B7's is
another 8, and their union is all 12 pitch classes, since a tritone is the
most harmonically distant interval and two mixolydian-family scales that far
apart share almost nothing. So `dissonance_scale` (renamed public —
`ensemble/sax.py` now calls it directly, the first time a `critic.py` helper
is needed by production code, not just tests/tooling) instead tolerates a
SINGLE extra pitch class for dominant chords — the tritone from the root —
matching exactly what David named ("a b5 substitution," a specific color
tone, not "the whole substitute chord is valid"). **ii-V-I simplification**:
new `ensemble/sax.py` functions `_ii_v_i_target`/`_functional_tonic_scale`
check whether the current bar could be the ii, V, or I of a textbook major
ii-V-I (root motion by descending fifths, qualities minor/dominant/major),
using `Song.chord_at`'s cyclic lookup (verified directly to never raise, even
near a chart's boundary) to look 1-2 bars ahead/behind — if matched, the
target I chord's own `dissonance_scale` is unioned in via a new
`extra_tolerated` parameter on `dissonance()` (default empty, reproducing
Phase 20 exactly for every existing call site). Verified numerically before
trusting it composes safely with Lever A/D: D-dorian (the ii of a C ii-V-I)
and C-major-widened differ by exactly one pitch class (the b6) — diatonically
related scales overlap almost entirely, unlike the tritone case, which is
why this lever can safely union a WHOLE scale while tritone-sub cannot. A
real selection-behaviour test (spy-and-recompute) confirms the extra
tolerance actually reaches `sax_generator`'s selection over a genuine
Dm7-G7-Cmaj7 chart. Explicitly deferred: a vi-ii-V-I (four-chord) extension,
a minor-tonic ii-V-i variant, and sub-bar-granular chord sequences (more than
one chord change per bar).

**Tension-and-resolution crediting** (Phase 22) — prompted directly by a
listening-test question: "I can hear the difference between conscious use of
discordant intervals and use due to getting lost, panic, or playing randomly.
I wonder how we could encode that." Generalises the passing-tone exception
(Phase 19 — moves THROUGH a dissonance between two flanking pitches) to a
second, distinct device: a clash approached from an in-scale note (a single
isolated reach outward, not mid-excursion) and resolved by step onto an
actual chord tone (`ensemble/critic.py`'s new `_is_resolved_tension`) — a b9
resolving down a half-step to the root, say. Unlike every earlier lever, this
one is opt-in: `dissonance()` gains `credit_resolved_tension: bool = False`,
threaded through as `sax_generator`'s own parameter of the same name and
`self_test.py`/`out_of_key_check.py`'s new `--credit-resolved-tension` flag —
default off (a "beginner" default), since unlike a passing tone this isn't
universally uncontroversial, it's the "advanced" behaviour itself. Verified
on real output, not assumed: without the flag, only 4.1% (2/49) of what
survives selection happens to already look like a resolved tension by
accident; with it, two separate 5-loop runs, 28.3% (15/53) then 52.4%
(22/42) of what's flagged are genuine resolved tensions — real and
repeatable, not a one-off (`out_of_key_check.py`'s own docstring has the
full numbers). Explicitly narrower than it might sound: covers only a
SINGLE isolated tension-then-resolution note, not a multi-note excursion
(genuine side-slipping needs real generation-time mechanics, not a scoring
exemption); doesn't reward tension use, only stops penalising it once
resolved; not wired to a live director gesture this phase (every practical
same-symbol gesture pattern is already claimed). Two more adjacent ideas
from the same discussion, deliberately not attempted here: register range as
a skill-level control (a call-site choice, not a code change — `register` is
already a real parameter) and phrasing/"speaking in sentences" with gaps
(a different critic dimension entirely).

**A director can now toggle the critic live** (Phase 13, DESIGN.md §11) — the
first real consumer of `DirectorSignal.gesture` since the dial channel was built
in Phase 5 (every phase since had repeated some version of "a director-emitted
gesture has nowhere to act"). A new seed gesture, two same-note-repeat runs in a
row (`gesture/vocabulary.py`'s `("S","S")` — not `("T","T")`, which was checked
and confirmed would have collided with the existing single-`"T"` `reset_tempo()`
rule and never actually fired: rules are matched in list order and a 1-length
pattern always wins as soon as its own tail element arrives), flips
`sax_generator`'s mutable `critic_weights["singability"]` between the default
and `0.0` — letting a director or teacher turn the metric off live for a
student's fast, exploratory playing that shouldn't be marked down for being
unsustained, without touching the other four metrics. `musicality_score` itself
just grew an optional `weights` parameter to make this possible (every sub-score
is still computed and reported regardless of what counts toward `overall`).

**Sax can now search, not just generate once** (Phase 14, DESIGN.md §13's
long-deferred "chess" idea — "generate a few bars, evaluate, branch/backtrack").
`sax_generator`'s new `n_candidates` parameter generates that many candidates
per chunk (identical arguments each call — the model's own RNG state naturally
diversifies successive calls, no extra diversity logic needed) and keeps the
highest-scoring one by `musicality_score(...).overall` — the evaluator §13
always said this idea needed, now that Phase 12 built one. §13's "poor fit for
live performance" reasoning was never actually checked against real numbers;
done directly once there was something to measure: even 20 candidates over a
full 4-bar chunk cost ~164ms against a 7.3-second real-time budget at blues
tempo (132bpm) — corrected, not inherited unquestioned, though honestly still
unverified for true live call-and-response specifically (that measures
computation against `Session`'s pacing budget, not human-perceived
conversational latency, a different question). At `n_candidates=1` (the
default) this is exactly the old behaviour: one `generate()` call, one score —
checked directly, byte-identical output either way. A real simplification fell
out of building this: `musicality_score` used to be computed only
`if memory is not None:`; now it's always computed once (selection needs it
regardless of whether memory is configured), and `memory.store()` reuses that
same computation rather than re-deriving it. Verified with a real,
independently-recomputed proof, not sax_generator's own bookkeeping taken on
faith: a local spy captured every candidate's actual notes, recomputed each
one's score from scratch, and confirmed both that the recomputed scores match
`generate.last_candidate_scores` exactly and that the dispensed notes came from
the genuinely highest-scoring candidate. Deliberately not attempted: varying
anything besides the random draw across candidates (temperature,
`rhythmic_density` — searching generation *parameters*, a larger idea than
resampling fixed ones); revision after committing (ImproteK's actual
architecture, §12); a director-gesture-driven `n_candidates` toggle (the
natural next use of Phase 13's `critic_weights`-mutation pattern, not built).

This is also the **first piece needing a binary artifact
not present in a fresh clone** — the trained weights
(`ensemble/wolfson/models/sax_best.pt`) are gitignored, not committed (see Running,
below), so its integration tests skip
gracefully and its demo section degrades gracefully without them, and the test
suite is no longer sub-second once torch is imported.

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
  `constant_director_source`, `ensemble_intensity_critic`), `transitions.py`
  (DESIGN.md §8 — `TransitionController`, `GestureSource`, `scripted_gesture_source`,
  `LiveGestureQueue`), `sax.py` (DESIGN.md §12 — real generation for the sax voice:
  `chord_to_wolfson_index`, `sax_generator`, `_bars_until_chord_change` +
  `_split_phrase_into_bars` for the multi-bar planning buffer, `_pick_achievable_motif`
  for choosing a recalled motif that's actually reachable (Phase 17), `_ii_v_i_target` +
  `_functional_tonic_scale` for the ii-V-I "simplify to the tonic" technique
  (Phase 21)), `memory.py`
  (DESIGN.md §12 — `RehearsalMemory`, the first state that persists across separate
  `Session.generate()` calls), `critic.py` (DESIGN.md §11/§12 — a musicality
  critic: `tonal_conformity`, `contour_smoothness`, `repetition`, `motif_adherence`
  (Phase 17 — distinct from `repetition`, measures adherence to an externally
  recalled target rather than self-similarity), `dissonance` (Phase 18 — higher is
  WORSE, unlike every other function here; a badness signal for selection to
  minimise, not a goodness signal blended into `overall`; excuses genuine chromatic
  passing tones since Phase 19, `_is_passing_tone`; judges against a widened,
  jazz-aware scale since Phase 20/21, `dissonance_scale` — public since Phase 21,
  `ensemble/sax.py` calls it directly), `call_response_relatedness`,
  `singability`, `musicality_score`; every function pure and deterministic, no
  model inference), `roles.py` (DESIGN.md §2 — the
  same-instrument-doubling slice of role assignment: `default_accompanist_roles`,
  a greedy register-overlap rule; consumed by `comping_generator`'s `lay_out`
  parameter).
- `output/midi_output.py` — the playback stage (DESIGN.md §4): `list_output_ports`,
  `build_schedule` (pure: `Timeline` + tempo + channel map -> a time-sorted MIDI
  schedule), `play_timeline` (real-time playback, reusing `ensemble.session`'s
  `Clock`/`FakeClock`, all-notes-off in a `finally` block regardless of how
  playback ends)
- `ensemble/wolfson/` — ported generative core from wolfson (DESIGN.md §12):
  `lstm_model.py` (`PhraseModel`), `phrase_generator.py` (`PhraseGenerator`),
  `encoding.py`, `chords.py`, `scales.py`, `motifs.py` (`extract_interval_motifs`,
  used by `ensemble/memory.py`); provenance and exactly what changed mechanically
  vs. the source is documented in `__init__.py`. `models/` is gitignored — copy
  `sax_best.pt` in manually (see Running, below).
- `tests/test_recognizer.py`, `tests/test_gesture_vocabulary.py`, `tests/test_song.py`,
  `tests/test_session.py`, `tests/test_drums.py`, `tests/test_listening.py`,
  `tests/test_comping.py`, `tests/test_director.py`, `tests/test_midi_sources.py`,
  `tests/test_transitions.py`, `tests/test_sax.py`, `tests/test_memory.py`,
  `tests/test_critic.py`, `tests/test_roles.py`, `tests/test_midi_output.py` — no
  MIDI hardware needed
- `tests/test_sax_wolfson_integration.py` — needs the real `sax_best.pt` weights;
  skips cleanly if they're not present (see Running, below)
- `listen.py` — small runnable script: starts every source in `config.MIDI_SOURCES`,
  prints recognised gestures (performers) and live intensity (directors)
- `gesture/demo.py` — small runnable script: replays synthetic gesture sequences
  (seed gestures, record + alias teaching) and prints what's recognised
  (`python -m gesture.demo`)
- `ensemble/demo.py` — small runnable script: generates a chart's worth of stub sax +
  drums output and prints it, then separate demonstrations of comping's duck/fill
  behaviour, the role-split between two comping voices, the director's intensity dial
  (low vs. high, plus the AI critic reading the sax+drums session), a scripted
  handover shifting section boundaries, and (if `sax_best.pt` is present) real
  generation for the sax voice (`python -m ensemble.demo`)
- `self_test.py` — small runnable script: builds the full AI-only ensemble over a
  chart and plays it through a real MIDI output port — see the three-step testing
  plan below (`python self_test.py`)
- `rehearsal_ab_test.py` — small runnable analysis script (no MIDI/audio): a
  controlled A/B comparison of `RehearsalMemory` shared across `--loop` iterations
  vs. a fresh one every loop, measuring `motif_adherence` and `repetition`
  directly rather than by ear — see Phase 17's rehearsal-memory paragraph above
  for what it found (`python rehearsal_ab_test.py`)
- `out_of_key_check.py` — small runnable analysis script (no MIDI/audio): counts
  how many of the sax's generated notes actually land out of key against the
  active chord's scale, at `self_test.py`'s real settings — see Phase 18's
  dissonance-avoidance paragraph above for what it found
  (`python out_of_key_check.py`)

## Running

```
pip install -r requirements.txt   # now pulls in torch — heavier/slower than before
python listen.py --list          # show available MIDI ports
python listen.py                 # start every source in config.MIDI_SOURCES
python -m gesture.demo           # no MIDI needed
python -m ensemble.demo          # no MIDI needed; sax section needs sax_best.pt (below)
python self_test.py --list-out   # show available MIDI output ports
python self_test.py              # play blues_in_f.chart once, no MIDI input needed
pytest
```

### Trying it out: a three-step testing plan

**1. Today, no MIDI input needed** — `python self_test.py` builds a full AI-only
ensemble (bass stand-in, sax, two role-split comping voices, drums), generates it
once, and plays it through a real MIDI output port to a synth (e.g. macOS's IAC
Driver into GarageBand or a simple synth — `python self_test.py --list-out` to find
the right port index, `--out N` to select it). This is combo's own version of
wolfson's AI-vs-AI self-play, generalised from one bass+sax pair to the whole
ensemble. `--loop 3` plays the chart three times, sharing one `RehearsalMemory`
across the loop — David's rehearsal idea (DESIGN.md, Phase 11). As of Phase 17
this is a reliable, measured effect, not just a hopeful one: watch the console
for `-> sax echoed a motif from an earlier rehearsal`, printed whenever a loop's
first phrase actually reuses something from an earlier one.

**2. Tomorrow, MIDI input hardware, recognition only** — `python listen.py` (already
built, no new code) prints recognised gestures and live intensity as you play, a
real sanity check that the device/gesture recognition pipeline works. It doesn't yet
feed anything into a running ensemble — that's step 3.

**3. Not yet built, named honestly rather than glossed over** — true live rehearsal
(your live playing actually driving the ensemble's real-time response, the way
wolfson's live bass+AI-sax mode worked) needs a "live performance driver" connecting
`input/sources.py`'s live streams into a running `Session.generate(mode=REAL_TIME)`.
Nothing in this codebase does that yet — `listen.py` only prints what it recognises.
A separate phase, to be scoped once steps 1–2 are proven.

Real generation for the sax voice (DESIGN.md §12) needs the trained model weights,
which aren't committed to this repo (gitignored — see `.gitignore`). Copy them in
manually:

```
cp ~/wolfson/models/sax_best.pt ensemble/wolfson/models/sax_best.pt
```

Without them, `pytest` skips `tests/test_sax_wolfson_integration.py` and
`ensemble/demo.py`'s sax section prints a message and skips — everything else runs
unaffected.
