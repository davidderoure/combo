"""Ported generative core from Wolfson (David's earlier LSTM-based jazz
improviser, private repo at ~/wolfson), adapted for combo's sax voice — DESIGN.md §12.

What's ported (unmodified logic, from wolfson's generator/lstm_model.py,
generator/phrase_generator.py, data/encoding.py, data/chords.py, data/scales.py):
the trained model architecture, its token vocabulary, and its full inference-time
bias-layer pipeline — none of that algorithmic content has been changed.

Also ported (Phase 11, DESIGN.md §12): motifs.py's extract_interval_motifs, from
wolfson's input/phrase_analyzer.py — a small, self-contained pure function (2/3/4-
note interval n-grams from a phrase) used by ensemble/memory.py's RehearsalMemory.
wolfson's own memory/phrase_memory.py (PhraseMemory) was NOT ported — its reset
policy (between 5-minute ArcController loops, within one live performance) doesn't
match what cross-performance rehearsal memory needs (persist across separate
Session.generate() calls, reset only when a caller starts a new RehearsalMemory);
ensemble/memory.py is combo-authored, inspired by PhraseMemory's store/recall shape
but not a port of it.

What's mechanically different from the wolfson source, purely to make these files
work as a package inside combo rather than as scripts run from wolfson's own repo
root:
  - Each `sys.path.insert(0, ...)` + `from data.X import ...` / `from generator.X
    import ...` has been replaced with a plain relative import (`from .encoding
    import ...` etc.) — the sys.path mutation would otherwise leak a path onto
    the whole process's sys.path, which combo never wants.
  - Six constants that wolfson's own config.py defined (LSTM_HIDDEN_SIZE=256,
    LSTM_NUM_LAYERS=2, MAX_GENERATED_NOTES=16, GENERATION_TEMPERATURE=0.9,
    DEFAULT_INSTRUMENT="sax", REST_PITCH=-1) are inlined directly where they're
    used instead of imported from a config module — combo has its own config.py
    for MIDI sources and importing wolfson's would collide. The two LSTM
    constants are architecture-defining and must exactly match the shapes baked
    into sax_best.pt's state_dict, so hardcoding them here is more correct than
    importing a value that could drift, not a compromise.

What's explicitly NOT ported: wolfson's ArcController/HarmonyController and
everything that fed phrase_generator.generate()'s ~12 bias-layer parameters
musically rich values in live performance — combo's ensemble/sax.py calls
generate() with their defaults instead (DESIGN.md §12, "deliberately dumb" in
the same spirit as Phase 1's chord_tone_generator).

The trained weights (models/sax_best.pt) are NOT committed to this public repo —
gitignored, copy in manually from ~/wolfson/models/sax_best.pt. See README.
"""
