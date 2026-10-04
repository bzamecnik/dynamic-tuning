# Plan

Goal: a practical, offline Python toolkit to **render symbolic multi-voice music with arbitrary tuning and timbre, measure its dissonance over time, and then optimize intonation/partials to control that dissonance**.

Not a research paper. Prior art exists (see [literature.md](literature.md): Sethares adaptive tunings, Villegas & Cohen, Kirsch, Schwär & Müller). We reuse known ideas and want working software that we can listen to and plot.

Scope for now: symbolic input (generated scores + MIDI) → additive synthesis → WAV. No waveform analysis, no real time, no VST/DAW integration.

## Guiding decisions

1. **Own the synthesizer.** Sample-based or SoundFont synths fix the partials, so they can't do dynamic timbre. Additive synthesis in numpy is about 50 lines and gives full control of every partial's frequency and amplitude. We don't need an external synth package, and nothing touches an audio device.
2. **One central representation: the resolved spectrum.** Everything (tuning, timbre, optimizer) just produces, per note and time segment, arrays `freqs[k]` and `amps[k]`. The synthesizer and the dissonance meter both consume only that. Because of this, fixed tunings, hand-made adjustments and optimizers all plug in the same way.
3. **Score is in Python, MIDI is for interchange.** See "Symbolic representation".
4. **Dissonance is measured on partials, not audio** (Sethares / Plomp–Levelt / Vassilakis on `(freq, amp)` pairs). It is cheap and differentiable. Later we can sanity-check it against an STFT of the rendered audio, but this isn't a goal.
5. **Optimizer is built on autodiff**, so the dissonance model has to be re-implemented in PyTorch. The `dissonant` package is numpy-only: we use it directly for non-optimization tasks (e.g. the dissonance profile of a whole score) and as the reference to test the port against.

## Symbolic representation

Recommendation: a small Python data model, with MIDI import/export. No new text format for now.

```
Note:    start, duration (in beats or seconds), pitch (MIDI int), voice (int), velocity
Section: start, end, tonic (pitch class), mode   # key annotation, needed for per-key just intonation
Score:   notes, sections, tempo
```

- **Why not MIDI only:** MIDI has no spelling (G♯ vs A♭) and no clear notion of the current key. Key-relative tunings (just intonation, meantone) need the tonic. Spelling matters only for meantone-style tunings with split accidentals. For a 12-note tuning we pick one fixed spelling (e.g. E♭ B♭ F C G D A E B F♯ C♯ G♯ for 1/4-comma meantone), so MIDI pitch numbers are enough. The tonic goes into `Section`.
- **MIDI I/O** via `mido` or `pretty_midi`. Key and section info is stored as MIDI marker/key-signature meta events. We can import `midi/well-tempered-clavier-i_bwv-846.mid` later as a real-music test, for which the key would be annotated by hand or detected from the MIDI.
- **Text format:** the progression generator is code, so a text format isn't needed yet. If we later want hand-written examples, we can add a minimal chord-per-line syntax such as `Cmaj7 | Am7 | Dm7 G7 | C`, which expands to voiced notes. This is not part of the first milestone.

## Test material: the 12-key cycling progression

One pattern, transposed to each of the 12 keys, where the end of each key's pattern pivots into the next key.

Cycle by **fourths** (C → F → B♭ → … → G → C). All 12 keys are visited and the loop closes. The I7 chord at the end of key K is the V7 of the next key (K + perfect fourth), so the modulation is smooth.

Pattern for key K, one chord per bar or two:

| Chord | Type covered |
|---|---|
| I | major triad |
| vi | minor triad |
| ii7 | minor seventh |
| V7 | dominant seventh |
| Imaj7 | major seventh |
| vii°ø / iii7 (one of them) | half-diminished or minor 7th; optionally dim7 via the V7♭9 of vi |
| I7 | dominant seventh, which becomes V7 of the next key |

Possible extensions (a second, more dissonant pattern): sus4, add9, augmented, minor-major 7.

Voicing: 4 voices (SATB-like, 5 for 9th chords), chosen by an automatic **minimal-motion voice-leading** search, within fixed voice ranges, so that voices move smoothly and the 12-key loop stays in one register. Voicing is generated once and saved as MIDI. This is deterministic, so every tuning renders identical notes.

The progression is a regression fixture: all later results are compared on it.

## Libraries

| Need | Choice | Note |
|---|---|---|
| Arrays, synthesis | numpy, scipy | additive synthesis, ADSR envelopes |
| WAV output | `soundfile` (or `scipy.io.wavfile`) | purely offline |
| MIDI I/O | `mido` or `pretty_midi` | no audio backend needed |
| Plots | matplotlib | |
| Autodiff | PyTorch (CPU) | decided; JAX would be equivalent but PyTorch code is a bit easier to read |
| Dissonance | `dissonant` (own package, PyPI) | used directly for non-optimization tasks (e.g. profile of a whole score); reimplemented in PyTorch for the loss |
| Optional | `pedalboard` for reverb on final renders | |

Alternatives for synthesis (Csound NRT, SuperCollider NRT, FluidSynth with MTS) work offline too, but add a second toolchain and don't give per-partial control that is as easy as numpy. Revisit only when real-time is on the table.

**Environment (decided):** a `uv` project with `pyproject.toml`, on a recent Python supported by PyTorch (3.12 or 3.13). PyTorch is CPU-only (about 200 MB), as no GPU is needed.

## Repository layout (proposed)

```
pyproject.toml
src/dynamic_tuning/
    score.py          # Note, Section, Score, MIDI I/O
    progression.py    # chord vocabulary, key cycle, voice-leading generator
    tuning.py         # fixed tunings: cents tables, key-relative lookup
    spectrum.py       # timbre models + ResolvedSpectrum (per note, per segment)
    synth.py          # additive synthesis, glide/phase-continuous rendering
    dissonance.py     # roughness models (numpy), later torch port
    analysis.py       # segmentation into slices, profile over time, plots
    optimize.py       # parametrization + loss + optimizer
scripts/            # one script per milestone producing outputs/
notebooks/          # optional exploration
tests/
outputs/            # gitignored: wav, png
```

## Milestones

Each milestone ends with something to look at or listen to.

### M0: Project setup
- `uv` project, dependencies, `pytest`, `.gitignore` entries for `outputs/`.
- Add `dissonant` as a dependency; inspect its API.

### M1: Score + progression generator
- `Score` model, MIDI export/import round trip.
- Chord vocabulary, key cycle by fourths, voice-leading search.
- **Deliverable:** `progression.mid` and a printed chord table; load it in any MIDI player to sanity check. Test: round trip, all 12 keys present, voice motion bounded.

### M2: Fixed tunings + additive synth → WAV
- Tunings as cents offsets from 12-TET per pitch class, relative to a reference/tonic:
  - 12-TET
  - 5-limit just intonation, **fixed** key (e.g. C) and **retuned per section** to the section tonic (this is the "static vs key-following" distinction)
  - 1/4-comma meantone
  - Kirnberger III
  - (cheap to add: Pythagorean, Werckmeister III)
- Timbre: harmonic partials `k·f0`, amplitudes `1/k^α`, ADSR envelope. Parametrize `n_partials` and `α`.
- Renderer takes a Score + Tuning + Timbre and writes WAV (mono first, optional stereo spread by voice).
- **Deliverable:** one WAV per tuning for the same progression. Test: frequency of each note matches the table (FFT peak check).

### M3: Dissonance measurement + visualization
- Slice the score at every note onset/offset into segments of constant sounding notes.
- For each segment, resolve spectrum → total roughness (Sethares/Plomp–Levelt first, Vassilakis next, optionally normalized per partial pair count and amplitude).
- Plots: dissonance vs time as a step curve with chord labels and key boundaries; per-pair contribution heatmap for a chosen chord; comparison across tunings overlaid; summary table (mean dissonance per chord type per tuning).
- **Deliverable:** the key observations — e.g. does fixed-key JI win in C and lose far from C; does 12-TET give a flat profile; how do meantone and Kirnberger vary with key. Also, which chord types are the most rough in any tuning.
- Caveat to check: roughness depends on register and on the amplitude model, so we want to see plots with varying `n_partials`/rolloff to know how sensitive the conclusions are.

### M4: Hand-controlled spectrum adjustment
- Make the resolved spectrum editable: per-note cents offsets, per-partial frequency ratio stretch (e.g. `f_k = f0·k^(1+s)` inharmonicity, or `f_k = f0·k·(1+ε_k)`), amplitude changes, optional muting of partials.
- Time-varying adjustments need phase-continuous synthesis (integrate instantaneous frequency) and short glides across segment boundaries.
- Try simple rules: e.g. stretch timbre toward a different scale (Sethares' idea: timbre matched to the tuning), or nudge each chord's notes toward its just ratios.
- **Deliverable:** renders and dissonance profiles for a few hand-made adjustments. This also validates the parametrization the optimizer will use.

### M5: Differentiable dissonance + optimization
- Port the roughness model to torch/jax; test against numpy reference (values and finite-difference gradients).
- Parameters per segment: note cents offsets `δ_n`, optionally partial offsets `ε_{n,k}` (frequency), later amplitudes.
- Loss (following [initial_vision.md](initial_vision.md)):
  `L = w_d · L_diss + w_r · L_reg + w_s · L_smooth`
  - `L_diss`: sum of roughness over segments (duration-weighted), or `(D_t − D*_t)²` for a **target dissonance** curve, which is the controllable part.
  - `L_reg`: distance of intonation to a base tuning (cents²) and of partials to their base ratios.
  - `L_smooth`: penalty on change of a note's parameters between adjacent segments, so a sustained note glides rather than jumps. This means parameters are indexed by **(note, segment)** and a note keeps its identity across the segments in which it sounds.
- Hard bounds, e.g. intonation ±30 cents, partials ±3%, via a tanh/clip parametrization.
- Known difficulties to watch: roughness is non-convex with many local minima, so initialize from a known tuning (12-TET and JI) and try multiple starts; unconstrained minimization collapses partials into unisons or drifts the pitch; long-term pitch drift (comma drift) needs an anchoring term.
- Baselines to compare against: per-chord brute-force/grid search on cents offsets only; a simple iterative Sethares-style adaptive-tuning step.
- Stages: (a) intonation only, (b) partials only, (c) both. Compare dissonance profiles and listen.
- **Deliverable:** optimized renders with before/after dissonance plots across the 12-key progression.

### M6: Controlling the dissonance (stretch)
- Target dissonance as a user parameter: global level (0 = minimum, 1 = base tuning), per chord class (keep dominant 7ths a bit tense), or an explicit curve over time.
- Regularization sweeps: trade-off curve of dissonance vs deviation from base tuning.
- Apply to real music: the Bach WTC prelude from `midi/`.
  - Download from https://www.kunstderfuge.com/-/midi.asp?file=bach/sankey/well-tempered-clavier-i_bwv-846_(c)sankey.mid

## Evaluation

- **Objective:** dissonance profile over time per tuning, plus deviation from base (cents) and smoothness (max cents/s).
- **Subjective:** A/B WAVs of the same progression. Keep notes of what is audible; roughness models are only proxies for perception.
- **Sanity checks:** a pure octave/fifth/major third in JI must come out less rough than in 12-TET with harmonic timbre. A sine timbre (no overtones) should make all tunings equal in roughness. Sethares' classic curve for two harmonic tones should show minima at simple ratios.

## Risks and open questions

- Which dissonance model is the right default, and how robust are the conclusions to the choice of model and amplitude weighting (Sethares vs Vassilakis, with or without a loudness model)?
- Dissonance is not the whole story for perception. Harmonic fusion / virtual pitch (partials matching a common fundamental) is not captured by roughness, so a fully optimized timbre may sound consonant but strange. We may want a regularizer on timbre.
- Gradient-based optimization for several hundred segments with up to ~5 notes × 16 partials is small (≈6k pair terms per segment), so performance should not be a problem, but we should confirm with torch on CPU.
