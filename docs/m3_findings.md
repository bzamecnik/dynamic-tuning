# M3 findings: dissonance of fixed tunings

Reproduce with `uv run python scripts/m3_dissonance.py` (plots and full tables in `outputs/m3/`).
Setup of the first run below: 12-key progression (cycle of fourths, 4 voices), 8 harmonic
partials with amplitudes 1/k, Sethares 1993 model, inter-note roughness only (pairs of partials within
one note don't depend on the tuning), duration-weighted means.

## Observations

- **Tuning differences are small compared to chord/voicing differences.** Across the
  progression, roughness varies about 2.5× between chords (0.11–0.33), but the tunings
  differ by only a few percent (mean over the piece: 0.217–0.224). Most of the roughness
  comes from pairs of partials that are moderately far apart (e.g. low-register partials
  inside one critical band). Those pairs hardly change with a few cents of retuning.
- **Coinciding partials that are slightly mistuned contribute little.** In the Plomp–Levelt/
  Sethares curve, roughness peaks at a beat rate of about 20–50 Hz and is close to zero for
  slow beats. So the 12-TET major third C4–E4 (its 5th/4th partials beat at ~10 Hz) scores
  about the same as the pure 5/4 third with 1/k amplitudes. With brighter timbres
  (rolloff 0.5) or the Vassilakis model, JI is clearly smoother. The plan's sanity check
  "pure intervals are less rough than 12-TET" holds only for some timbre and model settings.
- **Key dependence matches music theory** (`key_dependence.png`):
  - JI fixed in C is best near C (−2 % vs 12-TET) and worst in E/A (+3 %).
  - 1/4-comma meantone is good in flat keys near C and bad in Ab–B (+3–5 %), where the
    wolf fifth and the diminished fourths sit.
  - Kirnberger III and Werckmeister III stay within ±1 % of 12-TET.
  - JI retuned per key is the smoothest overall (−1 to −3 % in every key).
- **By chord type** (`progression_by_figure.png`): key-relative JI makes I, vi, V7, IM7
  and V7/IV 3–5 % smoother, but ii7 and viiø7 *rougher* (+2 %). That is the syntonic comma:
  in 5-limit JI, ii7 contains the narrow fifth D–A (680 cents). The dynamic optimizer
  should fix exactly this.
- **Sensitivity** (`sensitivity.png`): the ranking of tunings is fairly stable, except
  with only 4 partials (all tunings ≈ 12-TET or slightly worse). Vassilakis gives
  differences about twice as large as Sethares (key-relative JI ≈ −5 %).

## Implications for M4/M5

- An optimizer that minimizes total Sethares roughness gets most of its gradient from
  tuning-independent pairs. It may prefer to spread partials apart rather than to make
  intervals pure, and the purity of coinciding partials gives only a weak signal. Options:
  use Vassilakis, use brighter timbres, or add a term targeting slowly beating near-unison
  partial pairs (beating as distinct from roughness).
- Report results relative to 12-TET (per segment, as in the lower profile plot). Absolute
  values are dominated by voicing and register.

## BWV 846 (WTC I, C major prelude, from the music21 corpus)

- Estimated key sections: C → G → a → G → g → d → C → F → f → G (pedal) → C.
- Same picture as for the progression: tunings differ by a few percent per segment.
  The largest effect is fixed JI (C) in the A minor passage (up to +17 % vs 12-TET),
  which comes from the syntonic comma (D–A). Key-relative JI is the smoothest overall.
- The corpus edition writes the sustain out only for the lower voices (upper arpeggio
  notes are 16ths), so roughness drops to zero between some chord tones.

## Model comparison (added later)

Three pair models are available (`dynamic_tuning.dissonance.MODELS`): Sethares 1993,
Vassilakis 2001 (now the default) and Hutchinson & Knopoff 1978 (normalized by the sum of
squared amplitudes, as in Harrison & Pearce's `dycon`). See `dissonance_curves.png`:

- `dissonant` 0.1.1 computes Vassilakis' amplitude-fluctuation degree from the amplitude
  of the upper partial instead of `A_min` (paper: `Y = 2 A_min / (A_min + A_max)`). Our
  implementation follows the paper. It changes the results noticeably, since the upper
  partial is often the louder one (e.g. the fundamental of an upper note against a high
  partial of a lower note).
- Vassilakis has the sharpest minima at just ratios. Its tuning ranking is stable across
  timbres: key-relative JI is −4.5 % vs 12-TET for 8–16 partials and any rolloff. It also
  rates ii7 in key-relative JI smoother than 12-TET, unlike Sethares; viiø7 stays rougher.
- Hutchinson & Knopoff has a wider, smoother curve. Key-relative JI is −2 to −4 %, and fixed
  JI (C) and Pythagorean come out rougher than 12-TET on average.
- With 4 partials, Sethares and Vassilakis see almost no benefit from any tuning.
- None of the models penalizes slowly beating, nearly coinciding partials much. That would
  need an extra "beating" term (a possible custom model).
