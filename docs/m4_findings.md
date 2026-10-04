# M4 findings: hand-made spectrum adjustments

Reproduce with `uv run python scripts/m4_adjust.py` (outputs in `outputs/m4/`: WAVs,
profiles, tables, intonation trajectories). Setup: 8 harmonic partials, 1/k amplitudes,
Vassilakis model, roughness including pairs within a note (some variants change the timbre).

Parametrization (`dynamic_tuning.adjust.SpectrumParams`), aligned with the segments of a
resolved spectrum: cents offset per (note, segment), cents offset per (note, segment,
partial), amplitude gain per (note, segment, partial). Rules produce these parameters, and the
M5 optimizer uses the same structure. Changes between segments are rendered with glides by
the phase-continuous synthesizer.

Mean roughness relative to 12-TET:

| variant | progression | BWV 846 |
|---|---|---|
| Kirnberger III | −0.2 % | −0.7 % |
| JI (key) | −4.6 % | −3.7 % |
| 12-TET, partials moved to the 12-TET grid | **−15.5 %** | **−7.7 %** |
| 12-TET, stretched partials (k^1.005) | −0.2 % | +0.3 % |
| adaptive JI per chord, relative to the bass | −6.2 % | −3.9 % |
| adaptive JI per chord, best reference note | −7.3 % | −4.3 % |

Observations:

- Sethares' "timbre matched to the scale" has the largest effect. When partials are moved
  to the 12-TET grid (3rd harmonic −2 cents, 5th +14, 7th +31), all 12-TET intervals
  get exactly coinciding partials, in any key and with no retuning.
- Adaptive JI per chord beats per-key JI. The bass-relative variant needed the minor
  seventh as 9/5 (not 16/9), otherwise ii7 contains the wolf fifth 40/27 between the third
  and the seventh. The JI tables now use 9/5.
- Adaptive JI keeps the reference note at its 12-TET pitch (no drift). But held notes jump
  by up to ~30 cents between chords (see the intonation plots): the comma shifts.
  The smoothness term of M5 should trade this off against roughness.
