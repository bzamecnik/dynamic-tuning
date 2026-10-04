# M5 findings: differentiable roughness and optimization

Reproduce with `uv run python scripts/m5_optimize.py` (outputs in `outputs/m5/`: WAVs,
profiles, intonation trajectories, partial offsets, loss curves, `summary.md`).

## Implementation

- The pair models in `dissonance.py` are written against an array namespace, so torch
  uses exactly the same formulas as numpy. Tests check torch against numpy for all models
  and check gradients against finite differences.
- `optimize.SpectrumBatch` pads a resolved spectrum to tensors (segments × notes × partials).
  Parameters are the M4 `SpectrumParams`: cents per (note, segment) bounded to ±30, and
  cents per partial bounded to ±50, both via tanh. The fundamental partial is fixed.
- Loss = roughness relative to the base (duration-weighted), or the squared deviation from
  a target ratio; + `w_reg` · deviation from the base (duration-weighted); + `w_smooth` ·
  changes of a held note between adjacent segments. Adam, 400 steps, float64, CPU:
  2–15 s per run.
- Baselines: per-chord grid search (lowest note fixed, ±25 cents in 5-cent steps,
  coordinate-wise for large chords), and independent per-segment optimization
  (`w_smooth = 0`, like Sethares' adaptive tuning). Multi-start from the base and from
  adaptive JI.

## Results (Vassilakis, 8 partials 1/k, w_reg 0.02, w_smooth 0.05; incl. intra-note pairs)

| variant | progression | max held-note jump | BWV 846 | max held-note jump |
|---|---|---|---|---|
| Kirnberger III | −0.2 % | – | −0.8 % | – |
| adaptive JI (M4 rule) | −7.3 % | 15.6 c | −4.4 % | 27.4 c |
| grid search per chord | −5.9 % | 15.0 c | −4.5 % | 45.0 c |
| opt. intonation, independent | −7.3 % | 9.7 c | −5.1 % | 28.6 c |
| **opt. intonation (smooth)** | −7.1 % | **1.9 c** | −5.0 % | **3.4 c** |
| opt. partials only | −19.2 % | – | −15.9 % | – |
| opt. intonation + partials | **−19.6 %** | 0.9 c | **−17.1 %** | 4.7 c |
| Kirnberger III + opt. intonation | −6.8 % | 2.2 c | −4.8 % | 5.2 c |

- **Smoothness is almost free.** Optimized intonation with the smoothness term reaches the
  same roughness as the rule-based adaptive JI or the independent optimization, while held
  notes move at most 2–4 cents instead of 15–45 cents. Avoiding comma shifts costs only
  0.1–0.2 % roughness.
- **Partials matter more than intonation.** Moving partials (±13–15 cents RMS) reduces
  roughness about 2.5× more than retuning notes. The optimizer partly rediscovers the M4
  "partials on the 12-TET grid" rule (5th harmonic up, 7th harmonic up), but it adapts
  per chord and gets −19 % instead of −15 %.
- Starting from Kirnberger III instead of 12-TET leads to about the same result. The base
  tuning matters little once intonation is free within ±30 cents.
- The Adam runs converge in about 100 steps (`*_history.png`). Multi-start from adaptive JI
  gave the same optimum as starting from the base, so for intonation local minima don't
  seem to be a problem at this scale.
- The regularization has to be duration-weighted like the roughness term. Otherwise the long
  final chord of BWV 846 drifted +20 cents as a whole: these roughness models slightly
  reward transposing up, because the critical band is narrower relative to frequency
  in higher registers.
- With a weak regularization (w_reg 0.002) the partials spread to ~33 cents RMS (−30 %
  roughness). The timbre then changes a lot, a risk noted in the plan (harmonicity is not
  modeled).

## Open questions for M6

- Listen to the renders: is −7 % (intonation) or −20 % (partials) of model roughness
  audible, and do the partial changes sound like a different instrument?
- A target dissonance (`OptimizeConfig.target_ratio`) is already supported. A per-chord-type
  target and a regularization sweep (trade-off curve) remain.
- A harmonicity term to keep the timbre fused, and a beating term for slowly beating partials.
