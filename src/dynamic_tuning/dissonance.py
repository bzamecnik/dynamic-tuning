"""Sensory roughness of a spectrum of partials.

Pair models (Sethares 1993, Vassilakis 2001, Hutchinson & Knopoff 1978) are written
against an array namespace `xp` (numpy or torch), so the same code is used for
measurement (numpy) and for gradient-based optimization (torch). The `dissonant`
package serves as the reference implementation in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_MODEL = "vassilakis2001"
_MIN_AMP = 1e-6  # as in dissonant.dissonance()
_EPS = 1e-12

# Pair function: (f_lo, f_hi, a_lo, a_hi, xp) -> dissonance, elementwise; f_lo <= f_hi.
PairFunction = Callable[[Any, Any, Any, Any, Any], Any]


def _plomp_levelt(f_lo: Any, f_hi: Any, xp: Any) -> Any:
    """Sethares' parametrization of the Plomp-Levelt curve; maximum ~0.24/s at x ~ 0.24."""
    s = 0.24 / (0.0207 * f_lo + 18.96)
    x = s * (f_hi - f_lo)
    return xp.exp(-3.5 * x) - xp.exp(-5.75 * x)


def sethares1993(f_lo: Any, f_hi: Any, a_lo: Any, a_hi: Any, xp: Any = np) -> Any:
    """Sethares (1993): amplitude product times the Plomp-Levelt curve."""
    return a_lo * a_hi * _plomp_levelt(f_lo, f_hi, xp)


def vassilakis2001(f_lo: Any, f_hi: Any, a_lo: Any, a_hi: Any, xp: Any = np) -> Any:
    """Vassilakis (2001): (A_min*A_max)^0.1 * (2*A_min/(A_min+A_max))^3.11 * Z.

    The constant factor 0.5 of the paper is left out (as in `dissonant`, pairs are counted
    once). Unlike `dissonant` 0.1.1, the amplitude fluctuation degree uses A_min as in
    the paper (`dissonant` uses the amplitude of the upper partial instead).
    """
    a_min = xp.minimum(a_lo, a_hi)
    a_max = xp.maximum(a_lo, a_hi)
    intensity = (a_min * a_max + _EPS) ** 0.1
    degree = (2 * a_min / (a_min + a_max + _EPS)) ** 3.11
    return intensity * degree * _plomp_levelt(f_lo, f_hi, xp)


def hutchinson_knopoff1978(f_lo: Any, f_hi: Any, a_lo: Any, a_hi: Any, xp: Any = np) -> Any:
    """Hutchinson & Knopoff (1978), as implemented in `dycon` (Harrison & Pearce 2020).

    g(y) = ((y/0.25) * exp(1 - y/0.25))^2 for y <= 1.2, where y is the frequency
    difference in critical bandwidths, CBW = 1.72 * mean_f^0.65. The total is normalized
    by the sum of squared amplitudes, see `RoughnessModel.energy_normalized`.
    """
    mean_f = (f_lo + f_hi) / 2
    y = (f_hi - f_lo) / (1.72 * mean_f**0.65)
    g = ((y / 0.25) * xp.exp(1 - y / 0.25)) ** 2
    return a_lo * a_hi * xp.where(y <= 1.2, g, 0.0 * g)


@dataclass(frozen=True)
class RoughnessModel:
    name: str
    pair: PairFunction
    # divide the total by the sum of squared amplitudes of all partials
    energy_normalized: bool = False


MODELS: dict[str, RoughnessModel] = {
    m.name: m
    for m in [
        RoughnessModel("sethares1993", sethares1993),
        RoughnessModel("vassilakis2001", vassilakis2001),
        RoughnessModel("hutchinson_knopoff1978", hutchinson_knopoff1978, energy_normalized=True),
    ]
}


def get_model(model: str | RoughnessModel) -> RoughnessModel:
    if isinstance(model, RoughnessModel):
        return model
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}, available: {list(MODELS)}")
    return MODELS[model]


def pair_dissonance(
    f1: Any, a1: Any, f2: Any, a2: Any, model: str | RoughnessModel = DEFAULT_MODEL, xp: Any = np
) -> Any:
    """Dissonance of pairs of sinusoids (any order of f1, f2); zero where an amplitude is ~0."""
    pair = get_model(model).pair
    swap = f1 > f2
    f_lo, f_hi = xp.where(swap, f2, f1), xp.where(swap, f1, f2)
    a_lo, a_hi = xp.where(swap, a2, a1), xp.where(swap, a1, a2)
    d = pair(f_lo, f_hi, a_lo, a_hi, xp)
    active = (a_lo >= _MIN_AMP) & (a_hi >= _MIN_AMP)
    return xp.where(active, d, 0.0 * d)


def roughness_matrix(
    freqs: np.ndarray, amps: np.ndarray, model: str | RoughnessModel = DEFAULT_MODEL
) -> np.ndarray:
    """Roughness attributed to pairs of notes: upper-triangular matrix (n_notes, n_notes).

    Entry [i, j] (i < j) sums all pairs of partials between notes i and j;
    the diagonal holds the roughness among partials of the same note.
    `freqs`, `amps` have shape (n_notes, n_partials).
    """
    model = get_model(model)
    freqs, amps = np.atleast_2d(freqs), np.atleast_2d(amps)
    n_notes, n_partials = freqs.shape
    f, a = freqs.ravel().astype(float), amps.ravel().astype(float)
    note = np.repeat(np.arange(n_notes), n_partials)
    i, j = np.triu_indices(len(f), k=1)
    d = pair_dissonance(f[i], a[i], f[j], a[j], model)
    matrix = np.zeros((n_notes, n_notes))
    np.add.at(matrix, (note[i], note[j]), d)  # note[i] <= note[j] as i < j
    if model.energy_normalized:
        energy = float((a**2).sum())
        matrix = matrix / energy if energy > 0 else matrix
    return matrix


def roughness(
    freqs: np.ndarray,
    amps: np.ndarray,
    model: str | RoughnessModel = DEFAULT_MODEL,
    include_self: bool = True,
) -> float:
    """Total roughness of a set of notes' partials, shape (n_notes, n_partials).

    With `include_self` (default), pairs within the same note count too, which matches
    `dissonant.dissonance()` on the flattened spectrum. Without it, only pairs between
    different notes count; intra-note roughness depends on timbre and register only.
    """
    matrix = roughness_matrix(freqs, amps, model)
    total = float(matrix.sum())
    return total if include_self else total - float(np.trace(matrix))
