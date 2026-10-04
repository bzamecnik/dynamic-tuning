"""Sensory roughness of a spectrum of partials, using the models of the `dissonant` package.

The pair model (Sethares 1993, Vassilakis 2001, ...) is taken from `dissonant`; here we
only vectorize over all pairs of partials and attribute the result to pairs of notes.
"""

from __future__ import annotations

import dissonant
import numpy as np

DEFAULT_MODEL = "sethares1993"
_MIN_AMP = 1e-6  # as in dissonant.dissonance()


def pair_dissonance(
    f1: np.ndarray, a1: np.ndarray, f2: np.ndarray, a2: np.ndarray, model: str = DEFAULT_MODEL
) -> np.ndarray:
    """Dissonance of pairs of sinusoids (any order of f1, f2); zero where an amplitude is ~0."""
    f1, a1, f2, a2 = (np.asarray(x, dtype=float) for x in (f1, a1, f2, a2))
    swap = f1 > f2
    f_lo, f_hi = np.where(swap, f2, f1), np.where(swap, f1, f2)
    a_lo, a_hi = np.where(swap, a2, a1), np.where(swap, a1, a2)
    active = (a_lo >= _MIN_AMP) & (a_hi >= _MIN_AMP)
    d = np.zeros(np.broadcast(f_lo, a_lo).shape)
    if active.any():
        d[active] = dissonant.dissonance_pair(
            f_lo[active], f_hi[active], a_lo[active], a_hi[active], model
        )
    return d


def roughness_matrix(freqs: np.ndarray, amps: np.ndarray, model: str = DEFAULT_MODEL) -> np.ndarray:
    """Roughness attributed to pairs of notes: upper-triangular matrix (n_notes, n_notes).

    Entry [i, j] (i < j) sums all pairs of partials between notes i and j;
    the diagonal holds the roughness among partials of the same note.
    `freqs`, `amps` have shape (n_notes, n_partials).
    """
    freqs, amps = np.atleast_2d(freqs), np.atleast_2d(amps)
    n_notes, n_partials = freqs.shape
    f, a = freqs.ravel(), amps.ravel()
    note = np.repeat(np.arange(n_notes), n_partials)
    i, j = np.triu_indices(len(f), k=1)
    d = pair_dissonance(f[i], a[i], f[j], a[j], model)
    matrix = np.zeros((n_notes, n_notes))
    np.add.at(matrix, (note[i], note[j]), d)  # note[i] <= note[j] as i < j
    return matrix


def roughness(
    freqs: np.ndarray, amps: np.ndarray, model: str = DEFAULT_MODEL, include_self: bool = True
) -> float:
    """Total roughness of a set of notes' partials, shape (n_notes, n_partials).

    With `include_self` (default), pairs within the same note count too, which matches
    `dissonant.dissonance()` on the flattened spectrum. Without it, only pairs between
    different notes count; intra-note roughness depends on timbre and register only.
    """
    matrix = roughness_matrix(freqs, amps, model)
    total = float(matrix.sum())
    return total if include_self else total - float(np.trace(matrix))
