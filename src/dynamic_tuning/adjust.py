"""Adjustments of a resolved spectrum: per (note, segment) intonation and per-partial changes.

`SpectrumParams` holds, aligned with the segments of a `ResolvedSpectrum`:
- `cents[s][i]`: intonation offset of note i in segment s (shifts all its partials),
- `partial_cents[s][i, k]`: extra frequency offset of partial k,
- `gains[s][i, k]`: amplitude multiplier of partial k (0 mutes it).

Hand-made rules below produce such parameters; the optimizer (M5) uses the same
parametrization.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .dissonance import DEFAULT_MODEL, roughness
from .spectrum import ResolvedSpectrum, SegmentSpectrum
from .tuning import JUST_RATIOS


@dataclass
class SpectrumParams:
    cents: list[np.ndarray]  # per segment: (n_notes,)
    partial_cents: list[np.ndarray]  # per segment: (n_notes, n_partials)
    gains: list[np.ndarray]  # per segment: (n_notes, n_partials)

    @classmethod
    def zeros(cls, spectrum: ResolvedSpectrum) -> SpectrumParams:
        return cls(
            cents=[np.zeros(len(seg.notes)) for seg in spectrum],
            partial_cents=[np.zeros_like(seg.freqs) for seg in spectrum],
            gains=[np.ones_like(seg.amps) for seg in spectrum],
        )

    def copy(self) -> SpectrumParams:
        return SpectrumParams(
            [c.copy() for c in self.cents],
            [c.copy() for c in self.partial_cents],
            [g.copy() for g in self.gains],
        )

    def note_cents(self, spectrum: ResolvedSpectrum) -> dict[int, list[tuple[float, float]]]:
        """Intonation trajectory per note: {note index: [(segment start, cents), ...]}."""
        result: dict[int, list[tuple[float, float]]] = {}
        for seg, cents in zip(spectrum, self.cents, strict=True):
            for note, c in zip(seg.notes, cents, strict=True):
                result.setdefault(int(note), []).append((seg.start, float(c)))
        return result


def apply(spectrum: ResolvedSpectrum, params: SpectrumParams) -> ResolvedSpectrum:
    """New spectrum with the parameters applied (the input is not modified)."""
    segments = []
    for seg, cents, pcents, gains in zip(
        spectrum, params.cents, params.partial_cents, params.gains, strict=True
    ):
        factor = 2.0 ** ((cents[:, None] + pcents) / 1200.0)
        segments.append(replace(seg, freqs=seg.freqs * factor, amps=seg.amps * gains))
    return ResolvedSpectrum(spectrum.score, segments)


# --- rules: functions (spectrum, params) -> params, composable with `compose` ---

Rule = Callable[[ResolvedSpectrum, SpectrumParams], SpectrumParams]


def compose(*rules: Rule) -> Rule:
    def composed(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        for rule in rules:
            params = rule(spectrum, params)
        return params

    return composed


def adjusted(spectrum: ResolvedSpectrum, *rules: Rule) -> ResolvedSpectrum:
    """Apply rules starting from zero parameters."""
    return apply(spectrum, compose(*rules)(spectrum, SpectrumParams.zeros(spectrum)))


def detune(cents_of: Callable[[SegmentSpectrum, int], float]) -> Rule:
    """Add an intonation offset given by a function of (segment, note index)."""

    def rule(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        params = params.copy()
        for seg, cents in zip(spectrum, params.cents, strict=True):
            cents += [cents_of(seg, int(n)) for n in seg.notes]
        return params

    return rule


def _ratios(seg: SegmentSpectrum) -> np.ndarray:
    """Partial frequency ratios to the first partial, (n_notes, n_partials)."""
    return seg.freqs / seg.freqs[:, :1]


def stretch_partials(stretch: float) -> Rule:
    """Inharmonicity: partial ratio r -> r^(1 + stretch) (e.g. a stretched piano-like timbre)."""

    def rule(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        params = params.copy()
        for seg, pcents in zip(spectrum, params.partial_cents, strict=True):
            pcents += 1200.0 * stretch * np.log2(_ratios(seg))
        return params

    return rule


def partials_to_equal_temperament(steps_per_octave: int = 12) -> Rule:
    """Move each partial to the nearest step of the equal-tempered grid above its fundamental.

    Sethares' idea of a timbre matched to the scale: then the partials of notes in any
    12-TET interval coincide exactly, as harmonic partials do in just intervals.
    E.g. partial 3 (701.96 cents) -> 700 cents, partial 5 (386.31) -> 400.
    """

    def rule(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        params = params.copy()
        step = 1200.0 / steps_per_octave
        for seg, pcents in zip(spectrum, params.partial_cents, strict=True):
            current = 1200.0 * np.log2(_ratios(seg)) + pcents
            pcents += step * np.round(current / step) - current
        return params

    return rule


def scale_partials(gains: list[float] | np.ndarray) -> Rule:
    """Multiply partial amplitudes by per-partial gains (0 mutes a partial).

    If fewer gains than partials are given, the remaining partials are unchanged.
    """

    def rule(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        params = params.copy()
        g = np.asarray(gains, dtype=float)
        for gain in params.gains:
            n = min(len(g), gain.shape[1])
            gain[:, :n] *= g[:n]
        return params

    return rule


def mute_partials(indices: list[int]) -> Rule:
    """Mute partials by 0-based index (e.g. [6] mutes the 7th harmonic)."""

    def rule(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        params = params.copy()
        for gain in params.gains:
            gain[:, [i for i in indices if i < gain.shape[1]]] = 0.0
        return params

    return rule


JUST_CENTS = np.array([1200.0 * math.log2(r) for r in JUST_RATIOS])


def _just_offsets(seg: SegmentSpectrum, reference: int, cents: np.ndarray) -> np.ndarray:
    """Offsets that put each note at its 5-limit just interval above the reference note.

    The reference note keeps its current pitch; intervals are classified by the nearest
    12-TET semitone of the current pitch difference.
    """
    pitch = 1200.0 * np.log2(seg.freqs[:, 0]) + cents  # cents on an absolute scale
    diff = pitch - pitch[reference]
    semitones = np.round(diff / 100.0).astype(int)
    octaves, degree = np.divmod(semitones, 12)
    target = 1200.0 * octaves + JUST_CENTS[degree]
    return cents + (target - diff)


def just_chords(reference: str = "bass", model: str = DEFAULT_MODEL) -> Rule:
    """Adaptive just intonation: retune each segment's notes to 5-limit just intervals.

    reference: "bass" keeps the lowest note at its base pitch and tunes the others
    relative to it; "best" tries each sounding note as the reference and keeps the
    variant with the lowest roughness. The reference keeps its base pitch, so there is
    no long-term drift, but held notes may move (with a glide) between chords.
    """
    if reference not in ("bass", "best"):
        raise ValueError("reference must be 'bass' or 'best'")

    def rule(spectrum: ResolvedSpectrum, params: SpectrumParams) -> SpectrumParams:
        params = params.copy()
        for s, seg in enumerate(spectrum):
            cents = params.cents[s]
            if len(seg.notes) < 2:
                continue
            if reference == "bass":
                params.cents[s] = _just_offsets(seg, int(np.argmin(seg.freqs[:, 0])), cents)
                continue
            best, best_value = cents, math.inf
            for ref in range(len(seg.notes)):
                candidate = _just_offsets(seg, ref, cents)
                factor = 2.0 ** ((candidate[:, None] + params.partial_cents[s]) / 1200.0)
                value = roughness(
                    seg.freqs * factor, seg.amps * params.gains[s], model, include_self=False
                )
                if value < best_value:
                    best, best_value = candidate, value
            params.cents[s] = best
        return params

    return rule
