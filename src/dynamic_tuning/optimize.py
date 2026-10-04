"""Gradient-based optimization of intonation and partials to control roughness (PyTorch).

Parameters (see `adjust.SpectrumParams`): a cents offset per (note, segment) and optionally
a cents offset per (note, segment, partial), both bounded by a tanh parametrization.

Loss = w_diss * L_diss + w_reg * L_reg + w_smooth * L_smooth, where
- L_diss: duration-weighted mean roughness of the segments relative to the base spectrum,
  or, with a target ratio r, the squared deviation from r times the base roughness,
- L_reg: duration-weighted mean squared deviation (in units of `reg_scale` cents) from the
  base tuning (intonation) and from the base partials (weighted like L_diss, so a long
  chord can't drift away cheaply),
- L_smooth: mean squared change (in units of `smooth_scale` cents) of a sounding note's
  intonation between adjacent segments (and of its partials), so held notes don't jump.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .adjust import SpectrumParams
from .dissonance import DEFAULT_MODEL, RoughnessModel, get_model
from .spectrum import ResolvedSpectrum

_PAD_FREQ = 1000.0  # frequency of padding partials (with zero amplitude)


@dataclass(frozen=True)
class OptimizeConfig:
    model: str = DEFAULT_MODEL
    optimize_intonation: bool = True
    optimize_partials: bool = False
    include_self: bool | None = None  # intra-note pairs; default: only if partials move
    max_cents: float = 30.0  # bound of note intonation offsets
    max_partial_cents: float = 50.0  # bound of partial offsets (~3 %)
    fix_fundamental: bool = True  # the first partial moves only with the note's intonation
    w_diss: float = 1.0
    w_reg: float = 0.02
    w_smooth: float = 0.05
    reg_scale: float = 10.0  # cents
    smooth_scale: float = 10.0  # cents
    target_ratio: float | None = None  # None = minimize; r = aim at r * base roughness
    steps: int = 400
    lr: float = 0.05
    seed: int = 0


@dataclass
class OptimizeResult:
    params: SpectrumParams
    loss: float
    history: list[dict[str, float]] = field(default_factory=list)


class SpectrumBatch:
    """Padded tensors of a resolved spectrum: S segments x N notes x K partials."""

    def __init__(self, spectrum: ResolvedSpectrum, base: SpectrumParams | None = None) -> None:
        segs = spectrum.segments
        n_seg = len(segs)
        n_notes = max(len(s.notes) for s in segs)
        n_partials = segs[0].freqs.shape[1]
        freqs = np.full((n_seg, n_notes, n_partials), _PAD_FREQ)
        amps = np.zeros((n_seg, n_notes, n_partials))
        mask = np.zeros((n_seg, n_notes), dtype=bool)
        note_ids = np.full((n_seg, n_notes), -1)
        for s, seg in enumerate(segs):
            n = len(seg.notes)
            freqs[s, :n], amps[s, :n] = seg.freqs, seg.amps
            if base is not None:  # apply fixed base parameters (e.g. gains) to the batch
                factor = 2.0 ** ((base.cents[s][:, None] + base.partial_cents[s]) / 1200.0)
                freqs[s, :n] *= factor
                amps[s, :n] *= base.gains[s]
            mask[s, :n] = True
            note_ids[s, :n] = seg.notes
        self.spectrum = spectrum
        self.freqs = torch.tensor(freqs)
        self.amps = torch.tensor(amps)
        self.mask = torch.tensor(mask)
        self.note_ids = note_ids
        durations = np.array([s.duration for s in segs])
        self.weights = torch.tensor(durations / durations.sum())
        self.shape = (n_seg, n_notes, n_partials)

        # pairs of partials (flattened note*partial index), and whether they're intra-note
        i, j = np.triu_indices(n_notes * n_partials, k=1)
        self.pair_i, self.pair_j = torch.tensor(i), torch.tensor(j)
        self.intra = torch.tensor(i // n_partials == j // n_partials)

        # transitions: the same note in adjacent segments, as (s, row) -> (s + 1, row')
        transitions = []
        for s in range(n_seg - 1):
            if abs(segs[s].end - segs[s + 1].start) > 1e-9:
                continue
            rows = {int(n): r for r, n in enumerate(segs[s + 1].notes)}
            for r, n in enumerate(segs[s].notes):
                if int(n) in rows:
                    transitions.append((s, r, s + 1, rows[int(n)]))
        self.transitions = torch.tensor(transitions, dtype=torch.long).reshape(-1, 4)

    def roughness(
        self,
        cents: torch.Tensor,
        partial_cents: torch.Tensor,
        model: RoughnessModel,
        include_self: bool,
    ) -> torch.Tensor:
        """Roughness per segment, shape (S,), for offsets (S, N) and (S, N, K)."""
        factor = torch.pow(2.0, (cents[..., None] + partial_cents) / 1200.0)
        freqs = (self.freqs * factor).flatten(1)
        amps = self.amps.flatten(1)
        f1, f2 = freqs[:, self.pair_i], freqs[:, self.pair_j]
        a1, a2 = amps[:, self.pair_i], amps[:, self.pair_j]
        swap = f1 > f2
        f_lo, f_hi = torch.where(swap, f2, f1), torch.where(swap, f1, f2)
        a_lo, a_hi = torch.where(swap, a2, a1), torch.where(swap, a1, a2)
        d = model.pair(f_lo, f_hi, a_lo, a_hi, torch)
        active = (a_lo >= 1e-6) & (a_hi >= 1e-6)
        if not include_self:
            active = active & ~self.intra
        d = torch.where(active, d, torch.zeros_like(d))
        total = d.sum(dim=1)
        if model.energy_normalized:
            energy = (amps**2).sum(dim=1)
            total = total / torch.clamp(energy, min=1e-12)
        return total

    def to_params(self, cents: np.ndarray, partial_cents: np.ndarray) -> SpectrumParams:
        params = SpectrumParams.zeros(self.spectrum)
        for s, seg in enumerate(self.spectrum):
            n = len(seg.notes)
            params.cents[s] = cents[s, :n].copy()
            params.partial_cents[s] = partial_cents[s, :n].copy()
        return params

    def from_params(self, params: SpectrumParams) -> tuple[np.ndarray, np.ndarray]:
        cents = np.zeros(self.shape[:2])
        pcents = np.zeros(self.shape)
        for s, seg in enumerate(self.spectrum):
            n = len(seg.notes)
            cents[s, :n], pcents[s, :n] = params.cents[s], params.partial_cents[s]
        return cents, pcents


def _bounded(u: torch.Tensor, bound: float) -> torch.Tensor:
    return bound * torch.tanh(u)


def _unbounded(x: np.ndarray, bound: float) -> torch.Tensor:
    return torch.tensor(np.arctanh(np.clip(x / bound, -0.999, 0.999)))


def optimize(
    spectrum: ResolvedSpectrum,
    config: OptimizeConfig = OptimizeConfig(),  # noqa: B008 (immutable)
    init: SpectrumParams | None = None,
    fixed: SpectrumParams | None = None,
) -> OptimizeResult:
    """Optimize offsets relative to `spectrum` (the base tuning and timbre).

    `init` is the starting point (default: zeros = the base); `fixed` are parameters applied
    to the base before optimization and not changed (e.g. partial gains).
    The returned parameters include `fixed` and are relative to `spectrum`.
    """
    torch.manual_seed(config.seed)
    model = get_model(config.model)
    include_self = config.optimize_partials if config.include_self is None else config.include_self
    batch = SpectrumBatch(spectrum, fixed)
    mask = batch.mask.double()
    # duration-weighted number of sounding notes, normalizes L_reg
    w_seg = batch.weights[:, None]
    n_notes_mean = (w_seg * mask).sum()
    partial_mask = mask[..., None].repeat(1, 1, batch.shape[2])
    if config.fix_fundamental:
        partial_mask[..., 0] = 0.0

    init_cents, init_pcents = batch.from_params(init or SpectrumParams.zeros(spectrum))
    u_cents = _unbounded(init_cents, config.max_cents).requires_grad_(config.optimize_intonation)
    u_pcents = _unbounded(init_pcents, config.max_partial_cents).requires_grad_(
        config.optimize_partials
    )
    variables = [v for v in (u_cents, u_pcents) if v.requires_grad]

    with torch.no_grad():
        base_d = batch.roughness(
            torch.zeros(batch.shape[:2]), torch.zeros(batch.shape), model, include_self
        )
        base_mean = float((batch.weights * base_d).sum())

    def losses() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cents = _bounded(u_cents, config.max_cents) * mask
        pcents = _bounded(u_pcents, config.max_partial_cents) * partial_mask
        d = batch.roughness(cents, pcents, model, include_self)
        if config.target_ratio is None:
            l_diss = (batch.weights * d).sum() / base_mean
        else:
            target = config.target_ratio * base_d
            l_diss = (batch.weights * ((d - target) / base_mean) ** 2).sum()
        l_reg = (w_seg * (cents / config.reg_scale).pow(2)).sum() / n_notes_mean
        l_reg = l_reg + (w_seg[..., None] * (pcents / config.reg_scale).pow(2)).sum() / (
            n_notes_mean * (batch.shape[2] - int(config.fix_fundamental))
        )
        l_smooth = torch.zeros(())
        if len(batch.transitions):
            s0, r0, s1, r1 = batch.transitions.T
            dc = (cents[s0, r0] - cents[s1, r1]) / config.smooth_scale
            dp = (pcents[s0, r0] - pcents[s1, r1]) / config.smooth_scale
            l_smooth = dc.pow(2).mean() + dp.pow(2).mean()
        total = config.w_diss * l_diss + config.w_reg * l_reg + config.w_smooth * l_smooth
        parts = {
            "loss": total,
            "diss": l_diss,
            "reg": l_reg,
            "smooth": l_smooth,
            "roughness": (batch.weights * d).sum(),
        }
        return total, parts

    history: list[dict[str, float]] = []
    if variables:
        opt = torch.optim.Adam(variables, lr=config.lr)
        for step in range(config.steps):
            opt.zero_grad()
            total, parts = losses()
            total.backward()
            opt.step()
            if step % 10 == 0 or step == config.steps - 1:
                history.append({"step": step, **{k: float(v.detach()) for k, v in parts.items()}})

    with torch.no_grad():
        total, parts = losses()
        cents = (_bounded(u_cents, config.max_cents) * mask).numpy()
        pcents = (_bounded(u_pcents, config.max_partial_cents) * partial_mask).numpy()
    history.append({"step": config.steps, **{k: float(v) for k, v in parts.items()}})

    params = batch.to_params(cents, pcents)
    if fixed is not None:
        for s in range(len(params.cents)):
            params.cents[s] += fixed.cents[s]
            params.partial_cents[s] += fixed.partial_cents[s]
            params.gains[s] = fixed.gains[s].copy()
    return OptimizeResult(params, float(total), history)


def optimize_multistart(
    spectrum: ResolvedSpectrum,
    config: OptimizeConfig = OptimizeConfig(),  # noqa: B008 (immutable)
    inits: list[SpectrumParams | None] | None = None,
    fixed: SpectrumParams | None = None,
) -> OptimizeResult:
    """Run from several starting points (default: the base) and keep the lowest loss."""
    results = [optimize(spectrum, config, init, fixed) for init in (inits or [None])]
    return min(results, key=lambda r: r.loss)


# --- baselines ---


def _inter_note_roughness(freqs: np.ndarray, amps: np.ndarray, model: RoughnessModel) -> np.ndarray:
    """Inter-note roughness for a batch of candidate spectra: freqs (C, N, K), amps (N, K)."""
    n_notes, n_partials = amps.shape
    i, j = np.triu_indices(n_notes * n_partials, k=1)
    inter = i // n_partials != j // n_partials
    i, j = i[inter], j[inter]
    f = freqs.reshape(len(freqs), -1)
    a = amps.ravel()
    f1, f2 = f[:, i], f[:, j]
    a1, a2 = np.broadcast_to(a[i], f1.shape), np.broadcast_to(a[j], f1.shape)
    swap = f1 > f2
    d = model.pair(
        np.where(swap, f2, f1), np.where(swap, f1, f2),
        np.where(swap, a2, a1), np.where(swap, a1, a2), np,
    )  # fmt: skip
    d = np.where((a1 >= 1e-6) & (a2 >= 1e-6), d, 0.0).sum(axis=1)
    if model.energy_normalized:
        d = d / max(float((a**2).sum()), 1e-12)
    return d


def grid_search(
    spectrum: ResolvedSpectrum,
    model: str = DEFAULT_MODEL,
    max_cents: float = 25.0,
    step: float = 5.0,
    max_combinations: int = 20000,
    sweeps: int = 3,
) -> SpectrumParams:
    """Per-segment search over note offsets on a grid, independently for each segment.

    The lowest note stays at 0 (fixes the overall pitch). Exhaustive if the number of
    combinations is small, otherwise coordinate-wise search (`sweeps` passes over the notes).
    Only inter-note roughness counts; no smoothness between segments.
    """
    m = get_model(model)
    grid = np.arange(-max_cents, max_cents + 1e-9, step)
    params = SpectrumParams.zeros(spectrum)
    for s, seg in enumerate(spectrum):
        n = len(seg.notes)
        if n < 2:
            continue
        free = [i for i in range(n) if i != int(np.argmin(seg.freqs[:, 0]))]

        def evaluate(candidates: np.ndarray, seg: Any = seg) -> np.ndarray:
            freqs = seg.freqs[None] * 2.0 ** (candidates[:, :, None] / 1200.0)
            return _inter_note_roughness(freqs, seg.amps, m)

        if len(grid) ** len(free) <= max_combinations:
            candidates = np.zeros((len(grid) ** len(free), n))
            candidates[:, free] = np.array(list(itertools.product(grid, repeat=len(free))))
            best = candidates[int(np.argmin(evaluate(candidates)))]
        else:
            best = np.zeros(n)
            for _ in range(sweeps):
                for i in free:
                    candidates = np.tile(best, (len(grid), 1))
                    candidates[:, i] = grid
                    best = candidates[int(np.argmin(evaluate(candidates)))]
        params.cents[s] = best.copy()
    return params
