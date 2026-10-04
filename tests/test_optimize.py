import numpy as np
import pytest
import torch

from dynamic_tuning.adjust import SpectrumParams, apply, held_note_jumps, just_chords
from dynamic_tuning.dissonance import MODELS, get_model, roughness
from dynamic_tuning.optimize import (
    OptimizeConfig,
    SpectrumBatch,
    grid_search,
    optimize,
    optimize_multistart,
)
from dynamic_tuning.score import Note, Score
from dynamic_tuning.spectrum import Timbre, resolve
from dynamic_tuning.tuning import EQUAL


@pytest.fixture(scope="module")
def spectrum():
    # C major triad, then F major over a held C (C held through both chords)
    notes = [
        Note(0, 1, 48),
        Note(0, 1, 64),
        Note(0, 2, 72),
        Note(1, 1, 53),
        Note(1, 1, 69),
    ]
    return resolve(Score(notes), EQUAL, Timbre(n_partials=6))


def total_roughness(spec, include_self: bool = False) -> float:
    return sum(s.duration * roughness(s.freqs, s.amps, include_self=include_self) for s in spec)


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("include_self", [False, True])
def test_torch_roughness_matches_numpy(spectrum, model: str, include_self: bool) -> None:
    batch = SpectrumBatch(spectrum)
    rng = np.random.default_rng(1)
    cents = rng.normal(0, 10, batch.shape[:2])
    pcents = rng.normal(0, 5, batch.shape)
    d = batch.roughness(torch.tensor(cents), torch.tensor(pcents), get_model(model), include_self)
    params = batch.to_params(cents, pcents)
    for s, seg in enumerate(apply(spectrum, params)):
        expected = roughness(seg.freqs, seg.amps, model, include_self)
        assert float(d[s]) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("model", list(MODELS))
def test_gradient_matches_finite_differences(spectrum, model: str) -> None:
    batch = SpectrumBatch(spectrum)
    m = get_model(model)
    pcents = torch.zeros(batch.shape)
    # stay away from the HK cutoff discontinuity and exact coincidences
    cents = torch.tensor(np.random.default_rng(2).normal(0, 7, batch.shape[:2]))
    cents.requires_grad_(True)

    def f(c: torch.Tensor) -> torch.Tensor:
        return batch.roughness(c, pcents, m, False).sum()

    assert torch.autograd.gradcheck(f, (cents,), eps=1e-4, atol=1e-5, rtol=1e-3)


def test_transitions(spectrum) -> None:
    batch = SpectrumBatch(spectrum)
    # the held C5 (note 2) is the only note in both segments
    assert batch.transitions.tolist() == [[0, 2, 1, 0]]


def test_optimize_reduces_roughness_within_bounds(spectrum) -> None:
    config = OptimizeConfig(steps=200, max_cents=20.0)
    result = optimize(spectrum, config)
    out = apply(spectrum, result.params)
    assert total_roughness(out) < 0.95 * total_roughness(spectrum)
    assert all(np.abs(c).max() <= 20.0 for c in result.params.cents)
    assert result.history[-1]["loss"] < result.history[0]["loss"]


def test_major_triad_goes_towards_just(spectrum) -> None:
    result = optimize(spectrum, OptimizeConfig(steps=300, w_reg=0.0, w_smooth=0.0))
    c = result.params.cents[0]  # C3 E4 C5
    third = c[1] - c[0]  # E relative to C: just is -13.7 cents
    assert -16 < third < -8


def test_strong_smoothness_keeps_held_note_steady(spectrum) -> None:
    free = optimize(spectrum, OptimizeConfig(steps=200, w_smooth=0.0))
    smooth = optimize(spectrum, OptimizeConfig(steps=200, w_smooth=100.0))
    assert held_note_jumps(spectrum, smooth.params).max() < 0.5
    assert (
        held_note_jumps(spectrum, smooth.params).max()
        <= held_note_jumps(spectrum, free.params).max()
    )


def test_strong_regularization_keeps_base(spectrum) -> None:
    result = optimize(spectrum, OptimizeConfig(steps=100, w_reg=1000.0))
    assert all(np.abs(c).max() < 0.5 for c in result.params.cents)


def test_optimize_partials(spectrum) -> None:
    config = OptimizeConfig(steps=150, optimize_intonation=False, optimize_partials=True)
    result = optimize(spectrum, config)
    assert all(np.all(c == 0) for c in result.params.cents)
    assert all(np.all(p[:, 0] == 0) for p in result.params.partial_cents)  # fundamental fixed
    assert any(np.abs(p).max() > 0.1 for p in result.params.partial_cents)
    out = apply(spectrum, result.params)
    assert total_roughness(out, True) < total_roughness(spectrum, True)


def test_target_ratio(spectrum) -> None:
    result = optimize(spectrum, OptimizeConfig(steps=300, target_ratio=0.9, w_reg=0, w_smooth=0))
    ratio = total_roughness(apply(spectrum, result.params)) / total_roughness(spectrum)
    assert ratio == pytest.approx(0.9, abs=0.02)


def test_fixed_and_init(spectrum) -> None:
    fixed = SpectrumParams.zeros(spectrum)
    for g in fixed.gains:
        g[:, 3:] = 0.0  # mute upper partials
    init = just_chords("bass")(spectrum, SpectrumParams.zeros(spectrum))
    result = optimize_multistart(spectrum, OptimizeConfig(steps=50), [None, init], fixed)
    assert all(np.all(g[:, 3:] == 0) for g in result.params.gains)


def test_grid_search(spectrum) -> None:
    params = grid_search(spectrum, max_cents=20, step=2)
    c = params.cents[0]
    assert c[0] == 0  # lowest note fixed
    assert -16 < c[1] < -10  # E4 towards a just major third above C
    assert total_roughness(apply(spectrum, params)) < total_roughness(spectrum)
    coordinate = grid_search(spectrum, max_cents=20, step=2, max_combinations=1)
    assert total_roughness(apply(spectrum, coordinate)) < total_roughness(spectrum)
