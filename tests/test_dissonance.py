import dissonant
import numpy as np
import pytest
from scipy.signal import argrelmin

from dynamic_tuning.dissonance import MODELS, pair_dissonance, roughness, roughness_matrix
from dynamic_tuning.spectrum import SINE, Timbre
from dynamic_tuning.tuning import EQUAL, JUST_C, Tuning

BRIGHT = Timbre(n_partials=8, rolloff=0.5)


def chord(
    pitches: list[int],
    tuning: Tuning = EQUAL,
    timbre: Timbre = BRIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    return timbre.partials(tuning.frequency(np.array(pitches)))


@pytest.mark.parametrize("model", ["sethares1993", "vassilakis2001"])
def test_matches_dissonant_reference(model: str) -> None:
    rng = np.random.default_rng(0)
    freqs = np.sort(rng.uniform(100, 2000, size=15))
    # amplitudes decreasing with frequency: there dissonant's Vassilakis matches the paper
    amps = np.sort(rng.uniform(0.1, 1, size=15))[::-1]
    expected = dissonant.dissonance(freqs, amps, model=model)
    perm = rng.permutation(15)
    actual = roughness(freqs[perm].reshape(3, 5), amps[perm].reshape(3, 5), model)
    assert actual == pytest.approx(expected)


def test_vassilakis_symmetric_in_amplitudes() -> None:
    """Swapping which partial is louder doesn't change the result (it does in dissonant)."""
    f1, f2 = np.array([300.0]), np.array([320.0])
    a, b = np.array([1.0]), np.array([0.3])
    assert pair_dissonance(f1, a, f2, b, "vassilakis2001") == pytest.approx(
        pair_dissonance(f1, b, f2, a, "vassilakis2001")
    )


def test_hutchinson_knopoff() -> None:
    f1, f2 = 440.0, 460.0
    y = 20 / (1.72 * 450**0.65)
    g = ((y / 0.25) * np.exp(1 - y / 0.25)) ** 2
    freqs, amps = np.array([[f1], [f2]]), np.array([[1.0], [0.5]])
    assert roughness(freqs, amps, "hutchinson_knopoff1978") == pytest.approx(0.5 * g / 1.25)
    # maximum at a quarter of the critical bandwidth, zero beyond 1.2 CBW
    far = np.array([[440.0], [440.0 + 1.3 * 1.72 * 500**0.65]])
    assert roughness(far, np.ones((2, 1)), "hutchinson_knopoff1978") == 0


def test_unknown_model() -> None:
    with pytest.raises(ValueError):
        roughness(np.ones((2, 1)), np.ones((2, 1)), "nonexistent")


@pytest.mark.parametrize("model", list(MODELS))
def test_pair_dissonance_is_symmetric(model: str) -> None:
    f1, f2, a1, a2 = np.array([200.0]), np.array([210.0]), np.array([1.0]), np.array([0.5])
    d1 = pair_dissonance(f1, a1, f2, a2, model)
    d2 = pair_dissonance(f2, a2, f1, a1, model)
    assert d1 == pytest.approx(d2)
    assert d1[0] > 0


def test_pair_dissonance_zero_amplitude() -> None:
    d = pair_dissonance(np.array([200.0]), np.array([0.0]), np.array([210.0]), np.array([0.0]),
                        "vassilakis2001")  # fmt: skip
    assert d.tolist() == [0.0]


def test_unison_and_far_apart_are_smooth() -> None:
    f = np.array([440.0])
    a = np.array([1.0])
    assert pair_dissonance(f, a, f, a)[0] == pytest.approx(0)
    assert pair_dissonance(f, a, 4 * f, a)[0] < 1e-3


def test_roughness_matrix() -> None:
    freqs, amps = chord([60, 64, 67])
    m = roughness_matrix(freqs, amps)
    assert m.shape == (3, 3)
    assert np.allclose(np.tril(m, -1), 0)
    assert m.sum() == pytest.approx(roughness(freqs, amps))
    assert roughness(freqs, amps, include_self=False) == pytest.approx(m.sum() - np.trace(m))
    # each note alone: only the diagonal
    single = roughness(freqs[:1], amps[:1])
    assert single == pytest.approx(m[0, 0])


def test_roughness_note_order_invariant() -> None:
    freqs, amps = chord([60, 64, 67, 70])
    perm = [2, 0, 3, 1]
    assert roughness(freqs[perm], amps[perm]) == pytest.approx(roughness(freqs, amps))


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("interval", [3, 4, 7, 8, 9, 12])  # m3 M3 P5 m6 M6 P8
def test_just_intervals_smoother_than_equal(interval: int, model: str) -> None:
    # Note: with a darker timbre (1/k amplitudes) the Sethares model rates the JI and
    # 12-TET major third C4-E4 about equal, because the slightly mistuned coinciding
    # partials (5th of C, 4th of E) beat too slowly to count as rough in this register.
    ji = roughness(*chord([60, 60 + interval], JUST_C), model=model, include_self=False)
    et = roughness(*chord([60, 60 + interval], EQUAL), model=model, include_self=False)
    if interval == 12:  # the octave is pure in both
        assert ji == pytest.approx(et)
    else:
        assert ji < et


def test_sine_timbre_hides_tuning_differences() -> None:
    def diff(timbre: Timbre) -> float:
        ji = roughness(*chord([60, 64, 67], JUST_C, timbre), include_self=False)
        et = roughness(*chord([60, 64, 67], EQUAL, timbre), include_self=False)
        return abs(ji - et) / et

    assert diff(SINE) < 0.05
    assert diff(BRIGHT) > 3 * diff(SINE)


def test_sethares_curve_minima_at_simple_ratios() -> None:
    """Classic dissonance curve of two harmonic tones: local minima at simple ratios."""
    timbre = Timbre(n_partials=6, rolloff=0.0)
    base = 261.63
    ratios = np.linspace(1.0, 2.05, 2000)
    curve = []
    for r in ratios:
        freqs, amps = timbre.partials(np.array([base, base * r]), np.array([1.0, 1.0]))
        curve.append(roughness(freqs, amps * 0.88 ** np.arange(6), include_self=False))
    minima = ratios[argrelmin(np.array(curve), order=10)[0]]
    for simple in [6 / 5, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 2.0]:
        assert np.min(np.abs(minima - simple)) < 0.005, simple
