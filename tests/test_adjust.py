import numpy as np
import pytest

from dynamic_tuning.adjust import (
    SpectrumParams,
    adjusted,
    apply,
    compose,
    detune,
    just_chords,
    mute_partials,
    partials_to_equal_temperament,
    scale_partials,
    stretch_partials,
)
from dynamic_tuning.dissonance import roughness
from dynamic_tuning.score import Note, Score
from dynamic_tuning.spectrum import Timbre, resolve
from dynamic_tuning.tuning import EQUAL


@pytest.fixture
def spectrum():
    # C major triad in root position, then G7 over a held G
    notes = [
        Note(0, 1, 48),
        Note(0, 1, 64),
        Note(0, 2, 67),
        Note(1, 1, 71),
        Note(1, 1, 74),
        Note(1, 1, 77),
    ]
    return resolve(Score(notes), EQUAL, Timbre(n_partials=6))


def cents_between(f1: float, f2: float) -> float:
    return 1200 * np.log2(f2 / f1)


def test_zero_params_are_identity(spectrum) -> None:
    out = apply(spectrum, SpectrumParams.zeros(spectrum))
    for a, b in zip(spectrum, out, strict=True):
        assert np.allclose(a.freqs, b.freqs) and np.allclose(a.amps, b.amps)
    assert out is not spectrum


def test_apply_does_not_modify_input(spectrum) -> None:
    before = spectrum.segments[0].freqs.copy()
    adjusted(spectrum, detune(lambda seg, n: 10.0))
    assert np.array_equal(spectrum.segments[0].freqs, before)


def test_detune(spectrum) -> None:
    out = adjusted(spectrum, detune(lambda seg, n: 100.0 if n == 0 else 0.0))
    assert cents_between(spectrum.segments[0].freqs[0, 0], out.segments[0].freqs[0, 0]) == (
        pytest.approx(100)
    )
    assert np.allclose(out.segments[0].freqs[1:], spectrum.segments[0].freqs[1:])


def test_stretch_partials(spectrum) -> None:
    out = adjusted(spectrum, stretch_partials(0.01))
    f = out.segments[0].freqs[0]
    assert f[0] == pytest.approx(spectrum.segments[0].freqs[0, 0])
    assert np.allclose(f / f[0], np.arange(1, 7) ** 1.01)


def test_partials_to_equal_temperament(spectrum) -> None:
    out = adjusted(spectrum, partials_to_equal_temperament())
    ratios_cents = 1200 * np.log2(out.segments[0].freqs[0] / out.segments[0].freqs[0, 0])
    assert np.allclose(ratios_cents, [0, 1200, 1900, 2400, 2800, 3100])
    # partials of notes in 12-TET intervals coincide: partial 3 of C3 = G4 (12-TET)
    f = out.segments[0].freqs
    assert f[0, 2] == pytest.approx(f[2, 0])


def test_scale_and_mute_partials(spectrum) -> None:
    out = adjusted(spectrum, compose(scale_partials([1, 0.5]), mute_partials([2])))
    a0, a1 = spectrum.segments[0].amps[0], out.segments[0].amps[0]
    assert a1[0] == a0[0] and a1[1] == pytest.approx(0.5 * a0[1]) and a1[2] == 0
    assert np.allclose(a1[3:], a0[3:])


def test_just_chords_bass(spectrum) -> None:
    params = just_chords("bass")(spectrum, SpectrumParams.zeros(spectrum))
    out = apply(spectrum, params)
    f = out.segments[0].freqs[:, 0]  # C3 E4 G4
    assert f[0] == pytest.approx(spectrum.segments[0].freqs[0, 0])  # bass unchanged
    assert f[1] / f[0] == pytest.approx(5 / 2)  # C3 -> E4: major tenth
    assert f[2] / f[0] == pytest.approx(3.0)
    g = out.segments[1].freqs[:, 0]  # G4 B4 D5 F5
    assert g[1] / g[0] == pytest.approx(5 / 4)
    assert g[2] / g[0] == pytest.approx(3 / 2)
    assert g[3] / g[0] == pytest.approx(9 / 5)


def test_just_chords_retune_held_note(spectrum) -> None:
    params = just_chords("bass")(spectrum, SpectrumParams.zeros(spectrum))
    track = params.note_cents(spectrum)[2]  # G4: fifth above C3, then bass of G7
    assert track[0][1] == pytest.approx(1.955, abs=0.01)
    assert track[1][1] == pytest.approx(0.0)


def test_just_chords_reduce_roughness(spectrum) -> None:
    def total(spec) -> float:
        return sum(roughness(s.freqs, s.amps, include_self=False) for s in spec)

    bass = adjusted(spectrum, just_chords("bass"))
    best = adjusted(spectrum, just_chords("best"))
    assert total(bass) < total(spectrum)
    assert total(best) <= total(bass) + 1e-12


def test_just_chords_invalid() -> None:
    with pytest.raises(ValueError):
        just_chords("top")


def test_just_chords_minor_seventh_has_pure_fifths() -> None:
    # ii7 in C: D F A C, all fifths (D-A, F-C) pure
    notes = [Note(0, 1, p) for p in (50, 53, 57, 60)]
    spectrum = resolve(Score(notes), EQUAL, Timbre(n_partials=1))
    f = adjusted(spectrum, just_chords("bass")).segments[0].freqs[:, 0]
    assert f[2] / f[0] == pytest.approx(3 / 2)
    assert f[3] / f[1] == pytest.approx(3 / 2)
