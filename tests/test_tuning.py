import math

import numpy as np
import pytest

from dynamic_tuning.tuning import (
    EQUAL,
    JUST_C,
    JUST_KEY,
    KIRNBERGER_III,
    PURE_FIFTH,
    PYTHAGOREAN,
    QUARTER_COMMA_MEANTONE,
    SYNTONIC_COMMA,
    TUNINGS,
    WERCKMEISTER_III,
    Tuning,
    from_fifths,
    ratio_to_cents,
)

PURE_MAJOR_THIRD = ratio_to_cents(5 / 4)


def interval(t: Tuning, lo: int, hi: int) -> float:
    """Cents between pitch classes lo and hi (hi above lo, within an octave)."""
    return (t.cents[hi % 12] - t.cents[lo % 12]) % 1200


def test_equal_temperament() -> None:
    assert np.allclose(EQUAL.offsets(), 0)
    assert EQUAL.frequency(69) == pytest.approx(440.0)
    assert EQUAL.frequency(60) == pytest.approx(261.6256, rel=1e-6)
    assert np.allclose(EQUAL.frequency(np.array([57, 81])), [220, 880])


def test_reference_keeps_equal_frequency() -> None:
    for t in TUNINGS.values():
        assert t.frequency(60) == pytest.approx(EQUAL.frequency(60))
        assert t.offsets()[0] == 0


def test_just_intonation_ratios() -> None:
    c4 = EQUAL.frequency(60)
    assert JUST_C.frequency(64) / c4 == pytest.approx(5 / 4)
    assert JUST_C.frequency(67) / c4 == pytest.approx(3 / 2)
    assert JUST_C.frequency(72 + 9) / c4 == pytest.approx(2 * 5 / 3)
    # the syntonic comma: D-A is not a pure fifth in fixed C just intonation
    assert interval(JUST_C, 2, 9) == pytest.approx(PURE_FIFTH - SYNTONIC_COMMA)


def test_key_relative_just_intonation() -> None:
    # in G, B is a pure major third above the tonic G (kept at its 12-TET frequency)
    g = JUST_KEY.frequency(67, tonic=7)
    assert g == pytest.approx(EQUAL.frequency(67))
    assert JUST_KEY.frequency(71, tonic=7) / g == pytest.approx(5 / 4)
    # fixed JI ignores the tonic
    assert np.allclose(JUST_C.offsets(7), JUST_C.offsets())
    assert not np.allclose(JUST_KEY.offsets(7), JUST_KEY.offsets(0))


def test_meantone_pure_thirds_and_wolf() -> None:
    t = QUARTER_COMMA_MEANTONE
    for root in [0, 2, 3, 5, 7, 9, 10]:  # C D Eb F G A Bb
        assert interval(t, root, root + 4) == pytest.approx(PURE_MAJOR_THIRD)
    assert interval(t, 0, 7) == pytest.approx(PURE_FIFTH - SYNTONIC_COMMA / 4)
    wolf = interval(t, 8, 3)  # G# -> Eb
    assert wolf == pytest.approx(737.6, abs=0.1)


def test_pythagorean_pure_fifths() -> None:
    for root in [3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]:  # Eb .. C#
        assert interval(PYTHAGOREAN, root, root + 7) == pytest.approx(PURE_FIFTH)


def test_kirnberger_iii() -> None:
    assert interval(KIRNBERGER_III, 0, 4) == pytest.approx(PURE_MAJOR_THIRD)
    expected = [0, 90.22, 193.16, 294.13, 386.31, 498.04, 590.22, 696.58, 792.18, 889.74,
                996.09, 1088.27]  # fmt: skip
    assert np.allclose(KIRNBERGER_III.cents, expected, atol=0.01)


def test_werckmeister_iii() -> None:
    expected = [0, 90.22, 192.18, 294.13, 390.22, 498.04, 588.27, 696.09, 792.18, 888.27,
                996.09, 1092.18]  # fmt: skip
    assert np.allclose(WERCKMEISTER_III.cents, expected, atol=0.01)


def test_circle_closes() -> None:
    """The 12 fifths of each tuning sum to 7 octaves."""
    for t in TUNINGS.values():
        total = sum(interval(t, 7 * i, 7 * (i + 1)) for i in range(12))
        assert total == pytest.approx(7 * 1200)


def test_cents_monotonic() -> None:
    for t in TUNINGS.values():
        assert all(np.diff(t.cents) > 0), t.name


def test_invalid_tunings() -> None:
    with pytest.raises(ValueError):
        Tuning("short", (0.0, 100.0))
    with pytest.raises(ValueError):
        from_fifths("incomplete", {0: 700.0, 1: 700.0})


def test_ratio_to_cents() -> None:
    assert ratio_to_cents(2) == pytest.approx(1200)
    assert ratio_to_cents(math.sqrt(2)) == pytest.approx(600)
