"""Fixed 12-note tunings as cents tables, optionally relative to the section tonic.

A tuning is given by the cents of the 12 chromatic degrees above its reference pitch
class. The reference keeps its 12-TET frequency (A4 = 440 Hz grid), so tunings differ
only by small offsets from 12-TET. A key-relative tuning moves its reference to the
tonic of the current section (e.g. just intonation retuned per key).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

A4_FREQ = 440.0
A4_PITCH = 69

SYNTONIC_COMMA = 1200 * math.log2(81 / 80)  # ~21.506 cents
PYTHAGOREAN_COMMA = 1200 * math.log2(3**12 / 2**19)  # ~23.460 cents
PURE_FIFTH = 1200 * math.log2(3 / 2)  # ~701.955 cents


def ratio_to_cents(ratio: float) -> float:
    return 1200 * math.log2(ratio)


@dataclass(frozen=True)
class Tuning:
    name: str
    cents: tuple[float, ...]  # 12 values: cents of chromatic degree i above the reference
    reference: int = 0  # pitch class kept at its 12-TET frequency (if not key-relative)
    key_relative: bool = False  # reference follows the section tonic

    def __post_init__(self) -> None:
        if len(self.cents) != 12:
            raise ValueError("A 12-note tuning needs 12 cents values")

    def offsets(self, tonic: int | None = None) -> np.ndarray:
        """Deviation from 12-TET in cents for each pitch class 0-11."""
        ref = tonic if self.key_relative and tonic is not None else self.reference
        degrees = (np.arange(12) - ref) % 12
        return np.asarray(self.cents)[degrees] - 100.0 * degrees

    def frequency(self, pitch: int | np.ndarray, tonic: int | None = None) -> np.ndarray:
        pitch = np.asarray(pitch)
        cents = 100.0 * (pitch - A4_PITCH) + self.offsets(tonic)[pitch % 12]
        return A4_FREQ * 2.0 ** (cents / 1200.0)


def from_ratios(name: str, ratios: list[float], **kwargs: object) -> Tuning:
    return Tuning(name, tuple(ratio_to_cents(r) for r in ratios), **kwargs)  # type: ignore[arg-type]


def from_fifths(name: str, fifths: dict[int, float], **kwargs: object) -> Tuning:
    """Tuning from a chain of fifths: {position on the circle of fifths: fifth size in cents}.

    Position n means the fifth from degree 7*n to degree 7*(n+1) (mod 12), so position 0
    is C-G, -1 is F-C, etc. Missing positions are ignored; every degree must be
    reachable from the reference.
    """
    cents = {0: 0.0}
    for n in sorted(k for k in fifths if k >= 0):  # upwards from the reference
        cents[(7 * (n + 1)) % 12] = (cents[(7 * n) % 12] + fifths[n]) % 1200
    for n in sorted((k for k in fifths if k < 0), reverse=True):  # downwards
        cents[(7 * n) % 12] = (cents[(7 * (n + 1)) % 12] - fifths[n]) % 1200
    if len(cents) != 12:
        raise ValueError("The chain of fifths must reach all 12 degrees")
    return Tuning(name, tuple(cents[i] for i in range(12)), **kwargs)  # type: ignore[arg-type]


def _chain(
    lo: int, hi: int, sizes: dict[int, float] | None = None, default: float = PURE_FIFTH
) -> dict[int, float]:
    """Fifths at positions lo..hi-1 (lo < 0 <= hi), with optional custom sizes."""
    sizes = sizes or {}
    return {n: sizes.get(n, default) for n in range(lo, hi)}


EQUAL = Tuning("12-TET", tuple(100.0 * i for i in range(12)))

# 5-limit just intonation: major/minor thirds 5/4, 6/5; minor seventh 16/9 (as in V7).
JUST_RATIOS = [
    1, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2, 8 / 5, 5 / 3, 16 / 9, 15 / 8,
]  # fmt: skip
JUST_C = from_ratios("JI (C)", JUST_RATIOS)
JUST_KEY = from_ratios("JI (key)", JUST_RATIOS, key_relative=True)

# Pythagorean, Eb..G#: wolf fifth G#-Eb
PYTHAGOREAN = from_fifths("Pythagorean", _chain(-3, 8))

# 1/4-comma meantone, Eb..G#: pure major thirds C-E, wolf G#-Eb
QUARTER_COMMA_MEANTONE = from_fifths(
    "1/4-comma meantone", _chain(-3, 8, default=PURE_FIFTH - SYNTONIC_COMMA / 4)
)

# Kirnberger III: C-G-D-A-E tempered by 1/4 syntonic comma (C-E pure),
# F#-C# by a schisma, others pure.
_SCHISMA = PYTHAGOREAN_COMMA - SYNTONIC_COMMA
KIRNBERGER_III = from_fifths(
    "Kirnberger III",
    _chain(
        -4, 7, {n: PURE_FIFTH - SYNTONIC_COMMA / 4 for n in range(4)} | {6: PURE_FIFTH - _SCHISMA}
    ),
)

# Werckmeister III: C-G-D-A and B-F# tempered by 1/4 Pythagorean comma, others pure.
WERCKMEISTER_III = from_fifths(
    "Werckmeister III",
    _chain(-4, 7, {n: PURE_FIFTH - PYTHAGOREAN_COMMA / 4 for n in (0, 1, 2, 5)}),
)

TUNINGS: dict[str, Tuning] = {
    t.name: t
    for t in [
        EQUAL,
        JUST_C,
        JUST_KEY,
        QUARTER_COMMA_MEANTONE,
        KIRNBERGER_III,
        WERCKMEISTER_III,
        PYTHAGOREAN,
    ]
}
