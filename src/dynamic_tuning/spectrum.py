"""Timbre models, segmentation of a score and the resolved spectrum.

The resolved spectrum is the central representation: for each segment of constant
sounding notes, each note has arrays of partial frequencies and amplitudes.
Both the synthesizer and the dissonance meter consume only this.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from .score import Score
from .tuning import Tuning

_EPS = 1e-9


@dataclass(frozen=True)
class Timbre:
    """Harmonic timbre: partial k (1-based) at k*f0 with amplitude 1/k^rolloff.

    `stretch` gives inharmonic partials f_k = f0 * k^(1 + stretch) (0 = harmonic).
    """

    n_partials: int = 8
    rolloff: float = 1.0
    stretch: float = 0.0

    def ratios(self) -> np.ndarray:
        k = np.arange(1, self.n_partials + 1, dtype=float)
        return k ** (1.0 + self.stretch)

    def amplitudes(self) -> np.ndarray:
        k = np.arange(1, self.n_partials + 1, dtype=float)
        return k**-self.rolloff

    def partials(
        self, f0: float | np.ndarray, gain: float | np.ndarray = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Frequencies and amplitudes, shape (..., n_partials) for f0 of shape (...)."""
        f0 = np.asarray(f0, dtype=float)[..., None]
        gain = np.asarray(gain, dtype=float)[..., None]
        freqs = f0 * self.ratios()
        amps = np.broadcast_to(gain * self.amplitudes(), freqs.shape).copy()
        return freqs, amps


SINE = Timbre(n_partials=1)


@dataclass(frozen=True)
class Segment:
    """Time span with a constant set of sounding notes (indices into `score.notes`)."""

    start: float
    end: float
    notes: tuple[int, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment(score: Score, extra_boundaries: tuple[float, ...] = ()) -> list[Segment]:
    """Slice the score at every note onset/offset and section boundary.

    Segments without any sounding note are omitted.
    """
    times = {n.start for n in score.notes} | {n.end for n in score.notes}
    times |= {s.start for s in score.sections} | {s.end for s in score.sections}
    times |= set(extra_boundaries)
    bounds = sorted(t for t in times if 0 <= t <= score.duration)
    # merge boundaries closer than _EPS (floating point noise)
    merged = [bounds[0]] if bounds else []
    for t in bounds[1:]:
        if t - merged[-1] > _EPS:
            merged.append(t)

    starts = np.array([n.start for n in score.notes])
    ends = np.array([n.end for n in score.notes])
    segments = []
    for a, b in itertools.pairwise(merged):
        mid = (a + b) / 2
        idx = np.flatnonzero((starts <= mid) & (mid < ends))
        if len(idx):
            segments.append(Segment(a, b, tuple(int(i) for i in idx)))
    return segments


@dataclass
class SegmentSpectrum:
    """Spectrum of one segment: per sounding note, its partials."""

    start: float
    end: float
    notes: np.ndarray  # (n_notes,) indices into score.notes
    freqs: np.ndarray  # (n_notes, n_partials)
    amps: np.ndarray  # (n_notes, n_partials)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ResolvedSpectrum:
    score: Score
    segments: list[SegmentSpectrum] = field(default_factory=list)

    def __iter__(self) -> Iterator[SegmentSpectrum]:
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def note_track(self, note: int) -> list[tuple[SegmentSpectrum, int]]:
        """Segments in which the note sounds, with its row index in each segment."""
        track = []
        for seg in self.segments:
            rows = np.flatnonzero(seg.notes == note)
            if len(rows):
                track.append((seg, int(rows[0])))
        return track


# Hook to adjust a segment's spectrum (e.g. hand-made adjustments, optimizer output).
SpectrumAdjustment = Callable[[SegmentSpectrum], SegmentSpectrum]


def resolve(
    score: Score,
    tuning: Tuning,
    timbre: Timbre = Timbre(),  # noqa: B008 (immutable)
    adjust: SpectrumAdjustment | None = None,
) -> ResolvedSpectrum:
    """Resolve frequencies (tuning, using the section tonic) and partials (timbre) per segment.

    Amplitudes are steady-state: timbre profile scaled by velocity/127. Envelopes are
    applied only in the synthesizer.
    """
    result = ResolvedSpectrum(score)
    for seg in segment(score):
        section = score.section_at((seg.start + seg.end) / 2)
        tonic = section.tonic if section else None
        notes = [score.notes[i] for i in seg.notes]
        pitches = np.array([n.pitch for n in notes])
        gains = np.array([n.velocity / 127 for n in notes])
        freqs, amps = timbre.partials(tuning.frequency(pitches, tonic), gains)
        spec = SegmentSpectrum(seg.start, seg.end, np.array(seg.notes), freqs, amps)
        result.segments.append(adjust(spec) if adjust else spec)
    return result
