"""Offline additive synthesis of a resolved spectrum.

Each note is rendered as a sum of sinusoidal partials with phase-continuous
frequency trajectories (phase = integral of the instantaneous frequency), so the
frequencies and amplitudes may change between segments. Changes are smoothed by
short linear glides centered at the segment boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .score import Score
from .spectrum import ResolvedSpectrum, Timbre, resolve
from .tuning import Tuning

DEFAULT_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class Envelope:
    """Linear ADSR envelope (times in seconds, sustain as a level 0-1)."""

    attack: float = 0.02
    decay: float = 0.15
    sustain: float = 0.7
    release: float = 0.2

    def __call__(self, held: float, sample_rate: int) -> np.ndarray:
        """Envelope of a note held for `held` seconds, including the release tail."""
        n_held = round(held * sample_rate)
        n_release = round(self.release * sample_rate)
        t = np.arange(n_held + n_release) / sample_rate
        xp = [0.0, self.attack, self.attack + self.decay]
        fp = [0.0, 1.0, self.sustain]
        if self.attack == 0:
            xp, fp = xp[1:], fp[1:]
        env = np.interp(t, xp, fp)
        level = float(np.interp(held, xp, fp))
        if n_release:
            env[n_held:] = level * (1 - np.arange(n_release) / n_release)
        return env


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Moving average with edge padding; turns steps into linear ramps of `width` samples."""
    if width <= 1 or len(x) < 2:
        return x
    left, right = width // 2, width - 1 - width // 2
    padded = np.concatenate([np.full(left, x[0]), x, np.full(right, x[-1])])
    c = np.cumsum(np.concatenate([[0.0], padded]))
    return (c[width:] - c[:-width]) / width


def _pan_gains(voice: int, voices: list[int], spread: float) -> tuple[float, float]:
    """Constant-power pan; voices spread evenly over [-spread, spread]."""
    pos = 0.0 if len(voices) < 2 else -spread + 2 * spread * voices.index(voice) / (len(voices) - 1)
    angle = (pos + 1) * np.pi / 4
    return float(np.cos(angle)), float(np.sin(angle))


def synthesize(
    spectrum: ResolvedSpectrum,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    envelope: Envelope = Envelope(),  # noqa: B008 (immutable)
    glide: float = 0.03,
    stereo: bool = False,
    stereo_spread: float = 0.6,
    headroom: float = 0.8,
) -> np.ndarray:
    """Render to audio: shape (n_samples,) or (n_samples, 2) if `stereo`.

    The gain depends only on the amplitudes (not on frequencies), so renders of the same
    score and timbre in different tunings have the same loudness scale.
    """
    score = spectrum.score
    n_total = round((score.duration + envelope.release) * sample_rate) + 1
    out = np.zeros((n_total, 2) if stereo else n_total)
    nyquist = 0.5 * sample_rate
    glide_samples = max(1, round(glide * sample_rate))

    for note_idx, note in enumerate(score.notes):
        track = spectrum.note_track(note_idx)
        if not track:
            continue
        start = round(track[0][0].start * sample_rate)
        bounds = [round(seg.end * sample_rate) - start for seg, _ in track]
        lengths = np.diff([0, *bounds])
        env = envelope(bounds[-1] / sample_rate, sample_rate)
        n = len(env)
        lengths[-1] += n - bounds[-1]  # release tail keeps the last segment's spectrum

        freqs = np.stack([seg.freqs[row] for seg, row in track])  # (n_segments, n_partials)
        amps = np.stack([seg.amps[row] for seg, row in track])
        signal = np.zeros(n)
        for k in range(freqs.shape[1]):
            f = _smooth(np.repeat(freqs[:, k], lengths), glide_samples)
            a = _smooth(np.repeat(amps[:, k], lengths), glide_samples)
            a = np.where(f < 0.95 * nyquist, a, 0.0)
            if not a.any():
                continue
            phase = 2 * np.pi * np.cumsum(f) / sample_rate
            signal += a * np.sin(phase - phase[0])
        signal *= env
        end = min(start + n, n_total)
        if stereo:
            left, right = _pan_gains(note.voice, score.voices, stereo_spread)
            out[start:end, 0] += left * signal[: end - start]
            out[start:end, 1] += right * signal[: end - start]
        else:
            out[start:end] += signal[: end - start]

    max_amp = max((float(seg.amps.sum()) for seg in spectrum), default=0.0)
    if max_amp > 0:
        out *= headroom / max_amp
    return np.clip(out, -1.0, 1.0)


def render(
    score: Score,
    tuning: Tuning,
    timbre: Timbre = Timbre(),  # noqa: B008 (immutable)
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    **kwargs: object,
) -> np.ndarray:
    return synthesize(resolve(score, tuning, timbre), sample_rate, **kwargs)  # type: ignore[arg-type]


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")
