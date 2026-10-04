"""Dissonance profiles over time, summaries per chord and key, and plots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .dissonance import DEFAULT_MODEL, roughness, roughness_matrix
from .score import PITCH_CLASS_NAMES, Score
from .spectrum import ResolvedSpectrum, SegmentSpectrum


@dataclass
class Profile:
    """Dissonance per segment of a resolved spectrum."""

    name: str
    start: np.ndarray
    end: np.ndarray
    value: np.ndarray
    chords: list[str]  # chord label per segment ("" if unknown)
    qualities: list[str]
    keys: list[str]  # section key per segment ("" if unknown)

    @property
    def duration(self) -> np.ndarray:
        return self.end - self.start

    def mean(self, mask: np.ndarray | None = None) -> float:
        """Duration-weighted mean over all (or the masked) segments."""
        w = self.duration if mask is None else self.duration * mask
        return float((self.value * w).sum() / w.sum()) if w.sum() > 0 else float("nan")

    def group_means(self, by: str = "figure") -> dict[str, float]:
        """Duration-weighted mean per group: "figure" (e.g. V7), "quality", "chord", "key"."""
        labels = {
            "figure": [c.split(": ")[-1] for c in self.chords],
            "quality": self.qualities,
            "chord": self.chords,
            "key": self.keys,
        }[by]
        groups: dict[str, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label:
                groups[label].append(i)
        result = {}
        for label, idx in groups.items():
            mask = np.zeros(len(self.value), dtype=bool)
            mask[idx] = True
            result[label] = self.mean(mask)
        return result


def segment_dissonance(
    seg: SegmentSpectrum,
    model: str = DEFAULT_MODEL,
    include_self: bool = False,
    normalize: bool = False,
) -> float:
    """Roughness of one segment.

    `normalize` divides by the sum of amplitude products over all pairs of partials of
    different notes, making chords with more notes or partials comparable.
    """
    value = roughness(seg.freqs, seg.amps, model, include_self)
    if normalize:
        a = seg.amps.ravel()
        total = (a.sum() ** 2 - (a**2).sum()) / 2
        if not include_self:
            per_note = seg.amps.sum(axis=1)
            total -= ((per_note**2 - (seg.amps**2).sum(axis=1)) / 2).sum()
        value = value / total if total > 0 else 0.0
    return value


def dissonance_profile(
    spectrum: ResolvedSpectrum,
    model: str = DEFAULT_MODEL,
    include_self: bool = False,
    normalize: bool = False,
    name: str = "",
) -> Profile:
    """Dissonance over time. By default only roughness between different notes is counted:
    the roughness among partials of a single note does not depend on the tuning.
    """
    score = spectrum.score
    values, chords, qualities, keys = [], [], [], []
    for seg in spectrum:
        values.append(segment_dissonance(seg, model, include_self, normalize))
        mid = (seg.start + seg.end) / 2
        chord = score.chord_at(mid)
        section = score.section_at(mid)
        chords.append(chord.name if chord else "")
        qualities.append(chord.quality if chord else "")
        keys.append(section.name if section else "")
    return Profile(
        name=name,
        start=np.array([s.start for s in spectrum]),
        end=np.array([s.end for s in spectrum]),
        value=np.array(values),
        chords=chords,
        qualities=qualities,
        keys=keys,
    )


def relative_profiles(profiles: dict[str, Profile], reference: str) -> dict[str, Profile]:
    """Per-segment ratio to a reference profile (all profiles must share the segments).

    Segments with zero reference roughness (e.g. a single sounding note) get ratio 1.
    The reference itself is left out.
    """
    ref = profiles[reference].value
    return {
        name: replace(p, value=np.divide(p.value, ref, out=np.ones_like(ref), where=ref > 0))
        for name, p in profiles.items()
        if name != reference
    }


def summary_table(profiles: dict[str, Profile], by: str = "figure", fmt: str = ".3f") -> str:
    """Markdown table: rows = groups (in order of first appearance), columns = profiles."""
    means = {name: p.group_means(by) for name, p in profiles.items()}
    rows = list(dict.fromkeys(g for m in means.values() for g in m))
    header = f"| {by} | " + " | ".join(profiles) + " |"
    lines = [header, "|" + "---|" * (len(profiles) + 1)]
    for row in [*rows, "(all)"]:
        cells = [
            format(p.mean() if row == "(all)" else means[name].get(row, float("nan")), fmt)
            for name, p in profiles.items()
        ]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- plots ---


def note_name(pitch: int) -> str:
    return f"{PITCH_CLASS_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _section_spans(profile: Profile) -> list[tuple[float, float, str]]:
    spans: list[tuple[float, float, str]] = []
    for start, end, key in zip(profile.start, profile.end, profile.keys, strict=True):
        if spans and spans[-1][2] == key:
            spans[-1] = (spans[-1][0], float(end), key)
        else:
            spans.append((float(start), float(end), key))
    return spans


def plot_profiles(
    profiles: dict[str, Profile],
    ax: Axes | None = None,
    title: str = "",
    chord_labels: bool = False,
    colors: dict[str, str] | None = None,
) -> Axes:
    """Step curves of dissonance over time, with key sections shaded and labeled.

    `colors` maps profile names to colors, to keep them consistent across plots.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 5))
    first = next(iter(profiles.values()))
    for i, (start, end, key) in enumerate(_section_spans(first)):
        if i % 2:
            ax.axvspan(start, end, color="0.92", zorder=0)
        ax.text(
            (start + end) / 2,
            1.0,
            key,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for name, p in profiles.items():
        x = np.append(p.start, p.end[-1])
        y = np.append(p.value, p.value[-1])
        color = colors.get(name) if colors else None
        ax.step(x, y, where="post", label=name, linewidth=1.2, color=color)
    if chord_labels:
        for start, end, chord in zip(first.start, first.end, first.chords, strict=True):
            if chord:
                ax.text(
                    (start + end) / 2,
                    0.0,
                    chord.split(": ")[-1],
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="0.4",
                )
    ax.set_xlim(first.start[0], first.end[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("roughness")
    if title:
        ax.set_title(title, pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return ax


def plot_comparison(
    profiles: dict[str, Profile],
    reference: str,
    title: str = "",
    chord_labels: bool = False,
    colors: dict[str, str] | None = None,
) -> Figure:
    """Two panels: absolute profiles, and profiles relative to the reference."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    plot_profiles(profiles, ax=axes[0], title=title, colors=colors)
    rel = relative_profiles(profiles, reference)
    plot_profiles(rel, ax=axes[1], chord_labels=chord_labels, colors=colors)
    axes[1].axhline(1.0, color="k", linewidth=0.8)
    axes[1].set_ylabel(f"roughness / {reference}")
    return fig


def plot_intonation(
    note_cents: dict[int, list[tuple[float, float]]],
    score: Score,
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """Intonation offset (cents) of each note over time, colored by voice."""
    if ax is None:
        _, ax = plt.subplots(figsize=(18, 4))
    voices = score.voices
    for note_idx, points in note_cents.items():
        note = score.notes[note_idx]
        times = [t for t, _ in points] + [note.end]
        cents = [c for _, c in points]
        color = f"C{voices.index(note.voice) % 10}"
        ax.stairs(cents, times, baseline=None, color=color, linewidth=1.2)
    for i, v in enumerate(voices):
        ax.plot([], [], color=f"C{i % 10}", label=f"voice {v}")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cents vs base tuning")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_group_heatmap(
    profiles: dict[str, Profile],
    by: str = "figure",
    relative_to: str | None = None,
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """Heatmap of mean dissonance per group (rows) and profile (columns).

    With `relative_to`, values are ratios to that profile (e.g. "12-TET").
    """
    means = {name: p.group_means(by) for name, p in profiles.items()}
    rows = list(dict.fromkeys(g for m in means.values() for g in m))
    data = np.array([[means[c].get(r, np.nan) for c in profiles] for r in rows])
    if relative_to is not None:
        ref = data[:, list(profiles).index(relative_to)][:, None]
        data = data / ref
    if ax is None:
        _, ax = plt.subplots(figsize=(1.2 * len(profiles) + 2, 0.4 * len(rows) + 1.5))
    if relative_to is not None:
        lim = np.nanmax(np.abs(np.log2(data)))
        im = ax.imshow(np.log2(data), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = ax.imshow(data, cmap="viridis", aspect="auto")
    for (r, c), v in np.ndenumerate(data):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(profiles)), list(profiles), rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    label = f"ratio to {relative_to} (color: log2)" if relative_to else "roughness"
    ax.figure.colorbar(im, ax=ax, label=label)
    if title:
        ax.set_title(title)
    return ax


def plot_pair_heatmap(
    seg: SegmentSpectrum,
    score: Score,
    model: str = DEFAULT_MODEL,
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """Roughness contributions of pairs of notes in one segment (diagonal: within a note)."""
    order = np.argsort([score.notes[i].pitch for i in seg.notes])[::-1]  # highest first
    matrix = roughness_matrix(seg.freqs[order], seg.amps[order], model)
    full = matrix + np.triu(matrix, 1).T
    labels = [note_name(score.notes[seg.notes[i]].pitch) for i in order]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(full, cmap="magma")
    for (r, c), v in np.ndenumerate(full):
        ax.text(
            c,
            r,
            f"{v:.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if v < 0.6 * full.max() else "black",
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.figure.colorbar(im, ax=ax, label="roughness")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def save(fig_or_ax: Figure | Axes, path: str) -> None:
    fig = fig_or_ax if isinstance(fig_or_ax, Figure) else fig_or_ax.figure
    assert isinstance(fig, Figure)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
