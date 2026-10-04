"""M3: dissonance profiles of the progression (and a chorale) in each fixed tuning.

Writes plots and markdown summaries to outputs/m3/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dynamic_tuning.analysis import (
    Profile,
    dissonance_profile,
    plot_group_heatmap,
    plot_pair_heatmap,
    plot_profiles,
    save,
    summary_table,
)
from dynamic_tuning.chorales import load_chorale
from dynamic_tuning.progression import key_cycle, progression_score
from dynamic_tuning.score import Score, key_name
from dynamic_tuning.spectrum import Timbre, resolve
from dynamic_tuning.tuning import EQUAL, TUNINGS

OUT = Path("outputs/m3")
REF = EQUAL.name
COLORS = {name: f"C{i}" for i, name in enumerate(TUNINGS)}


def profiles_for(score: Score, timbre: Timbre, model: str) -> dict[str, Profile]:
    return {
        name: dissonance_profile(resolve(score, t, timbre), model, name=name)
        for name, t in TUNINGS.items()
    }


def relative(profiles: dict[str, Profile]) -> dict[str, Profile]:
    """Per-segment ratio to the reference tuning (segments are identical across tunings)."""
    ref = profiles[REF].value
    return {
        name: Profile(name, p.start, p.end, p.value / ref, p.chords, p.qualities, p.keys)
        for name, p in profiles.items()
        if name != REF
    }


def analyze(score: Score, prefix: str, timbre: Timbre, model: str, chord_labels: bool) -> str:
    profiles = profiles_for(score, timbre, model)
    desc = f"{model}, {timbre.n_partials} partials, rolloff {timbre.rolloff}"

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    plot_profiles(profiles, ax=axes[0], title=f"{prefix}: roughness ({desc})", colors=COLORS)
    plot_profiles(relative(profiles), ax=axes[1], chord_labels=chord_labels, colors=COLORS)
    axes[1].axhline(1.0, color="k", linewidth=0.8)
    axes[1].set_ylabel(f"roughness / {REF}")
    save(fig, str(OUT / f"{prefix}_profiles.png"))

    report = [f"## {prefix}\n", f"Model: {desc}. Inter-note roughness, duration-weighted means.\n"]
    for by in ["figure", "quality", "key"] if chord_labels else ["key"]:
        save(
            plot_group_heatmap(profiles, by=by, relative_to=REF, title=f"{prefix}: by {by}"),
            str(OUT / f"{prefix}_by_{by}.png"),
        )
        report += [f"### By {by}\n", summary_table(profiles, by=by), ""]
    return "\n".join(report)


def pair_heatmaps(score: Score, timbre: Timbre, model: str, chord: str) -> None:
    """Pair contributions of one chord (first occurrence of the label) across tunings."""
    label = next(c for c in score.chords if c.name == chord)
    tunings = list(TUNINGS.items())
    n_cols = 4
    n_rows = -(-len(tunings) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)
    for ax in axes.flat[len(tunings) :]:
        ax.set_visible(False)
    for ax, (name, t) in zip(axes.flat, tunings, strict=False):
        spec = resolve(score, t, timbre)
        seg = next(s for s in spec if s.start <= label.start + 1e-6 < s.end)
        plot_pair_heatmap(seg, score, model, ax=ax, title=f"{chord} - {name}")
    save(fig, str(OUT / f"pairs_{chord.replace(': ', '_')}.png"))


def key_dependence(score: Score, timbre: Timbre, model: str) -> None:
    """Mean roughness per key (in cycle order) relative to 12-TET, one line per tuning."""
    rel = {name: p.group_means("key") for name, p in profiles_for(score, timbre, model).items()}
    keys = [key_name(t) for t in key_cycle()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, means in rel.items():
        if name != REF:
            ratios = [means[k] / rel[REF][k] for k in keys]
            ax.plot(keys, ratios, marker="o", label=name, color=COLORS[name])
    ax.axhline(1.0, color="k", linewidth=0.8)
    ax.set_xlabel("key (cycle of fourths)")
    ax.set_ylabel(f"mean roughness / {REF}")
    ax.set_title(
        f"Key dependence ({model}, {timbre.n_partials} partials, rolloff {timbre.rolloff})"
    )
    ax.legend(fontsize=8)
    save(fig, str(OUT / "key_dependence.png"))


def sensitivity(score: Score) -> str:
    """Mean roughness relative to 12-TET over timbre parameters and models."""
    rows = []
    for model in ["sethares1993", "vassilakis2001"]:
        for n_partials in [4, 8, 16]:
            for rolloff in [0.5, 1.0, 2.0]:
                profiles = profiles_for(score, Timbre(n_partials, rolloff), model)
                ref = profiles[REF].mean()
                ratios = [profiles[name].mean() / ref for name in TUNINGS if name != REF]
                rows.append((model, n_partials, rolloff, ratios))
    names = [name for name in TUNINGS if name != REF]
    lines = [
        f"Mean roughness relative to {REF} (< 1 = smoother than {REF}).\n",
        "| model | partials | rolloff | " + " | ".join(names) + " |",
        "|" + "---|" * (len(names) + 3),
    ]
    for model, n, r, ratios in rows:
        lines.append(f"| {model} | {n} | {r} | " + " | ".join(f"{x:.3f}" for x in ratios) + " |")

    data = np.array([ratios for *_, ratios in rows])
    fig, ax = plt.subplots(figsize=(10, 7))
    lim = np.abs(np.log2(data)).max()
    im = ax.imshow(np.log2(data), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [f"{m[:4]} n={n} r={r}" for m, n, r, _ in rows])
    fig.colorbar(im, ax=ax, label=f"log2 ratio to {REF}")
    ax.set_title("Sensitivity of the tuning comparison to timbre and model")
    save(fig, str(OUT / "sensitivity.png"))
    return "## Sensitivity\n\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-partials", type=int, default=8)
    parser.add_argument("--rolloff", type=float, default=1.0)
    parser.add_argument("--model", default="sethares1993")
    parser.add_argument("--chorale", default="bach/bwv66.6")
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    timbre = Timbre(args.n_partials, args.rolloff)
    score = progression_score()

    report = ["# M3: dissonance of fixed tunings\n"]
    report.append(analyze(score, "progression", timbre, args.model, chord_labels=True))
    key_dependence(score, timbre, args.model)
    pair_heatmaps(score, timbre, args.model, "C: V7")
    pair_heatmaps(score, timbre, args.model, "C: I")
    if args.chorale:
        chorale = load_chorale(args.chorale)
        name = args.chorale.split("/")[-1]
        report.append(analyze(chorale, name, timbre, args.model, chord_labels=False))
    if not args.no_sensitivity:
        report.append(sensitivity(score))

    (OUT / "summary.md").write_text("\n\n".join(report) + "\n")
    print("\n\n".join(report))
    print(f"\nWrote plots and summary.md to {OUT}")


if __name__ == "__main__":
    main()
