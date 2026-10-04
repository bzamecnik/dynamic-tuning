"""M5: optimized intonation and partials vs baselines, with renders and before/after plots.

Variants per piece (all relative to their base tuning, roughness incl. intra-note pairs):
- 12-TET and Kirnberger III (fixed tunings)
- adaptive JI (best reference note per chord, rule from M4)
- grid search per chord (baseline: independent per segment, lowest note fixed)
- optimized intonation, independent segments (no smoothness: Sethares-style adaptation)
- optimized intonation (smooth), multi-start from the base and from adaptive JI
- optimized partials only, and intonation + partials
- Kirnberger III + optimized intonation
Outputs go to outputs/m5/.
"""

import argparse
import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dynamic_tuning.adjust import SpectrumParams, apply, held_note_jumps, just_chords
from dynamic_tuning.analysis import (
    Profile,
    dissonance_profile,
    plot_comparison,
    plot_group_heatmap,
    plot_intonation,
    save,
)
from dynamic_tuning.dissonance import DEFAULT_MODEL, MODELS
from dynamic_tuning.optimize import (
    OptimizeConfig,
    OptimizeResult,
    grid_search,
    optimize,
    optimize_multistart,
)
from dynamic_tuning.pieces import PIECES, load_corpus
from dynamic_tuning.progression import progression_score
from dynamic_tuning.score import Score
from dynamic_tuning.spectrum import ResolvedSpectrum, Timbre, resolve
from dynamic_tuning.synth import synthesize, write_wav
from dynamic_tuning.tuning import EQUAL, KIRNBERGER_III

OUT = Path("outputs/m5")
REF = "12-TET"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_piece(
    score: Score, prefix: str, timbre: Timbre, config: OptimizeConfig, render: bool
) -> list[str]:
    base = resolve(score, EQUAL, timbre)
    kirnberger = resolve(score, KIRNBERGER_III, timbre)
    zero = SpectrumParams.zeros(base)
    ji = just_chords("best", config.model)(base, zero)
    intonation = replace(config, include_self=True)

    variants: dict[str, tuple[ResolvedSpectrum, SpectrumParams]] = {}
    results: dict[str, OptimizeResult] = {}
    timings: dict[str, float] = {}

    def run(
        name: str, spectrum: ResolvedSpectrum, fn: Callable[[], SpectrumParams | OptimizeResult]
    ) -> None:
        t0 = time.perf_counter()
        out = fn()
        timings[name] = time.perf_counter() - t0
        if isinstance(out, OptimizeResult):
            results[name] = out
            out = out.params
        variants[name] = (spectrum, out)

    independent = replace(intonation, w_smooth=0.0)
    partials_only = replace(config, optimize_intonation=False, optimize_partials=True)
    both = replace(config, optimize_partials=True)
    run("12-TET", base, lambda: zero)
    run("Kirnberger III", kirnberger, lambda: SpectrumParams.zeros(kirnberger))
    run("adaptive JI (rule)", base, lambda: ji)
    run("grid search per chord", base, lambda: grid_search(base, config.model))
    run("opt. intonation, independent", base, lambda: optimize(base, independent))
    run("opt. intonation", base, lambda: optimize_multistart(base, intonation, [None, ji]))
    run("opt. partials", base, lambda: optimize(base, partials_only))
    run("opt. intonation + partials", base, lambda: optimize(base, both))
    run("Kirnberger III + opt. intonation", kirnberger, lambda: optimize(kirnberger, intonation))

    profiles: dict[str, Profile] = {}
    rows = []
    for name, (spectrum, params) in variants.items():
        out = apply(spectrum, params)
        profiles[name] = dissonance_profile(out, config.model, include_self=True, name=name)
        jumps = held_note_jumps(spectrum, params)
        cents = np.concatenate(params.cents)
        pcents = np.concatenate([p[:, 1:].ravel() for p in params.partial_cents])
        rows.append((name, profiles[name].mean(), np.sqrt((cents**2).mean()),
                     jumps.max() if len(jumps) else 0.0, np.sqrt((pcents**2).mean()),
                     timings[name]))  # fmt: skip
        if render:
            write_wav(OUT / f"{prefix}_{slug(name)}.wav", synthesize(out, stereo=True))
        if np.abs(cents).max() > 0 and name not in ("12-TET", "Kirnberger III"):
            ax = plot_intonation(params.note_cents(spectrum), score, title=f"{prefix}: {name}")
            save(ax, str(OUT / f"{prefix}_intonation_{slug(name)}.png"))

    colors = {name: f"C{i}" for i, name in enumerate(profiles)}
    title = f"{prefix}: roughness ({config.model}, incl. intra-note pairs)"
    fig = plot_comparison(profiles, REF, title, bool(score.chords), colors)
    save(fig, str(OUT / f"{prefix}_profiles.png"))
    by = "figure" if score.chords else "key"
    save(plot_group_heatmap(profiles, by=by, relative_to=REF, title=f"{prefix}: by {by}"),
         str(OUT / f"{prefix}_by_{by}.png"))  # fmt: skip
    plot_partials(variants, prefix)
    plot_history(results, prefix)

    ref = profiles[REF].mean()
    lines = [
        f"## {prefix}\n",
        "| variant | roughness vs 12-TET | rms cents vs base | max jump of a held note "
        "[cents] | rms partial cents | time [s] |",
        "|---|---|---|---|---|---|",
    ]
    for name, mean, rms_c, jump, rms_p, t in rows:
        lines.append(
            f"| {name} | {100 * (mean / ref - 1):+.1f} % | {rms_c:.1f} | {jump:.1f} "
            f"| {rms_p:.1f} | {t:.1f} |"
        )
    return lines


def plot_partials(
    variants: dict[str, tuple[ResolvedSpectrum, SpectrumParams]], prefix: str
) -> None:
    """Mean and spread of partial offsets per partial number, for variants that move them."""
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = 0
    for name, (_, params) in variants.items():
        p = np.concatenate(params.partial_cents)
        if np.abs(p).max() == 0:
            continue
        plotted += 1
        k = np.arange(1, p.shape[1] + 1) + 0.12 * plotted
        ax.errorbar(k, p.mean(axis=0), yerr=p.std(axis=0), fmt="o", capsize=3, label=name)
    harmonic = np.arange(1, 9)
    et = 100 * np.round(1200 * np.log2(harmonic) / 100) - 1200 * np.log2(harmonic)
    ax.plot(harmonic, et, "k_", markersize=14, label="12-TET grid (M4 rule)")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("partial number")
    ax.set_ylabel("offset [cents] (mean ± std)")
    ax.set_title(f"{prefix}: optimized partial offsets")
    ax.legend(fontsize=8)
    save(fig, str(OUT / f"{prefix}_partials.png"))


def plot_history(results: dict[str, OptimizeResult], prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, r in results.items():
        steps = [h["step"] for h in r.history]
        axes[0].plot(steps, [h["loss"] for h in r.history], label=name)
        axes[1].plot(steps, [h["diss"] for h in r.history], label=name)
    axes[0].set_title("total loss")
    axes[1].set_title("roughness relative to base")
    for ax in axes:
        ax.set_xlabel("step")
    axes[1].legend(fontsize=8)
    save(fig, str(OUT / f"{prefix}_history.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-partials", type=int, default=8)
    parser.add_argument("--rolloff", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--w-reg", type=float, default=OptimizeConfig.w_reg)
    parser.add_argument("--w-smooth", type=float, default=OptimizeConfig.w_smooth)
    parser.add_argument("--steps", type=int, default=OptimizeConfig.steps)
    parser.add_argument("--pieces", nargs="*", default=["bach/bwv846"])
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    timbre = Timbre(args.n_partials, args.rolloff)
    config = OptimizeConfig(
        model=args.model, w_reg=args.w_reg, w_smooth=args.w_smooth, steps=args.steps
    )
    render = not args.no_render
    report = [f"# M5: optimization ({config})\n"]
    report += run_piece(progression_score(), "progression", timbre, config, render)
    for piece in args.pieces:
        score = load_corpus(piece, PIECES.get(piece))
        report += ["", *run_piece(score, slug(piece.split("/")[-1]), timbre, config, render)]
    (OUT / "summary.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
