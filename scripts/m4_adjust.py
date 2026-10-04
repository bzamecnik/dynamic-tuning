"""M4: hand-made spectrum adjustments compared with fixed tunings.

For each piece: renders (WAV), dissonance profiles relative to 12-TET, summary tables and
intonation trajectories of the adaptive variants. Outputs go to outputs/m4/.
Roughness here includes pairs within a note, since some variants change the timbre.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from dynamic_tuning.adjust import (
    Rule,
    SpectrumParams,
    apply,
    just_chords,
    partials_to_equal_temperament,
    stretch_partials,
)
from dynamic_tuning.analysis import (
    Profile,
    dissonance_profile,
    plot_comparison,
    plot_group_heatmap,
    plot_intonation,
    save,
    summary_table,
)
from dynamic_tuning.dissonance import DEFAULT_MODEL, MODELS
from dynamic_tuning.pieces import PIECES, load_corpus
from dynamic_tuning.progression import progression_score
from dynamic_tuning.score import Score
from dynamic_tuning.spectrum import ResolvedSpectrum, Timbre, resolve
from dynamic_tuning.synth import synthesize, write_wav
from dynamic_tuning.tuning import EQUAL, JUST_KEY, KIRNBERGER_III, Tuning

OUT = Path("outputs/m4")
REF = "12-TET"

# name -> (base tuning, rule producing the adjustment parameters, or None)
Variant = tuple[Tuning, Rule | None]


def variants(model: str) -> dict[str, Variant]:
    return {
        "12-TET": (EQUAL, None),
        "Kirnberger III": (KIRNBERGER_III, None),
        "JI (key)": (JUST_KEY, None),
        "12-TET, ET-matched partials": (EQUAL, partials_to_equal_temperament()),
        "12-TET, stretched partials": (EQUAL, stretch_partials(0.005)),
        "adaptive JI (bass)": (EQUAL, just_chords("bass")),
        "adaptive JI (best)": (EQUAL, just_chords("best", model)),
    }


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_piece(score: Score, prefix: str, timbre: Timbre, model: str, render: bool) -> str:
    profiles: dict[str, Profile] = {}
    spectra: dict[str, ResolvedSpectrum] = {}
    for name, (tuning, rule) in variants(model).items():
        base = resolve(score, tuning, timbre)
        params = SpectrumParams.zeros(base)
        if rule is not None:
            params = rule(base, params)
            if any(abs(c).max() > 0 for c in params.cents if len(c)):
                plot = plot_intonation(params.note_cents(base), score, title=f"{prefix}: {name}")
                save(plot, str(OUT / f"{prefix}_intonation_{slug(name)}.png"))
        spectra[name] = apply(base, params)
        profiles[name] = dissonance_profile(spectra[name], model, include_self=True, name=name)
        if render:
            write_wav(OUT / f"{prefix}_{slug(name)}.wav", synthesize(spectra[name], stereo=True))

    colors = {name: f"C{i}" for i, name in enumerate(profiles)}
    chord_labels = bool(score.chords)
    title = f"{prefix}: roughness ({model}, incl. intra-note pairs)"
    save(
        plot_comparison(profiles, REF, title, chord_labels, colors),
        str(OUT / f"{prefix}_profiles.png"),
    )
    report = [f"## {prefix}\n"]
    for by in ["figure", "key"] if chord_labels else ["key"]:
        heatmap = plot_group_heatmap(profiles, by=by, relative_to=REF, title=f"{prefix}: by {by}")
        save(heatmap, str(OUT / f"{prefix}_by_{by}.png"))
        report += [f"### By {by}\n", summary_table(profiles, by=by), ""]
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-partials", type=int, default=8)
    parser.add_argument("--rolloff", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--pieces", nargs="*", default=["bach/bwv846"])
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    timbre = Timbre(args.n_partials, args.rolloff)
    render = not args.no_render
    report = [f"# M4: spectrum adjustments ({args.model})\n"]
    report.append(run_piece(progression_score(), "progression", timbre, args.model, render))
    for piece in args.pieces:
        score = load_corpus(piece, PIECES.get(piece))
        report.append(run_piece(score, slug(piece.split("/")[-1]), timbre, args.model, render))
    (OUT / "summary.md").write_text("\n\n".join(report) + "\n")
    print("\n\n".join(report))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
