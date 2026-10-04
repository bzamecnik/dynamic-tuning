"""M2: render the 12-key progression and corpus pieces in each fixed tuning to WAV."""

import argparse
import re
import time
from pathlib import Path

from dynamic_tuning.pieces import PIECES, load_corpus
from dynamic_tuning.progression import progression_score
from dynamic_tuning.score import Score
from dynamic_tuning.spectrum import Timbre
from dynamic_tuning.synth import render, write_wav
from dynamic_tuning.tuning import TUNINGS

OUT = Path("outputs/m2")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def render_all(score: Score, prefix: str, timbre: Timbre, stereo: bool) -> None:
    for tuning in TUNINGS.values():
        t0 = time.perf_counter()
        audio = render(score, tuning, timbre, stereo=stereo)
        path = OUT / f"{prefix}_{slug(tuning.name)}.wav"
        write_wav(path, audio)
        print(f"{path}  ({time.perf_counter() - t0:.1f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-partials", type=int, default=8)
    parser.add_argument("--rolloff", type=float, default=1.0)
    parser.add_argument("--pieces", nargs="*", default=list(PIECES), help="music21 corpus names")
    parser.add_argument("--mono", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    timbre = Timbre(n_partials=args.n_partials, rolloff=args.rolloff)
    render_all(progression_score(), "progression", timbre, not args.mono)
    for piece in args.pieces:
        name = piece.split("/")[-1]
        render_all(load_corpus(piece), slug(name), timbre, not args.mono)


if __name__ == "__main__":
    main()
