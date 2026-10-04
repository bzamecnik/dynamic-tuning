# CLAUDE.md

Offline Python toolkit: symbolic multi-voice music → arbitrary tuning and timbre (additive synthesis) → WAV, plus dissonance measurement and (later) optimization of intonation/partials. See `docs/plan.md` for the plan and milestones, `docs/initial_vision.md` and `docs/literature.md` for background.

## Commands

- `uv sync` – install the environment (Python 3.13, uv project)
- `uv run pytest` – run tests
- `uv run ruff format` / `uv run ruff check --fix` – format and lint
- `uv run mypy` – type check (src, tests, scripts)
- `uv run python scripts/<name>.py` – milestone scripts, they write to `outputs/` (gitignored)

Before committing, run all of: `ruff format`, `ruff check`, `mypy`, `pytest`.

## Conventions

- Source in `src/dynamic_tuning/`, tests in `tests/` (pytest, one test module per source module), scripts in `scripts/`.
- Use type hints wherever they help (public functions, dataclasses); don't contort code to satisfy the checker. numpy arrays are typed as `np.ndarray` (or `npt.NDArray[np.float64]` where the dtype matters).
- Write unit tests along with the code. Keep tests fast (short renders, low sample rates where possible).
- Central representation is the resolved spectrum: per note and segment, arrays `freqs[k]`, `amps[k]`. Synthesizer and dissonance meter consume only that.
- Score model: our own small dataclasses, time in seconds. `music21` only at generation/import time (Roman-numeral chords, key analysis, Bach chorales corpus); `pretty_midi` for MIDI I/O.
- Dissonance models live in `dissonance.py`, written against an array namespace (`xp`: numpy or torch) so the optimizer (`optimize.py`, PyTorch CPU) uses the same formulas. The `dissonant` package (author's own, numpy-only) is the reference in tests.
- Nothing touches an audio device; everything renders offline to files.
- Commit coherent pieces of work separately, with a descriptive message.
