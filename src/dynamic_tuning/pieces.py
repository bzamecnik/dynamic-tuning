"""Import of pieces from the music21 corpus (Bach chorales, WTC I prelude BWV 846) into `Score`.

music21 is used only here (and in `progression`), the rest of the code works on `Score`.
"""

from __future__ import annotations

from pathlib import Path

from music21 import corpus, stream
from music21.analysis import discrete, windowed

from .score import Note, Score, Section, merge_sections

# Pieces used in the experiments, with their default tempo (quarter notes per minute).
PIECES: dict[str, float] = {"bach/bwv66.6": 80.0, "bach/bwv846": 72.0}


def list_bach_works() -> list[str]:
    """Names of Bach works in the music21 corpus, e.g. "bach/bwv66.6".

    Nearly all are 4-part chorales; "bach/bwv846" is the C major prelude from WTC I.
    """
    return sorted({f"bach/{Path(str(p)).stem}" for p in corpus.getComposer("bach")})


def load_corpus(
    name: str = "bach/bwv66.6",
    tempo: float | None = None,
    key_window: int | None = 8,
    min_section_beats: float = 4.0,
) -> Score:
    """Load a piece from the corpus; each part becomes a voice (top part first).

    E.g. "bach/bwv66.6" (chorale: 0 = soprano, ..., 3 = bass) or "bach/bwv846"
    (C major prelude from WTC I, two staves, written-out sustain).

    Sections: if `key_window` is set (in quarter notes), keys are estimated by a sliding
    Krumhansl-Schmuckler window and short sections are merged into neighbors. Otherwise
    the whole piece is one section with the globally estimated key.
    The tempo defaults to the one in `PIECES` (or 80).
    """
    s = corpus.parse(name)
    assert isinstance(s, stream.Score)
    tempo = tempo or PIECES.get(name, 80.0)
    return corpus_to_score(s, tempo, key_window, min_section_beats)


def corpus_to_score(
    s: stream.Score,
    tempo: float = 80.0,
    key_window: int | None = 8,
    min_section_beats: float = 4.0,
) -> Score:
    sec_per_beat = 60.0 / tempo
    notes = []
    for voice, part in enumerate(s.parts):
        for n in part.stripTies().flatten().notes:
            for p in n.pitches:  # a chord in a part yields several notes
                notes.append(
                    Note(
                        start=float(n.offset) * sec_per_beat,
                        duration=float(n.quarterLength) * sec_per_beat,
                        pitch=int(p.midi),
                        voice=voice,
                    )
                )
    total_beats = float(s.highestTime)
    if key_window:
        beat_keys = _windowed_keys(s, key_window, total_beats)
    else:
        k = s.analyze("key")
        beat_keys = [(k.tonic.pitchClass, k.mode)] * max(1, int(total_beats))
    sections = _keys_to_sections(beat_keys, total_beats, min_section_beats)
    sections = [
        Section(x.start * sec_per_beat, x.end * sec_per_beat, x.tonic, x.mode) for x in sections
    ]
    return Score(notes=notes, sections=sections, tempo=tempo).sorted()


def _windowed_keys(s: stream.Score, window: int, total_beats: float) -> list[tuple[int, str]]:
    """Key for each beat (quarter note), from the window centered on that beat."""
    n_beats = max(1, round(total_beats))
    window = min(window, n_beats)
    wa = windowed.WindowedAnalysis(s, discrete.KrumhanslSchmuckler())
    solutions, _, _ = wa.process(window, window, windowStepSize=1, includeTotalWindow=False)
    keys = [(p.pitchClass, mode) for p, mode, _ in solutions[0]]
    # window i covers beats [i, i + window); assign it to the beat at its center
    return [keys[min(max(0, b - window // 2), len(keys) - 1)] for b in range(n_beats)]


def _keys_to_sections(
    beat_keys: list[tuple[int, str]], total_beats: float, min_beats: float
) -> list[Section]:
    """Per-beat keys -> sections (in beats), absorbing too short sections into the previous one."""
    sections = merge_sections(
        Section(b, b + 1, tonic, mode) for b, (tonic, mode) in enumerate(beat_keys)
    )
    result: list[Section] = []
    for sec in sections:
        if result and sec.end - sec.start < min_beats:
            last = result[-1]
            result[-1] = Section(last.start, sec.end, last.tonic, last.mode)
        else:
            result.append(sec)
    result = merge_sections(result)
    # a short first section is absorbed into the next one
    if len(result) > 1 and result[0].end - result[0].start < min_beats:
        first, second = result[0], result[1]
        result[:2] = [Section(first.start, second.end, second.tonic, second.mode)]
    last = result[-1]
    result[-1] = Section(last.start, total_beats, last.tonic, last.mode)
    return result
