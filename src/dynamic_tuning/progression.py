"""Test progression: one chord pattern cycled through all 12 keys by fourths, with voice leading.

Chord pitch classes come from music21 Roman numerals. Voicings are chosen by a
minimal-motion search (Viterbi over all valid voicings of each chord).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from music21 import key as m21key
from music21 import roman

from .score import ChordLabel, Note, Score, Section, key_name

# I vi ii7 V7 Imaj7 viiø7 I7; the final I7 (= V7/IV) is V7 of the next key in the cycle.
DEFAULT_PATTERN = ("I", "vi", "ii7", "V7", "IM7", "viiø7", "V7/IV")

# MIDI ranges per voice, from the lowest (bass) to the highest (soprano).
DEFAULT_RANGES = ((40, 60), (48, 67), (55, 74), (60, 79))

_M21_TONIC_NAMES = ("C", "D-", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")


@dataclass(frozen=True)
class Chord:
    figure: str  # Roman numeral
    tonic: int  # key tonic pitch class
    pitch_classes: tuple[int, ...]  # chord tones, bass first
    quality: str

    @property
    def bass(self) -> int:
        return self.pitch_classes[0]

    @property
    def key(self) -> str:
        return key_name(self.tonic)


def make_chord(figure: str, tonic: int) -> Chord:
    rn = roman.RomanNumeral(figure, m21key.Key(_M21_TONIC_NAMES[tonic % 12]))
    pcs = [p.pitchClass for p in rn.pitches]
    bass = rn.bass().pitchClass
    ordered = [bass, *(pc for pc in dict.fromkeys(pcs) if pc != bass)]
    return Chord(figure, tonic % 12, tuple(ordered), rn.commonName)


def key_cycle(start: int = 0, step: int = 5) -> list[int]:
    """Tonics of all 12 keys; step 5 semitones = cycle of fourths (C F Bb ... G)."""
    return [(start + i * step) % 12 for i in range(12)]


def chord_sequence(
    pattern: tuple[str, ...] = DEFAULT_PATTERN,
    tonics: list[int] | None = None,
    final_tonic_chord: bool = True,
) -> list[Chord]:
    tonics = key_cycle() if tonics is None else tonics
    chords = [make_chord(fig, t) for t in tonics for fig in pattern]
    if final_tonic_chord:
        chords.append(make_chord("I", tonics[0]))
    return chords


# --- voice leading ---


def voicings(
    chord: Chord,
    ranges: tuple[tuple[int, int], ...] = DEFAULT_RANGES,
    max_upper_spacing: int = 12,
    max_bass_spacing: int = 19,
) -> np.ndarray:
    """All valid voicings of a chord as an array (n_voicings, n_voices), bass first.

    Rules: bass plays the chord's bass; voices strictly ascend (no crossing, no unisons);
    all chord tones are present if there are enough voices; adjacent upper voices are
    at most an octave apart.
    """
    n_voices = len(ranges)
    pcs = set(chord.pitch_classes)
    options = [
        [p for p in range(lo, hi + 1) if p % 12 in pcs and (i > 0 or p % 12 == chord.bass)]
        for i, (lo, hi) in enumerate(ranges)
    ]
    must_cover = len(pcs) <= n_voices
    result = []
    for v in itertools.product(*options):
        if any(v[i] >= v[i + 1] for i in range(n_voices - 1)):
            continue
        if n_voices > 1 and v[1] - v[0] > max_bass_spacing:
            continue
        if any(v[i + 1] - v[i] > max_upper_spacing for i in range(1, n_voices - 1)):
            continue
        if must_cover and {p % 12 for p in v} != pcs:
            continue
        result.append(v)
    if not result:
        raise ValueError(f"No valid voicing for {chord} in ranges {ranges}")
    return np.array(result, dtype=int)


def _parallels(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Count of parallel fifths/octaves for all pairs of voicings: (len(prev), len(nxt))."""
    n_voices = prev.shape[1]
    count = np.zeros((len(prev), len(nxt)), dtype=int)
    for i, j in itertools.combinations(range(n_voices), 2):
        iv_prev = ((prev[:, j] - prev[:, i]) % 12)[:, None]
        iv_next = ((nxt[:, j] - nxt[:, i]) % 12)[None, :]
        moved = (prev[:, i][:, None] != nxt[:, i][None, :]) & (
            prev[:, j][:, None] != nxt[:, j][None, :]
        )
        perfect = (iv_prev == iv_next) & ((iv_prev == 0) | (iv_prev == 7))
        count += perfect & moved
    return count


def voice_lead(
    chords: list[Chord],
    ranges: tuple[tuple[int, int], ...] = DEFAULT_RANGES,
    parallel_penalty: float = 6.0,
    center_weight: float = 0.05,
) -> np.ndarray:
    """Minimal-motion voicing of a chord sequence; returns (n_chords, n_voices) MIDI pitches.

    Cost = total semitone motion of all voices + penalty per parallel fifth/octave
    + small pull towards the middle of each voice's range (keeps the register stable).
    Deterministic: ties are broken by the order of the voicings.
    """
    centers = np.array([(lo + hi) / 2 for lo, hi in ranges])
    cands = [voicings(c, ranges) for c in chords]
    node_cost = [center_weight * np.abs(c - centers).sum(axis=1) for c in cands]

    cost = node_cost[0]
    back = []
    for prev, nxt, nc in zip(cands, cands[1:], node_cost[1:], strict=False):
        motion = np.abs(prev[:, None, :] - nxt[None, :, :]).sum(axis=2)
        trans = motion + parallel_penalty * _parallels(prev, nxt)
        total = cost[:, None] + trans
        best = total.argmin(axis=0)
        back.append(best)
        cost = total[best, np.arange(len(nxt))] + nc

    idx = [int(cost.argmin())]
    for b in reversed(back):
        idx.append(int(b[idx[-1]]))
    idx.reverse()
    return np.array([c[i] for c, i in zip(cands, idx, strict=True)])


# --- score generation ---


def progression_score(
    chords: list[Chord] | None = None,
    voicing: np.ndarray | None = None,
    tempo: float = 120.0,
    beats_per_chord: float = 4.0,
    velocity: int = 80,
    tie_common_tones: bool = True,
) -> Score:
    """Build a score: one chord per bar, voices numbered from the bass (0) upwards.

    With `tie_common_tones`, a voice repeating its pitch holds the note instead of
    re-striking it, so the note keeps its identity across chords.
    Sections are created per key (consecutive chords sharing a tonic).
    """
    chords = chord_sequence() if chords is None else chords
    voicing = voice_lead(chords) if voicing is None else voicing
    dur = beats_per_chord * 60.0 / tempo

    notes: list[Note] = []
    for v in range(voicing.shape[1]):
        current: Note | None = None
        for i in range(len(chords)):
            pitch = int(voicing[i, v])
            if tie_common_tones and current is not None and current.pitch == pitch:
                current = Note(current.start, current.duration + dur, pitch, v, velocity)
            else:
                if current is not None:
                    notes.append(current)
                current = Note(i * dur, dur, pitch, v, velocity)
        if current is not None:
            notes.append(current)

    labels = [
        ChordLabel(i * dur, (i + 1) * dur, figure=c.figure, key=c.key, quality=c.quality)
        for i, c in enumerate(chords)
    ]
    sections: list[Section] = []
    for i, c in enumerate(chords):
        if sections and sections[-1].tonic == c.tonic:
            last = sections[-1]
            sections[-1] = Section(last.start, (i + 1) * dur, last.tonic, last.mode)
        else:
            sections.append(Section(i * dur, (i + 1) * dur, c.tonic, "major"))
    return Score(notes=notes, sections=sections, chords=labels, tempo=tempo).sorted()


def chord_table(chords: list[Chord], voicing: np.ndarray) -> str:
    """Plain-text table of chords and their voicings (bass first)."""
    from .score import PITCH_CLASS_NAMES

    def name(p: int) -> str:
        return f"{PITCH_CLASS_NAMES[p % 12]}{p // 12 - 1}"

    rows = [f"{'#':>3}  {'key':<3} {'figure':<7} {'quality':<30} voicing"]
    for i, (c, v) in enumerate(zip(chords, voicing, strict=True)):
        rows.append(
            f"{i:>3}  {c.key:<3} {c.figure:<7} {c.quality:<30} {' '.join(name(p) for p in v)}"
        )
    return "\n".join(rows)
