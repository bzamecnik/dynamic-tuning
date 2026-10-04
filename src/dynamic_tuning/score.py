"""Symbolic score model: notes, key sections and chord labels, with MIDI I/O.

Time is in seconds. Pitches are MIDI note numbers (12-TET grid); the actual
frequencies come from a tuning, see `dynamic_tuning.tuning`.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pretty_midi

PITCH_CLASS_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MODES = ("major", "minor")

# Prefixes of MIDI text events used to store chord labels.
_CHORD_PREFIX = "chord:"


@dataclass(frozen=True)
class Note:
    start: float
    duration: float
    pitch: int
    voice: int = 0
    velocity: int = 80

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Section:
    """Key annotation for a time span, needed for key-relative tunings."""

    start: float
    end: float
    tonic: int  # pitch class 0-11
    mode: str = "major"

    @property
    def name(self) -> str:
        return key_name(self.tonic, self.mode)


@dataclass(frozen=True)
class ChordLabel:
    """Harmonic annotation, e.g. figure "V7" in key "C", quality "dominant seventh"."""

    start: float
    end: float
    figure: str
    key: str = ""
    quality: str = ""

    @property
    def name(self) -> str:
        return f"{self.key}: {self.figure}" if self.key else self.figure


@dataclass
class Score:
    notes: list[Note]
    sections: list[Section] = field(default_factory=list)
    chords: list[ChordLabel] = field(default_factory=list)
    tempo: float = 120.0  # BPM, informational (times are already in seconds)

    @property
    def duration(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    @property
    def voices(self) -> list[int]:
        return sorted({n.voice for n in self.notes})

    def sorted(self) -> Score:
        return replace(
            self,
            notes=sorted(self.notes, key=lambda n: (n.start, n.voice, n.pitch)),
            sections=sorted(self.sections, key=lambda s: s.start),
            chords=sorted(self.chords, key=lambda c: c.start),
        )

    def section_at(self, time: float) -> Section | None:
        return _span_at(self.sections, time)

    def chord_at(self, time: float) -> ChordLabel | None:
        return _span_at(self.chords, time)


def _span_at[T: (Section, ChordLabel)](spans: list[T], time: float) -> T | None:
    starts = [s.start for s in spans]
    i = bisect.bisect_right(starts, time) - 1
    if i >= 0 and time < spans[i].end:
        return spans[i]
    return None


def key_name(tonic: int, mode: str = "major") -> str:
    name = PITCH_CLASS_NAMES[tonic % 12]
    return name if mode == "major" else name.lower()


# --- MIDI I/O ---


def to_midi(score: Score) -> pretty_midi.PrettyMIDI:
    """One MIDI instrument (track) per voice; sections as key signatures, chords as text events."""
    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=score.tempo)
    for voice in score.voices:
        inst = pretty_midi.Instrument(program=0, name=f"voice {voice}")
        inst.notes = [
            pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
            for n in score.notes
            if n.voice == voice
        ]
        pm.instruments.append(inst)
    for s in score.sections:
        key_number = s.tonic % 12 + (12 if s.mode == "minor" else 0)
        pm.key_signature_changes.append(pretty_midi.KeySignature(key_number, s.start))
    for c in score.chords:
        text = f"{_CHORD_PREFIX}{c.key}|{c.figure}|{c.quality}"
        pm.text_events.append(pretty_midi.Text(text, c.start))
    return pm


def from_midi(pm: pretty_midi.PrettyMIDI) -> Score:
    """Each MIDI instrument becomes a voice. Drum tracks are skipped.

    Sections come from key signatures (each lasts until the next one or the end).
    Chord labels come from our own text events (each lasts until the next one).
    """
    notes = [
        Note(start=n.start, duration=n.end - n.start, pitch=n.pitch, voice=v, velocity=n.velocity)
        for v, inst in enumerate(i for i in pm.instruments if not i.is_drum)
        for n in inst.notes
    ]
    end = max((n.end for n in notes), default=0.0)

    keys = sorted(pm.key_signature_changes, key=lambda k: k.time)
    sections = [
        Section(
            start=k.time,
            end=keys[i + 1].time if i + 1 < len(keys) else end,
            tonic=k.key_number % 12,
            mode="minor" if k.key_number >= 12 else "major",
        )
        for i, k in enumerate(keys)
    ]

    texts = sorted(
        (t for t in pm.text_events if t.text.startswith(_CHORD_PREFIX)), key=lambda t: t.time
    )
    chords = []
    for i, t in enumerate(texts):
        key, figure, quality = t.text.removeprefix(_CHORD_PREFIX).split("|")
        chord_end = texts[i + 1].time if i + 1 < len(texts) else end
        chords.append(ChordLabel(t.time, chord_end, figure=figure, key=key, quality=quality))

    tempo_changes = pm.get_tempo_changes()[1]
    tempo = float(tempo_changes[0]) if len(tempo_changes) else 120.0
    return Score(notes=notes, sections=sections, chords=chords, tempo=tempo).sorted()


def write_midi(score: Score, path: str | Path) -> None:
    to_midi(score).write(str(path))


def read_midi(path: str | Path) -> Score:
    return from_midi(pretty_midi.PrettyMIDI(str(path)))


def merge_sections(sections: Iterable[Section]) -> list[Section]:
    """Merge adjacent sections with the same key."""
    merged: list[Section] = []
    for s in sorted(sections, key=lambda s: s.start):
        if merged and (merged[-1].tonic, merged[-1].mode) == (s.tonic, s.mode):
            merged[-1] = replace(merged[-1], end=s.end)
        else:
            merged.append(s)
    return merged
