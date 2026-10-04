import numpy as np
import pytest

from dynamic_tuning.progression import (
    DEFAULT_PATTERN,
    DEFAULT_RANGES,
    _parallels,
    chord_sequence,
    chord_table,
    key_cycle,
    make_chord,
    progression_score,
    voice_lead,
    voicings,
)
from dynamic_tuning.score import from_midi, to_midi


@pytest.fixture(scope="module")
def chords():
    return chord_sequence()


@pytest.fixture(scope="module")
def voicing(chords):
    return voice_lead(chords)


def test_key_cycle_by_fourths() -> None:
    assert key_cycle() == [0, 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7]
    assert sorted(key_cycle()) == list(range(12))


@pytest.mark.parametrize(
    "figure,tonic,expected",
    [
        ("I", 0, (0, 4, 7)),
        ("vi", 0, (9, 0, 4)),
        ("ii7", 0, (2, 5, 9, 0)),
        ("V7", 0, (7, 11, 2, 5)),
        ("IM7", 0, (0, 4, 7, 11)),
        ("viiø7", 0, (11, 2, 5, 9)),
        ("V7/IV", 0, (0, 4, 7, 10)),
        ("V7", 10, (5, 9, 0, 3)),
    ],
)
def test_make_chord(figure: str, tonic: int, expected: tuple[int, ...]) -> None:
    chord = make_chord(figure, tonic)
    assert chord.bass == expected[0]
    assert set(chord.pitch_classes) == set(expected)


def test_last_chord_of_key_is_dominant_of_next() -> None:
    i7 = make_chord("V7/IV", 0)
    v7_of_f = make_chord("V7", 5)
    assert set(i7.pitch_classes) == set(v7_of_f.pitch_classes)


def test_chord_sequence(chords) -> None:
    assert len(chords) == 12 * len(DEFAULT_PATTERN) + 1
    assert {c.tonic for c in chords} == set(range(12))
    assert chords[0].figure == chords[-1].figure == "I"
    assert chords[0].tonic == chords[-1].tonic == 0


def test_voicings_are_valid() -> None:
    chord = make_chord("V7", 0)
    vs = voicings(chord)
    assert len(vs) > 0
    assert np.all(np.diff(vs, axis=1) > 0)  # no crossing, no unison
    assert np.all(vs[:, 0] % 12 == chord.bass)
    for v in vs:
        assert {p % 12 for p in v} == set(chord.pitch_classes)
    for i, (lo, hi) in enumerate(DEFAULT_RANGES):
        assert np.all((vs[:, i] >= lo) & (vs[:, i] <= hi))


def test_voicings_impossible() -> None:
    with pytest.raises(ValueError):
        voicings(make_chord("I", 0), ranges=((61, 62), (63, 63)))


def test_parallels() -> None:
    prev = np.array([[48, 55]])  # C3 G3: fifth
    nxt = np.array([[50, 57], [50, 59], [48, 55]])  # parallel fifth / third / no motion
    assert _parallels(prev, nxt).tolist() == [[1, 0, 0]]


def test_voice_lead_is_smooth_and_in_range(chords, voicing) -> None:
    assert voicing.shape == (len(chords), len(DEFAULT_RANGES))
    motion = np.abs(np.diff(voicing, axis=0))
    assert motion[:, 1:].max() <= 5  # upper voices move at most a fourth
    assert motion.max() <= 7
    for i, (lo, hi) in enumerate(DEFAULT_RANGES):
        assert voicing[:, i].min() >= lo and voicing[:, i].max() <= hi


def test_voice_lead_is_deterministic(chords, voicing) -> None:
    assert np.array_equal(voice_lead(chords), voicing)


def test_voice_lead_prefers_common_tones() -> None:
    # I -> vi in C shares C and E; at least 2 voices should hold
    chords = [make_chord("I", 0), make_chord("vi", 0)]
    v = voice_lead(chords)
    assert (v[0] == v[1]).sum() >= 2


def test_progression_score(chords, voicing) -> None:
    score = progression_score(chords, voicing, tempo=120, beats_per_chord=4)
    assert score.duration == pytest.approx(len(chords) * 2.0)
    assert len(score.chords) == len(chords)
    assert [s.tonic for s in score.sections] == [*key_cycle(), 0]
    assert score.voices == [0, 1, 2, 3]
    # every chord time has exactly 4 sounding notes with the voiced pitches
    for i, label in enumerate(score.chords):
        t = label.start + 0.1
        sounding = sorted(n.pitch for n in score.notes if n.start <= t < n.end)
        assert sounding == sorted(voicing[i].tolist())


def test_progression_tied_vs_restruck(chords, voicing) -> None:
    tied = progression_score(chords, voicing, tie_common_tones=True)
    restruck = progression_score(chords, voicing, tie_common_tones=False)
    assert len(restruck.notes) == voicing.size
    assert len(tied.notes) < len(restruck.notes)


def test_progression_midi_round_trip(chords, voicing) -> None:
    score = progression_score(chords, voicing)
    back = from_midi(to_midi(score))
    assert [(n.pitch, n.voice) for n in back.notes] == [(n.pitch, n.voice) for n in score.notes]
    assert [s.tonic for s in back.sections] == [s.tonic for s in score.sections]
    assert [c.name for c in back.chords] == [c.name for c in score.chords]


def test_chord_table(chords, voicing) -> None:
    table = chord_table(chords, voicing)
    lines = table.splitlines()
    assert len(lines) == len(chords) + 1
    assert "V7/IV" in table and "C4" in table
