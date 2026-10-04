import pytest

from dynamic_tuning.score import (
    ChordLabel,
    Note,
    Score,
    Section,
    from_midi,
    key_name,
    merge_sections,
    read_midi,
    to_midi,
    write_midi,
)


def make_score() -> Score:
    notes = [
        Note(0.0, 1.0, 60, voice=0, velocity=90),
        Note(0.0, 2.0, 64, voice=1),
        Note(1.0, 1.0, 67, voice=0),
        Note(2.0, 0.5, 69, voice=2, velocity=50),
    ]
    sections = [Section(0.0, 1.5, 0, "major"), Section(1.5, 2.5, 9, "minor")]
    chords = [
        ChordLabel(0.0, 1.0, "I", "C", "major triad"),
        ChordLabel(1.0, 2.5, "V7", "C", "dominant seventh chord"),
    ]
    return Score(notes, sections, chords, tempo=100.0)


def test_note_end() -> None:
    assert Note(1.5, 2.0, 60).end == 3.5


def test_score_properties() -> None:
    score = make_score()
    assert score.duration == 2.5
    assert score.voices == [0, 1, 2]


def test_lookup_section_and_chord() -> None:
    score = make_score()
    s0 = score.section_at(0.2)
    assert s0 is not None and s0.name == "C"
    s1 = score.section_at(2.0)
    assert s1 is not None and s1.name == "a"
    assert score.section_at(3.0) is None
    chord = score.chord_at(1.0)
    assert chord is not None and chord.name == "C: V7"


def test_key_name() -> None:
    assert key_name(10) == "Bb"
    assert key_name(6, "minor") == "f#"


def test_merge_sections() -> None:
    merged = merge_sections(
        [Section(1, 2, 0), Section(0, 1, 0), Section(2, 3, 7), Section(3, 4, 7, "minor")]
    )
    assert merged == [Section(0, 2, 0), Section(2, 3, 7), Section(3, 4, 7, "minor")]


def assert_scores_equal(a: Score, b: Score, tol: float = 1e-3) -> None:
    a, b = a.sorted(), b.sorted()
    assert len(a.notes) == len(b.notes)
    for x, y in zip(a.notes, b.notes, strict=True):
        assert (x.pitch, x.voice, x.velocity) == (y.pitch, y.voice, y.velocity)
        assert x.start == pytest.approx(y.start, abs=tol)
        assert x.duration == pytest.approx(y.duration, abs=tol)
    assert [(s.tonic, s.mode) for s in a.sections] == [(s.tonic, s.mode) for s in b.sections]
    for sa, sb in zip(a.sections, b.sections, strict=True):
        assert (sa.start, sa.end) == pytest.approx((sb.start, sb.end), abs=tol)
    assert [(c.figure, c.key, c.quality) for c in a.chords] == [
        (c.figure, c.key, c.quality) for c in b.chords
    ]
    assert a.tempo == pytest.approx(b.tempo)


def test_midi_round_trip_in_memory() -> None:
    score = make_score()
    assert_scores_equal(score, from_midi(to_midi(score)))


def test_midi_round_trip_file(tmp_path) -> None:
    score = make_score()
    path = tmp_path / "test.mid"
    write_midi(score, path)
    assert_scores_equal(score, read_midi(path))
