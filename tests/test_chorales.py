import itertools

import pytest

from dynamic_tuning.chorales import _keys_to_sections, list_bach_works, load_chorale
from dynamic_tuning.score import Section


def test_list_bach_works() -> None:
    names = list_bach_works()
    assert "bach/bwv66.6" in names
    assert len(names) == len(set(names)) > 300


def test_keys_to_sections_absorbs_short_sections() -> None:
    keys = [(0, "major")] * 8 + [(7, "major")] * 2 + [(0, "major")] * 4 + [(9, "minor")] * 6
    sections = _keys_to_sections(keys, total_beats=20.0, min_beats=4)
    assert sections == [Section(0, 14, 0, "major"), Section(14, 20.0, 9, "minor")]


def test_keys_to_sections_short_first() -> None:
    keys = [(7, "major")] * 2 + [(0, "major")] * 10
    assert _keys_to_sections(keys, 12.0, 4) == [Section(0, 12.0, 0, "major")]


@pytest.fixture(scope="module")
def chorale():
    return load_chorale("bach/bwv66.6", tempo=60.0)


def test_load_chorale(chorale) -> None:
    assert chorale.voices == [0, 1, 2, 3]
    assert chorale.duration == pytest.approx(36.0)  # 36 quarter notes at 60 BPM
    assert all(n.duration > 0 for n in chorale.notes)

    # soprano above bass on average
    def mean_pitch(v: int) -> float:
        ps = [n.pitch for n in chorale.notes if n.voice == v]
        return sum(ps) / len(ps)

    assert mean_pitch(0) > mean_pitch(3)


def test_chorale_sections_cover_the_piece(chorale) -> None:
    sections = chorale.sections
    assert sections[0].start == 0
    assert sections[-1].end == pytest.approx(chorale.duration)
    for a, b in itertools.pairwise(sections):
        assert a.end == b.start
        assert (a.tonic, a.mode) != (b.tonic, b.mode)
    assert any(s.name == "f#" for s in sections)  # the chorale's main key


def test_chorale_single_key() -> None:
    score = load_chorale("bach/bwv66.6", key_window=None)
    assert [s.name for s in score.sections] == ["f#"]
