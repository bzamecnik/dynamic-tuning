import numpy as np
import pytest

from dynamic_tuning.score import Note, Score, Section
from dynamic_tuning.spectrum import SINE, Timbre, resolve, segment
from dynamic_tuning.tuning import EQUAL, JUST_KEY


def make_score() -> Score:
    # voice 0 holds C4 for 2 s; voice 1 plays E4 then G4; key changes at 1.5 s
    notes = [Note(0, 2, 60, 0, 127), Note(0, 1, 64, 1, 127), Note(1, 1, 67, 1, 127)]
    return Score(notes, sections=[Section(0, 1.5, 0), Section(1.5, 2, 7)])


def test_timbre_partials() -> None:
    t = Timbre(n_partials=4, rolloff=1.0)
    freqs, amps = t.partials(100.0)
    assert freqs.shape == (4,)
    assert np.allclose(freqs, [100, 200, 300, 400])
    assert np.allclose(amps, [1, 1 / 2, 1 / 3, 1 / 4])
    freqs, amps = t.partials(np.array([100.0, 150.0]), gain=np.array([1.0, 0.5]))
    assert freqs.shape == amps.shape == (2, 4)
    assert np.allclose(amps[1], 0.5 * amps[0])


def test_timbre_stretch() -> None:
    t = Timbre(n_partials=3, stretch=0.01)
    assert t.ratios()[0] == 1
    assert np.all(t.ratios()[1:] > [2, 3])


def test_sine_timbre() -> None:
    freqs, amps = SINE.partials(440.0)
    assert freqs.tolist() == [440.0] and amps.tolist() == [1.0]


def test_segment() -> None:
    segs = segment(make_score())
    assert [(s.start, s.end) for s in segs] == [(0, 1), (1, 1.5), (1.5, 2)]
    assert [s.notes for s in segs] == [(0, 1), (0, 2), (0, 2)]


def test_segment_skips_silence() -> None:
    score = Score([Note(0, 1, 60), Note(2, 1, 62)])
    assert [(s.start, s.end) for s in segment(score)] == [(0, 1), (2, 3)]


def test_resolve_equal() -> None:
    spec = resolve(make_score(), EQUAL, Timbre(n_partials=3))
    assert len(spec) == 3
    seg = spec.segments[0]
    assert seg.freqs.shape == (2, 3)
    assert seg.freqs[0, 0] == pytest.approx(EQUAL.frequency(60))
    assert seg.freqs[1, 1] == pytest.approx(2 * EQUAL.frequency(64))
    assert np.allclose(seg.amps[:, 0], 1.0)  # velocity 127


def test_resolve_key_relative_retunes_held_note() -> None:
    spec = resolve(make_score(), JUST_KEY, SINE)
    track = spec.note_track(0)  # C4 held across the key change
    assert len(track) == 3
    f_in_c = track[0][0].freqs[track[0][1], 0]
    f_in_g = track[2][0].freqs[track[2][1], 0]
    assert f_in_c == pytest.approx(EQUAL.frequency(60))
    # in G, C is the pure fourth below... above G3: 4/3 * G3 (G at 12-TET)
    assert f_in_g == pytest.approx(EQUAL.frequency(55) * 4 / 3)
    assert f_in_c != pytest.approx(f_in_g)


def test_resolve_adjust_hook() -> None:
    def detune(seg):
        seg.freqs = seg.freqs * 2
        return seg

    spec = resolve(make_score(), EQUAL, SINE, adjust=detune)
    assert spec.segments[0].freqs[0, 0] == pytest.approx(2 * EQUAL.frequency(60))
