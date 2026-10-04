import numpy as np
import pytest

from dynamic_tuning.analysis import (
    Profile,
    dissonance_profile,
    note_name,
    plot_group_heatmap,
    plot_pair_heatmap,
    plot_profiles,
    save,
    segment_dissonance,
    summary_table,
)
from dynamic_tuning.progression import chord_sequence, progression_score
from dynamic_tuning.spectrum import Timbre, resolve
from dynamic_tuning.tuning import EQUAL, JUST_C, JUST_KEY


@pytest.fixture(scope="module")
def score():
    # two keys (C and F) of the progression
    return progression_score(chord_sequence(tonics=[0, 5], final_tonic_chord=False))


@pytest.fixture(scope="module")
def profiles(score):
    return {
        t.name: dissonance_profile(resolve(score, t, Timbre(n_partials=6)), name=t.name)
        for t in [EQUAL, JUST_C, JUST_KEY]
    }


def test_profile_shape(score, profiles) -> None:
    p = profiles["12-TET"]
    assert len(p.value) == len(p.start) == len(p.chords) == len(p.keys)
    assert p.start[0] == 0 and p.end[-1] == pytest.approx(score.duration)
    assert np.all(p.value > 0)
    assert p.chords[0] == "C: I" and p.keys[0] == "C" and p.qualities[0] == "major triad"
    assert p.keys[-1] == "F"


def test_profile_mean_is_duration_weighted() -> None:
    p = Profile("x", np.array([0.0, 1.0]), np.array([1.0, 4.0]), np.array([1.0, 2.0]),
                ["C: I", "C: V"], ["", ""], ["C", "C"])  # fmt: skip
    assert p.mean() == pytest.approx((1 * 1 + 2 * 3) / 4)
    assert p.group_means("figure") == {"I": 1.0, "V": 2.0}
    assert p.group_means("key") == {"C": pytest.approx(7 / 4)}


def test_key_relative_ji_beats_fixed_ji_outside_c(profiles) -> None:
    in_f = {name: p.group_means("key")["F"] for name, p in profiles.items()}
    in_c = {name: p.group_means("key")["C"] for name, p in profiles.items()}
    assert in_c["JI (key)"] == pytest.approx(in_c["JI (C)"])
    assert in_f["JI (key)"] < in_f["JI (C)"]
    assert in_c["JI (C)"] < in_c["12-TET"]


def test_major_triad_smoother_in_ji(profiles) -> None:
    triad_et = profiles["12-TET"].group_means("figure")["I"]
    triad_ji = profiles["JI (key)"].group_means("figure")["I"]
    assert triad_ji < triad_et


def test_normalized_dissonance(score) -> None:
    spec = resolve(score, EQUAL, Timbre(n_partials=4))
    seg = spec.segments[0]
    m = "sethares1993"  # amplitude-bilinear model
    raw = segment_dissonance(seg, m)
    norm = segment_dissonance(seg, m, normalize=True)
    assert 0 < norm < 1
    doubled = type(seg)(seg.start, seg.end, seg.notes, seg.freqs, 2 * seg.amps)
    assert segment_dissonance(doubled, m) == pytest.approx(4 * raw)
    assert segment_dissonance(doubled, m, normalize=True) == pytest.approx(norm)


def test_summary_table(profiles) -> None:
    table = summary_table(profiles, by="figure")
    lines = table.splitlines()
    assert lines[0] == "| figure | 12-TET | JI (C) | JI (key) |"
    assert any(line.startswith("| V7 |") for line in lines)
    assert lines[-1].startswith("| (all) |")


def test_note_name() -> None:
    assert note_name(60) == "C4"
    assert note_name(70) == "Bb4"


def test_plots(tmp_path, score, profiles) -> None:
    save(plot_profiles(profiles, title="test", chord_labels=True), str(tmp_path / "p.png"))
    save(plot_group_heatmap(profiles, relative_to="12-TET"), str(tmp_path / "g.png"))
    save(plot_group_heatmap(profiles, by="key"), str(tmp_path / "k.png"))
    spec = resolve(score, EQUAL, Timbre(n_partials=6))
    save(plot_pair_heatmap(spec.segments[3], score, title="seg"), str(tmp_path / "h.png"))
    assert all((tmp_path / f).stat().st_size > 1000 for f in ["p.png", "g.png", "k.png", "h.png"])
