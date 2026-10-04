import numpy as np
import pytest
import soundfile as sf

from dynamic_tuning.score import Note, Score, Section
from dynamic_tuning.spectrum import SINE, Timbre, resolve
from dynamic_tuning.synth import Envelope, _pan_gains, _smooth, render, synthesize, write_wav
from dynamic_tuning.tuning import EQUAL, JUST_C, JUST_KEY

SR = 8000


def peak_freq(x: np.ndarray, sr: int) -> float:
    """Frequency of the strongest FFT peak, refined by parabolic interpolation."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=8 * len(x)))
    i = int(spectrum.argmax())
    a, b, c = np.log(spectrum[i - 1 : i + 2])
    i_ref = i + 0.5 * (a - c) / (a - 2 * b + c)
    return i_ref * sr / (8 * len(x))


def test_envelope_shape() -> None:
    env = Envelope(attack=0.1, decay=0.1, sustain=0.5, release=0.2)(1.0, 1000)
    assert len(env) == 1200
    assert env[0] == 0 and env[100] == pytest.approx(1.0)
    assert env[500] == pytest.approx(0.5)
    assert env[-1] == pytest.approx(0.5 / 200, abs=1e-6)
    assert np.all(env >= 0) and np.all(env <= 1)


def test_envelope_short_note() -> None:
    env = Envelope(attack=0.1, decay=0.1, sustain=0.5, release=0.1)(0.05, 1000)
    assert len(env) == 150
    assert env.max() <= 0.5 + 1e-9  # never reaches full attack


def test_smooth_makes_ramps() -> None:
    x = np.array([0.0] * 10 + [1.0] * 10)
    y = _smooth(x, 4)
    assert len(y) == len(x)
    assert y[0] == 0 and y[-1] == 1
    assert np.all(np.diff(y) >= 0)
    assert 0 < y[9] < 1


def test_pan_gains() -> None:
    left, right = _pan_gains(0, [0], 0.6)
    assert left == pytest.approx(right)
    left, right = _pan_gains(0, [0, 1], 1.0)
    assert left == pytest.approx(1.0) and right == pytest.approx(0, abs=1e-12)
    assert left**2 + right**2 == pytest.approx(1)


@pytest.mark.parametrize(
    "pitch,tuning,expected",
    [(69, EQUAL, 440.0), (64, JUST_C, 261.6256 * 5 / 4), (67, JUST_C, 261.6256 * 3 / 2)],
)
def test_note_frequency_fft(pitch: int, tuning, expected: float) -> None:
    audio = render(Score([Note(0, 1.0, pitch)]), tuning, SINE, sample_rate=SR)
    assert peak_freq(audio[: SR // 2 + SR // 4], SR) == pytest.approx(expected, rel=1e-4)


def test_harmonic_partials_present() -> None:
    audio = render(Score([Note(0, 1.0, 57)]), EQUAL, Timbre(n_partials=3), sample_rate=SR)
    spectrum = np.abs(np.fft.rfft(audio[:SR]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    for f in [220, 440, 660]:
        assert spectrum[np.argmin(np.abs(freqs - f))] > 0.1 * spectrum.max()
    assert spectrum[np.argmin(np.abs(freqs - 880))] < 0.01 * spectrum.max()


def test_partials_above_nyquist_are_dropped() -> None:
    audio = render(Score([Note(0, 0.5, 100)]), EQUAL, Timbre(n_partials=4), sample_rate=SR)
    # pitch 100 ~ 2637 Hz; only the first partial is below 0.95 * 4000 Hz
    pure = render(Score([Note(0, 0.5, 100)]), EQUAL, SINE, sample_rate=SR)
    corr = np.dot(audio, pure) / np.linalg.norm(audio) / np.linalg.norm(pure)
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_output_shape_and_level() -> None:
    score = Score([Note(0, 0.5, 60), Note(0.25, 0.5, 64, voice=1)])
    env = Envelope(release=0.1)
    audio = render(score, EQUAL, sample_rate=SR, envelope=env)
    assert audio.shape == (round(0.85 * SR) + 1,)
    assert 0.1 < np.abs(audio).max() <= 1.0
    stereo = render(score, EQUAL, sample_rate=SR, envelope=env, stereo=True)
    assert stereo.shape == (len(audio), 2)
    assert not np.allclose(stereo[:, 0], stereo[:, 1])


def test_gain_independent_of_tuning() -> None:
    # (chords would differ in RMS: coinciding partials in JI add up coherently)
    score = Score([Note(0, 0.5, 64), Note(0.5, 0.5, 69)])
    a = render(score, EQUAL, sample_rate=SR)
    b = render(score, JUST_C, sample_rate=SR)
    assert np.sqrt(np.mean(a**2)) == pytest.approx(np.sqrt(np.mean(b**2)), rel=0.01)


def test_held_note_retunes_without_clicks() -> None:
    """A held note crossing a key change glides to its new frequency, phase-continuously."""
    score = Score([Note(0, 2.0, 60)], sections=[Section(0, 1, 0), Section(1, 2, 7)])
    spec = resolve(score, JUST_KEY, SINE)
    audio = synthesize(spec, SR, envelope=Envelope(release=0.05), glide=0.05)
    f1 = peak_freq(audio[int(0.2 * SR) : int(0.9 * SR)], SR)
    f2 = peak_freq(audio[int(1.1 * SR) : int(1.9 * SR)], SR)
    assert f1 == pytest.approx(spec.segments[0].freqs[0, 0], rel=1e-4)
    assert f2 == pytest.approx(spec.segments[1].freqs[0, 0], rel=1e-4)
    # no discontinuity: sample-to-sample change bounded by the max slope of a sine
    max_step = 2 * np.pi * max(f1, f2) / SR * np.abs(audio).max()
    assert np.abs(np.diff(audio)).max() <= 1.05 * max_step


def test_write_wav(tmp_path) -> None:
    audio = render(Score([Note(0, 0.2, 60)]), EQUAL, sample_rate=SR)
    path = tmp_path / "x.wav"
    write_wav(path, audio, SR)
    data, sr = sf.read(path)
    assert sr == SR and len(data) == len(audio)
    assert np.allclose(data, audio, atol=1e-5)
