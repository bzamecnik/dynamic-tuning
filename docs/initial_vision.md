# Initial vision

## Prompt

Let's discuss a bit about musical tuning, consonance and dissonance. The most common class of musical instruments are (approximately) harmonic oscillators, so one tone consists of several integer multiples of a base frequency. When multiple tones are played simultaneously as chords, there are many overlapping harmonic sequences which make up the whole spectrum. When a pair of sinusoids come close together, they can interfere, which results in "beating" (amplitude envelope) at the difference of their frequencies. Thus chords with beating pairs sound more dissonant, compared to those where the pairs are spaced furrher apart. Harmonic oscillators lead to diatonic scales. Tuning is assigning actual frequencies to abstract scale notes. This is full of tradeoffs. Given the assumption the tuning is fixed (as for a physical instrument), we can optimize for good consonance in a single scale, while leaving other scales dissonant, or on the other end we can distribute the errors evenly across all scales (as with equal temperament), or we can distribute it unevenly (such as as in Kirnberger III), where each scale sounds differently.

With electronic or digital synthesizers (such as conputer based), we can remove the assumption of fixed tuning and tune dynamically based on the content being played (in real-time or off-line in advance). On one hand we can just adjust the base frequencies and keep the harmonic ratios integer. If I understand correctly, this is done in Harmode. But we could go further and even adjust the individual harmonics.

I could imagine this as a basic gradient descent optimization, with harmonic frequency multipliers as parameters and a loss consisting terms for overall dissonance plus distance to some base temperament (~ L2 regularization) and temporal continuity.

In the simplest case, we may just want to reduce dissonance. But then some chords may lose their character, so maybe we may want some level of dissonance for some chord class. Or the target level of dissonance could be a parameter to be controlled by the user.

What do you think about this concept? Has anyone tried the variant with dynamic harmonics?

I'm imagining that you could play a synth piano in any key and it would still sound bright as if optimized for that key.

What may be some practical way to test this idea? I could imagine numpy yo get started with some static samples quickly, but it would be limiting soon. Can some open-source synthesizer be used where it's possible to control individual harmonics? How about Strudel?

Could you try to formalize the optimization?

Many years ago I implemented a few models of dissonance in a Python package, published as `dissonant` on PyPI. It could likely serve within the loss function.

In order to get some objective view on the various tunings it may be good to be able to measure and compare the dissonance profile of a musical piece over time.

## Gemini 3.1 Pro

It is a fascinating and structurally profound idea. You are describing a real-time, polyphonic **Adaptive Timbre / Spectral Tuning** system.

The concept of dynamically adapting tunings or partials based on the played chord to maximize consonance was pioneered by William Sethares in the 1990s, culminating in his book *Tuning, Timbre, Spectrum, Scale*. While "adaptive tuning" (like Hermode tuning, which adjusts the fundamental frequencies of notes to pure integer ratios while keeping harmonics rigid) is somewhat common, **adaptive timbre**—where the individual partials of a synthesizer bend and morph to minimize dissonance against the partials of other sounding notes—is much rarer and highly experimental.

It is brilliant that you authored the `dissonant` package on PyPI. Having the Plomp-Levelt, Sethares, and Vassilakis models already implemented gives you the exact perceptual distance metric needed for the loss function.

Here is a breakdown of how we can formalize this optimization, along with practical ways to build and test it.

### Formalizing the Optimization

Let’s define the state of the synthesizer at time $t$. You are playing a set of notes $N$.
For each note $n \in N$, there is a base fundamental frequency $F_n$ (e.g., from Equal Temperament).
The note consists of $K$ partials. We introduce an array of harmonic multipliers $\mathbf{h}_n = [h_{n,1}, h_{n,2}, \dots, h_{n,K}]$.
For a perfectly harmonic oscillator, $h_{n,k} = k$.

The actual frequency of the $k$-th partial of note $n$ is:


$$f_{n,k} = F_n \cdot h_{n,k}$$


Let $a_{n,k}$ be the amplitude of that partial.

We want to find the optimal multipliers $\mathbf{h}_{n}$ for all active notes to minimize a composite loss function $L$:

$$L = w_{diss} L_{diss} + w_{reg} L_{reg} + w_{smooth} L_{smooth}$$

**1. Dissonance Loss ($L_{diss}$):**
Using your `dissonant` package (e.g., the Sethares or Vassilakis model), we sum the pairwise dissonance across all active partials in the current chord.


$$L_{diss} = \sum_{n, m \in N} \sum_{k, j \in K} D(f_{n,k}, f_{m,j}, a_{n,k}, a_{m,j})$$


*(You can introduce a user-controlled target dissonance by squaring the error against a target value: $L_{diss} = (\text{Current\_Dissonance} - \text{Target\_Dissonance})^2$)*.

**2. Regularization / Timbre Preservation ($L_{reg}$):**
If we only minimize dissonance, the partials might drift endlessly or collapse into unisons, destroying the "piano" or "synth" character. We apply an $L_2$ penalty to keep the multipliers close to their harmonic ideals (or a base inharmonic profile like a stiff piano string).


$$L_{reg} = \sum_{n \in N} \sum_{k=1}^K (h_{n,k} - k)^2$$

**3. Temporal Continuity ($L_{smooth}$):**
To prevent partials from glitching or jumping discontinuously when a new note is struck, we penalize the rate of change of the multipliers.


$$L_{smooth} = \sum_{n \in N} \sum_{k=1}^K \left( \frac{d h_{n,k}}{dt} \right)^2$$

### Practical Ways to Test and Build This

**1. Offline Proof-of-Concept (Python / JAX / PyTorch)**
Because you want to optimize this via gradient descent, standard NumPy might be slow for autograd. If you port the mathematical core of your `dissonant` package to **JAX** or **PyTorch**, you get automatic differentiation for free.

* **Workflow:** Input a MIDI file $\rightarrow$ chunk it into time windows $\rightarrow$ run gradient descent on the $\mathbf{h}$ parameters for each window using the loss function $\rightarrow$ synthesize the resulting audio using `scipy.io.wavfile` by generating the sine waves with the optimized time-varying frequencies.

**2. Real-Time Engines (SuperCollider or Csound)**
For real-time testing, standard VST synths will not work because they do not expose per-partial frequency control.

* **SuperCollider** is arguably the best tool for this. You can easily write an additive synthesizer `SynthDef` that consists of a bank of `SinOsc` unit generators. You can calculate the target partial frequencies in Python (using a live MIDI feed and your `dissonant` loss function) and send the optimized $f_{n,k}$ values over UDP/OSC (Open Sound Control) directly to the SuperCollider server to modulate the partials in real-time.

**3. Web Audio API / Strudel**
Strudel is excellent for sequencing and runs entirely in the browser using the Web Audio API.

* The standard Web Audio API has an `OscillatorNode`. You can build a custom additive synth patch in Strudel or vanilla JS by spawning banks of oscillators.
* However, running live gradient descent in JavaScript might be a bottleneck. If you use an `AudioWorklet`, you could implement a lightweight gradient descent solver in WebAssembly (Rust/C++) or plain JS, operating on the frequencies right before they are sent to the oscillators.

### Profiling a Piece Over Time

To get an objective view of various tunings, your idea of measuring a dissonance profile over time is highly practical. You could take an audio rendering of a Bach Chorale or a jazz progression and run it through a sliding-window Short-Time Fourier Transform (STFT).

1. Extract the spectral peaks (frequencies and amplitudes) for each time frame.
2. Feed those peaks into the `dissonant.dissonance()` function.
3. Plot the resulting 1D time-series.

You could compare the exact same MIDI performance rendered in Equal Temperament, Just Intonation, Kirnberger III, and your "Dynamic Harmonic" tuning. Visually graphing the dissonance curve over time would beautifully demonstrate whether your gradient descent algorithm is successfully carving out a more consonant path through the harmonic landscape.