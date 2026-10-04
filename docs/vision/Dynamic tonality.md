# Dynamic tonality

2017-05-09

<https://en.wikipedia.org/wiki/Dynamic_tonality>

<http://www.dynamictonality.com/>

* dynamic tuning - tuning of base frequencies for the whole key
* dynamic intonation - changing base frequencies
* dynamic timbre - changing frequencies of individual partials

Dynamic tuning and dynamic intonation can be treated together - just changing base frequencies.

What should be the measure of dissonance?

* standard models - total roughness of pairs of sinusoids (Sethares, Vassilakis, Cook, ...)
* a model based on amplitude modulations of the raw waveform

We could express dissonance as loss function, compute its gradient with respect of the offset of each partial and apply gradient-based optimization.

We could separately move base frequency and and partials or just partials.

Constraints:

* the base frequences of tones should not bo too far from the tuning
* the partials should not be too far for the ideal harmonics

Goal: Decrease dissonance of an instrument audio recording by modifying its runing and intonation dynamically.

Motivation:

Dissonance occurs when two partials of a harmonic tone get close to each other within some range. The interference of a pair of sinusoids causes amplitude modulation - low frequency beating. Beating at some frequency range can be perceived by humans as dissonance. For complex tone with N partials in total there are N^2 pairs that can add up to the total dissonance. The resulting sound may be perceived as dull, untuned or harsh.

A few things can be done to overcome this:

\- change the intonation (the base frequencies of harmonic tones) dynamically
\- change the timbre (especially frequencies of each tone partials, possibly amplitude as well)
  - Changing the amplitude a lot might affect the queality of the instrument sound. We could possibly mute some offending partials that interfere a lot and cannot be moved well.

A capella choirs can change intonation dynamically and thus sound really bright. But they cannot change the timbre frequencies.

A piano due to its construction has fixed both intonation and timbre (except from unintentional detuning and timbre fluctuations due to dynamics of metalic strings). A piano is typically tuned to equal temperament tuning...
