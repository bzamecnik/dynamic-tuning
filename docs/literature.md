# Literature

It seems there's already a body of literature dealing with this problem and exploring similar ways to solving it. 

- [Adaptive tunings for musical scales, W. Sethares, 1994](https://www.semanticscholar.org/paper/Adaptive-tunings-for-musical-scales-Sethares/a01af68d3ab3a241ea2c0c13ebe2662bf2bae152)
- [Real-Time Adaptive Tunings Using Max, W. Sethares, 2002](https://www.semanticscholar.org/paper/Real-Time-Adaptive-Tunings-Using-Max-Sethares/5ba41f07f34043d7b023c681a6e44fb9707b2c96)
- [Roughness Minimization Through Automatic Intonation Adjustments, J. Villegas, Michael Cohen, 2010](https://www.semanticscholar.org/paper/Roughness-Minimization-Through-Automatic-Intonation-Villegas-Cohen/8d095bbecefffc2544f1327e8d07e9519078becd)
- [Reducing Dissonance with Dynamic Tuning Algorithms for MIDI Synthesis, K. Kirsch, 2021](https://www.semanticscholar.org/paper/Reducing-Dissonance-with-Dynamic-Tuning-Algorithms-Kirsch/49c06436202c6157dcf058b189c1e37beab6b4a7)
- [A Differentiable Cost Measure for Intonation Processing in Polyphonic Music, Simon J. Schwär, Sebastian Rosenzweig, Meinard Müller, 2021](https://www.semanticscholar.org/paper/A-Differentiable-Cost-Measure-for-Intonation-in-Schw%C3%A4r-Rosenzweig/9fc44843f255bed21641823f7c13e87a3ca15a08)
- [Multi-Voice Intonation Adaptation via Gradient Descent, Simon J. Schwär, Meinard Müller, 2026](https://www.semanticscholar.org/paper/Multi-Voice-Intonation-Adaptation-via-Gradient-Schw%C3%A4r-M%C3%BCller/5b264604674ab974e37d07d9118c4f1796309ff2)

## Consonance / roughness models

- [Simultaneous consonance in music perception and composition, P. Harrison, M. Pearce, Psychological Review 2020](https://cms.mus.cam.ac.uk/publications/harrison-pearce-simultaneous-consonance/): consonance is a composite of interference (roughness), periodicity/harmonicity and cultural familiarity. Of the roughness models, Hutchinson & Knopoff (1978) performed best. Reference implementations are in the R packages [incon](https://github.com/pmcharrison/incon) and [dycon](https://github.com/pmcharrison/dycon) (Hutchinson & Knopoff, Sethares, Vassilakis).
- [Timbral effects on consonance disentangle psychoacoustic mechanisms and suggest perceptual origins for musical scales, R. Marjieh, P. Harrison, H. Lee, F. Deligiannaki, N. Jacoby, Nature Communications 2024](https://www.nature.com/articles/s41467-024-45812-z): large online experiments with stretched/compressed timbres. Consonance peaks follow the timbre, in line with interference (roughness) models and not with harmonicity models. Directly relevant to dynamic timbre.
- [Register impacts perceptual consonance through roughness and sharpness, Wang et al. (Psychonomic Bulletin & Review 2021)](https://link.springer.com/article/10.3758/s13423-021-02033-5).
- [Optimising Computational Measures from Behavioural Data Predicts Perceived Consonance, TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.243): fitting model parameters to behavioural data.
- Vassilakis (2001) roughness model, formula: [SRA roughness estimation model](http://www.acousticslab.org/learnmoresra/moremodel.html).
