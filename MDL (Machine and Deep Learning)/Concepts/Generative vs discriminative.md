---
tags: [MDL, concept]
aliases: []
---
# Generative vs discriminative

Generative classifiers model $p(x\mid y)p(y)$ and apply Bayes. Discriminative classifiers model $p(y\mid x)$ or the boundary $f(x;w)$ directly. Generative gives densities and outlier detection but needs more data; discriminative is flexible but needs a loss and an optimiser.

**Lectures:** [[ML 02 Density-based classification]] · [[ML 03 Linear classifiers]]

**Related:** [[Plug-in Bayes classifier]] · [[Linear discriminant]]
