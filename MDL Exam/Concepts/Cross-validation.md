---
tags: [MDL, concept]
aliases: []
---
# Cross-validation

Rotate which part of the data is the test set, train on the rest, and average the errors. Leave-one-out is the extreme with one object per fold.

$$\hat\varepsilon=\frac1n\sum_{i=1}^n\hat\varepsilon_i$$

**Lectures:** [[ML 05 Evaluation]]

**Related:** [[Bootstrapping]] · [[True error and apparent error]]
