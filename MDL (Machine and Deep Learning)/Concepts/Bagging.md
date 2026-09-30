---
tags: [MDL, concept]
aliases: []
---
# Bagging

Train many classifiers on bootstrap samples (drawn with replacement) and average them. Reduces variance of unstable classifiers.

$$\hat y_{bag}(x)=\frac1M\sum_{m=1}^M\hat y_m(x)$$

**Lectures:** [[ML 04 Nonlinear classifiers]]

**Related:** [[Random forest]] · [[Boosting]] · [[Bootstrapping]]
