---
tags: [MDL, concept]
aliases: []
---
# Logistic regression

Model the log-odds as linear in $x$; the posterior is a sigmoid of a linear function. Trained by maximum likelihood, which is the same as minimising cross-entropy.

$$p(y_1\mid x)=\frac{1}{1+e^{-(\beta_0+\beta^Tx)}}$$

**Lectures:** [[ML 03 Linear classifiers]] · [[DL 02 Loss functions and maximum likelihood]]

**Related:** [[Sigmoid]] · [[Cross-entropy]] · [[Maximum likelihood estimation]]
