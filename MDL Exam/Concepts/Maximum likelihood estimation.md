---
tags: [MDL, concept]
aliases: []
---
# Maximum likelihood estimation

Choose the parameters that make the observed data most probable. With i.i.d. data the likelihood is a product, and the log turns it into a sum.

$$\theta_{ML}=\arg\max_\theta\sum_{i=1}^m\log p_{model}(x^{(i)};\theta)$$

**Lectures:** [[DL 02 Loss functions and maximum likelihood]] · [[ML 02 Density-based classification]] · [[ML 03 Linear classifiers]]

**Related:** [[KL divergence]] · [[Cross-entropy]] · [[Logistic regression]]
