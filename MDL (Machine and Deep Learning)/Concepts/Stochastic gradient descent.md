---
tags: [MDL, concept]
aliases: []
---
# Stochastic gradient descent

Gradient descent using the gradient of a small mini-batch as an estimate of the full-data gradient. Cheap steps at the cost of noise.

$$\theta\leftarrow\theta-\epsilon\,\frac1k\sum_{i=1}^k\nabla_\theta L(x^{(i)},y^{(i)},\theta)$$

**Lectures:** [[DL 01 Feed-forward networks and SGD]] · [[DL 04 Optimisers]]

**Related:** [[Gradient descent]] · [[Momentum]] · [[Adam]]
