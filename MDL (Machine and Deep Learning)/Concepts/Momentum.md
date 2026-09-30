---
tags: [MDL, concept]
aliases: []
---
# Momentum

SGD on a running average of the gradient. Consistent directions build up, oscillating ones cancel.

$$v\leftarrow\rho v+(1-\rho)\nabla_\theta,\qquad\theta\leftarrow\theta-\epsilon v$$

**Lectures:** [[DL 04 Optimisers]]

**Related:** [[Exponentially weighted moving average]] · [[Adam]] · [[Stochastic gradient descent]]
