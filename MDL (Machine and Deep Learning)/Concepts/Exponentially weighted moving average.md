---
tags: [MDL, concept]
aliases: []
---
# Exponentially weighted moving average

A running average that weights recent values more. Bias-corrected by dividing by $1-\rho^t$.

$$S_t=\rho S_{t-1}+(1-\rho)y_t,\qquad\hat S_t=\frac{S_t}{1-\rho^t}$$

**Lectures:** [[DL 04 Optimisers]]

**Related:** [[Momentum]] · [[RMSProp]] · [[Adam]]
