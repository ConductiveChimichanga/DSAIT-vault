---
tags: [MDL, concept]
aliases: []
---
# RMSProp

Divides the gradient by the root of a running average of its square, giving each parameter its own step size.

$$r\leftarrow\rho r+(1-\rho)\nabla_\theta^2,\qquad\theta\leftarrow\theta-\epsilon\frac{\nabla_\theta}{\sqrt{r+\delta}}$$

**Lectures:** [[DL 04 Optimisers]]

**Related:** [[Exponentially weighted moving average]] · [[Adam]]
