---
tags: [MDL, concept]
aliases: []
---
# Adam

Momentum plus RMSProp plus bias correction of both running averages.

$$\theta\leftarrow\theta-\epsilon\frac{\hat v}{\sqrt{\hat r+\delta}},\qquad\hat v=\frac{v}{1-\rho_1^i},\ \hat r=\frac{r}{1-\rho_2^i}$$

**Lectures:** [[DL 04 Optimisers]]

**Related:** [[Momentum]] · [[RMSProp]] · [[Stochastic gradient descent]]
