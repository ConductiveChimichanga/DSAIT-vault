---
tags: [MDL, concept]
aliases: []
---
# L1 regularisation

Penalty on the sum of absolute weights. The pull toward zero is constant, so weights become exactly zero: sparse solutions and feature selection.

$$\tilde J=\alpha\lVert w\rVert_1+J,\qquad\nabla_w\tilde J=\alpha\operatorname{sign}(w)+\nabla_wJ$$

**Lectures:** [[ML 07 Regularisation]]

**Related:** [[Weight decay]] · [[Regularisation]]
