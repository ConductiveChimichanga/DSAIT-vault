---
tags: [MDL, concept]
aliases: []
---
# Weight decay

L2 parameter norm penalty. Each update first shrinks the weights by a constant factor. Also called ridge or Tikhonov regularisation.

$$\tilde J=\tfrac\alpha2w^Tw+J,\qquad w\leftarrow(1-\epsilon\alpha)w-\epsilon\nabla_wJ$$

**Lectures:** [[ML 07 Regularisation]]

**Related:** [[L1 regularisation]] · [[Regularisation]] · [[Early stopping]]
