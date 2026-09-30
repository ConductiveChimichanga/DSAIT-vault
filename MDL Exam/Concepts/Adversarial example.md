---
tags: [MDL, concept]
aliases: []
---
# Adversarial example

An input with a tiny perturbation along the sign of the loss gradient that the network misclassifies with high confidence.

$$x+\epsilon\operatorname{sign}(\nabla_xJ(\theta,x,y))$$

**Lectures:** [[ML 07 Regularisation]]

**Related:** [[Data augmentation]]
