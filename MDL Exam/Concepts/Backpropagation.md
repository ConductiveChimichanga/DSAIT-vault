---
tags: [MDL, concept]
aliases: []
---
# Backpropagation

Algorithm that computes the gradient of the loss with respect to every parameter by applying the chain rule backwards through the computational graph, re-using shared terms.

$$\bar n_i=\sum_{n_j\in\text{Children}(n_i)}\bar n_j\frac{\partial n_j}{\partial n_i}$$

**Lectures:** [[DL 03 Backpropagation]] · [[ML 04 Nonlinear classifiers]]

**Related:** [[Chain rule]] · [[Computational graph]] · [[Bar notation]] · [[Gradient descent]]
