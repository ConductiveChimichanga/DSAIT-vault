---
tags: [MDL, concept]
aliases: []
---
# Feed-forward network

A chain of layers, each a linear map followed by an element-wise nonlinearity, approximating a target function $y=f(x;\theta)$. The hidden layers learn the features.

$$f(x)=w^T\max\{0,W^Tx+c\}+b$$

**Lectures:** [[DL 01 Feed-forward networks and SGD]] · [[ML 04 Nonlinear classifiers]]

**Related:** [[Activation function]] · [[ReLU]] · [[XOR problem]] · [[Backpropagation]] · [[Stochastic gradient descent]]
