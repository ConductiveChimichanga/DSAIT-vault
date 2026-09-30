---
tags: [MDL, concept]
aliases: []
---
# XOR problem

Four points with labels 0, 1, 1, 0 that no linear model can separate. One hidden ReLU layer solves it by folding the two positive points onto each other.

$$W=\begin{pmatrix}1&1\\1&1\end{pmatrix},\ c=\begin{pmatrix}0\\-1\end{pmatrix},\ w=\begin{pmatrix}1\\-2\end{pmatrix},\ b=0$$

**Lectures:** [[DL 01 Feed-forward networks and SGD]]

**Related:** [[Feed-forward network]] · [[ReLU]] · [[VC dimension]]
