---
tags: [MDL, concept]
aliases: []
---
# Perceptron

Linear classifier trained by gradient descent on the summed margins of misclassified points. Converges if the data is separable, otherwise never stops.

$$J(w)=\sum_{\text{miscl.}}-y_iw^Tx_i,\qquad w\leftarrow w+\rho\sum_{\text{miscl.}}y_ix_i$$

**Lectures:** [[ML 03 Linear classifiers]] · [[ML 04 Nonlinear classifiers]]

**Related:** [[Linear discriminant]] · [[Gradient descent]] · [[Feed-forward network]]
