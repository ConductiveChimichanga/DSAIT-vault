---
tags: [MDL, concept]
aliases: []
---
# Kernel trick

Replace every inner product by a kernel $K(x,y)=\Phi(x)^T\Phi(y)$, which maps the data implicitly to a high-dimensional space without computing $\Phi$.

$$K_{poly}=(x^Ty+1)^d,\qquad K_{RBF}=\exp\!\big(-\lVert x-y\rVert^2/\sigma^2\big)$$

**Lectures:** [[ML 06 Complexity and SVM]] · [[ML 04 Nonlinear classifiers]] · [[DL 01 Feed-forward networks and SGD]]

**Related:** [[Support vector machine]] · [[Parzen density estimate]]
