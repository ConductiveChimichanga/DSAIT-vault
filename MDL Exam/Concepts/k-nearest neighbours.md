---
tags: [MDL, concept]
aliases: []
---
# k-nearest neighbours

Fix the count $k$ and grow a sphere until it holds $k$ training points. As a classifier it is a majority vote among the $k$ nearest neighbours. Small $k$: high variance. Large $k$: high bias.

$$\hat p(x)=\frac{k}{n\,V_k}$$

**Lectures:** [[ML 02 Density-based classification]] · [[ML 03 Linear classifiers]]

**Related:** [[Parzen density estimate]] · [[Bias-variance tradeoff]] · [[Feature scaling]]
