---
tags: [MDL, concept]
aliases: []
---
# Covariance matrix

Describes the spread of each feature (diagonal) and how features vary together (off-diagonal). Its estimate is singular when there are too few objects for the dimensionality.

$$\hat\Sigma=\frac1N\sum_i(x_i-\hat\mu)(x_i-\hat\mu)^T$$

**Lectures:** [[ML 02 Density-based classification]] · [[ML 06 Complexity and SVM]]

**Related:** [[Gaussian distribution]] · [[Regularised covariance]]
