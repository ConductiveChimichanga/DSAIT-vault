---
tags: [MDL, concept]
aliases: []
---
# Bias-variance tradeoff

The expected squared error splits into variance (sensitivity to the training set) and squared bias (systematic error of the average model). Simple models: high bias, low variance. Flexible models: the reverse.

$$E_D[(g-E[y\mid x])^2]=E_D[(g-E_D[g])^2]+(E_D[g]-E[y\mid x])^2$$

**Lectures:** [[ML 03 Linear classifiers]] · [[ML 05 Evaluation]] · [[ML 06 Complexity and SVM]]

**Related:** [[Overfitting]] · [[Feature curve]] · [[Learning curve]] · [[Regularisation]]
