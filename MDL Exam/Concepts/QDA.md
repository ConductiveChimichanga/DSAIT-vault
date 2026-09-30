---
tags: [MDL, concept]
aliases: []
---
# QDA

Gaussian plug-in classifier with a separate covariance per class. The decision boundary is quadratic. Needs the most data of the Gaussian classifiers.

$$g_i(x)=-\tfrac12\log\det\Sigma_i-\tfrac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)+\log p(y_i)$$

**Lectures:** [[ML 02 Density-based classification]] · [[ML 06 Complexity and SVM]]

**Related:** [[LDA]] · [[Nearest mean classifier]] · [[Covariance matrix]]
