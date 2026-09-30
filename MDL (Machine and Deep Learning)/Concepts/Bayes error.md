---
tags: [MDL, concept]
aliases: []
---
# Bayes error

The minimum attainable classification error, caused by class overlap. It depends on the data distribution, not on the classifier, and is typically above zero.

$$\varepsilon^*=\int\min\big[p(x\mid y_1)p(y_1),\,p(x\mid y_2)p(y_2)\big]dx$$

**Lectures:** [[ML 01 Bayes decision theory]] · [[ML 05 Evaluation]]

**Related:** [[Bayes classifier]] · [[True error and apparent error]]
