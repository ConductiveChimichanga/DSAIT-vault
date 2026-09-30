---
tags: [MDL, concept]
aliases: []
---
# AdaBoost

The standard boosting algorithm. A weak learner with weighted error $\varepsilon_m$ gets weight $\alpha_m$; misclassified objects are up-weighted by $e^{\alpha_m}$.

$$\alpha_m=\tfrac12\log\frac{1-\varepsilon_m}{\varepsilon_m},\qquad\hat y=\operatorname{sign}\Big(\sum_m\alpha_m\hat y_m(x)\Big)$$

**Lectures:** [[ML 04 Nonlinear classifiers]]

**Related:** [[Boosting]] · [[Decision stump]]
