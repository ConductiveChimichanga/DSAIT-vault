---
tags: [MDL, concept]
aliases: []
---
# Misclassification cost

$\lambda_{ji}$ is the cost of assigning an object from class $j$ to class $i$. Costs rescale the posteriors and move the boundary away from the class that is expensive to miss.

$$l_i(x)=\sum_j\lambda_{ji}\,p(y_j\mid x)\qquad\text{assign }y_1\text{ if }\lambda_{12}p(y_1\mid x)>\lambda_{21}p(y_2\mid x)$$

**Lectures:** [[ML 01 Bayes decision theory]]

**Related:** [[Conditional risk]] · [[Bayes classifier]]
