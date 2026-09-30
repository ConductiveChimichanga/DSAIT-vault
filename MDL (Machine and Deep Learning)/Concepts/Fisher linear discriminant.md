---
tags: [MDL, concept]
aliases: []
---
# Fisher linear discriminant

Find the projection direction that maximises between-class separation relative to within-class scatter. Gives the same direction as LDA.

$$J_F=\frac{\lvert w^T\mu_A-w^T\mu_B\rvert^2}{w^T\Sigma_Aw+w^T\Sigma_Bw},\qquad w=\Sigma_W^{-1}(\mu_A-\mu_B)$$

**Lectures:** [[ML 03 Linear classifiers]]

**Related:** [[LDA]] · [[Linear discriminant]]
