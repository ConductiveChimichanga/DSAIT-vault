---
tags: [MDL, concept]
aliases: []
---
# Sigmoid

S-shaped function squashing a logit into $(0,1)$. Output unit for binary classification.

$$\sigma(z)=\frac1{1+e^{-z}},\qquad\sigma'(z)=\sigma(z)(1-\sigma(z))$$

**Lectures:** [[DL 02 Loss functions and maximum likelihood]] · [[ML 03 Linear classifiers]]

**Related:** [[Logit]] · [[Softmax]] · [[Logistic regression]]
