---
tags: [MDL, concept]
aliases: []
---
# Support vector machine

The linear classifier with the largest margin. Its solution depends only on the support vectors; slack variables handle overlap and kernels make it nonlinear.

$$\min_{w,b}\tfrac12\lVert w\rVert^2\ \text{ s.t. }\ y_i(w^Tx_i+b)\ge1$$

**Lectures:** [[ML 06 Complexity and SVM]] · [[ML 04 Nonlinear classifiers]]

**Related:** [[Margin]] · [[Support vectors]] · [[Slack variables]] · [[Kernel trick]] · [[VC dimension]]
