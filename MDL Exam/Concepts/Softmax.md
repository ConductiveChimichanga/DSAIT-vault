---
tags: [MDL, concept]
aliases: []
---
# Softmax

Turns $K$ logits into positive numbers that sum to one. Generalises the sigmoid to several classes.

$$y_k=\frac{e^{z_k}}{\sum_{k'}e^{z_{k'}}}$$

**Lectures:** [[DL 02 Loss functions and maximum likelihood]]

**Related:** [[Cross-entropy]] · [[One-hot encoding]] · [[Sigmoid]]
