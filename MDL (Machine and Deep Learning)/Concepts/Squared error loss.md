---
tags: [MDL, concept]
aliases: []
---
# Squared error loss

Half the squared difference between prediction and target. The negative log-likelihood of a Gaussian output; a poor match for a sigmoid output because its gradient vanishes for confident mistakes.

$$L_{SE}=\tfrac12(y-t)^2$$

**Lectures:** [[DL 01 Feed-forward networks and SGD]] · [[DL 02 Loss functions and maximum likelihood]] · [[DL 03 Backpropagation]]

**Related:** [[Cross-entropy]] · [[Least squares]]
