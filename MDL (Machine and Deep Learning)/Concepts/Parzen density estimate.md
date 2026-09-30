---
tags: [MDL, concept]
aliases: []
---
# Parzen density estimate

Put a kernel of fixed width $h$ on every training point and average them. Small $h$ overfits, large $h$ oversmooths.

$$\hat p(z\mid h)=\frac1n\sum_{i=1}^nK(\lVert z-x_i\rVert,h)$$

**Lectures:** [[ML 02 Density-based classification]]

**Related:** [[k-nearest neighbours]] · [[Histogram density estimate]] · [[Kernel trick]]
