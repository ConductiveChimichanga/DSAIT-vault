---
tags: [MDL, concept]
aliases: []
---
# Histogram density estimate

Split a feature into bins of width $h$ and count. Bin width trades precision against stability; useless in high dimensions.

$$\hat p(x)=\frac{k_N}{N\,h}$$

**Lectures:** [[ML 02 Density-based classification]]

**Related:** [[Parzen density estimate]] · [[Curse of dimensionality]]
