---
tags: [MDL, concept]
aliases: []
---
# Cross-entropy

Entropy of the data plus the KL divergence to the model. As a loss it is the negative log-likelihood of the labels. With sigmoid or softmax outputs its gradient with respect to the logits is $y-t$.

$$H(p,q)=H(p)+D_{KL}(p\Vert q),\qquad L_{CE}=-\sum_kt_k\log y_k$$

**Lectures:** [[DL 02 Loss functions and maximum likelihood]]

**Related:** [[KL divergence]] · [[Softmax]] · [[Sigmoid]] · [[Maximum likelihood estimation]]
