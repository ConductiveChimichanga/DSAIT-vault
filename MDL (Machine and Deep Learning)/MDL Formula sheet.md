---
tags: [MDL, exam, formulas]
---
# MDL formula sheet

Back to [[00 MDL Index]]. Everything worth having in your head, grouped by lecture.

## Bayes → [[ML 01 Bayes decision theory]]
$$p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}\qquad p(x)=\sum_ip(x\mid y_i)p(y_i)$$
$$\varepsilon^*=\int\min_i\big[p(x\mid y_i)p(y_i)\big]dx\ \ (\text{two classes})\qquad\text{min-risk: }\arg\min_j\sum_kL_{kj}\,p(y_k\mid x)$$

## Gaussian classifiers → [[ML 02 Density-based classification]]
$$p(x\mid y)=\frac{1}{\sqrt{(2\pi)^p\det\Sigma}}\exp\!\big(-\tfrac12(x-\mu)^T\Sigma^{-1}(x-\mu)\big)$$
$$\hat\mu=\frac1N\sum_ix_i\qquad\hat\Sigma=\frac1N\sum_i(x_i-\hat\mu)(x_i-\hat\mu)^T\qquad\hat p(y)=\frac{N_y}{N}$$
$$g_i(x)=-\tfrac12\log\det\Sigma_i-\tfrac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)+\log p(y_i)$$
$$\text{LDA: }w=\Sigma^{-1}(\mu_1-\mu_2)\qquad\text{Nearest mean: }w=\mu_1-\mu_2$$

## Non-parametric
$$\text{Histogram: }\hat p(x)=\frac{k_N}{Nh}\qquad\text{Parzen: }\hat p(z)=\frac1n\sum_iK(\lVert z-x_i\rVert,h)\qquad\text{k-NN: }\hat p(x)=\frac{k}{nV_k}$$
$$\text{Rim fraction in }p\text{ dims: }1-0.9^p$$

## Linear classifiers → [[ML 03 Linear classifiers]]
$$g(x)=w^Tx+w_0\qquad\theta_{t+1}=\theta_t-\rho\frac{\partial J}{\partial\theta}$$
$$\text{Perceptron: }J=\sum_{\text{miscl.}}-y_iw^Tx_i\qquad w\leftarrow w+\rho\sum_{\text{miscl.}}y_ix_i$$
$$\text{Fisher: }J_F=\frac{\lvert w^T\mu_A-w^T\mu_B\rvert^2}{w^T\Sigma_Aw+w^T\Sigma_Bw}\qquad w=\Sigma_W^{-1}(\mu_A-\mu_B)$$
$$\text{Least squares: }\hat w=(X^TX)^{-1}X^Ty$$
$$\text{Logistic: }p(y_1\mid x)=\frac{1}{1+e^{-(\beta_0+\beta^Tx)}}$$
$$\text{MSE}=\underbrace{E_D[(g-E_D[g])^2]}_{\text{variance}}+\underbrace{(E_D[g]-E[y\mid x])^2}_{\text{bias}^2}$$

## Trees and ensembles → [[ML 04 Nonlinear classifiers]]
$$\text{Entropy }Q=-\sum_ip_i\log p_i\qquad\text{Gini }Q=\sum_ip_i(1-p_i)$$
$$\hat y_{\text{bag}}=\frac1M\sum_m\hat y_m\qquad E_{\text{bag}}=\tfrac1ME_{\text{AV}}\ (\text{uncorrelated errors})$$
$$\text{AdaBoost: }\alpha_m=\tfrac12\log\frac{1-\varepsilon_m}{\varepsilon_m}\qquad\hat y=\operatorname{sign}\Big(\sum_m\alpha_m\hat y_m(x)\Big)$$

## Evaluation → [[ML 05 Evaluation]]
$$\text{recall}=\text{sensitivity}=\frac{TP}{TP+FN}\quad\text{specificity}=\frac{TN}{TN+FP}\quad\text{precision}=\frac{TP}{TP+FP}\quad\text{FPR}=\frac{FP}{FP+TN}$$
$$\hat\varepsilon_{CV}=\frac1n\sum_i\hat\varepsilon_i\qquad\text{std of test error}\approx\sqrt{\varepsilon(1-\varepsilon)/N_{\text{test}}}\qquad\text{AUC: 1 perfect, 0.5 random}$$

## SVM → [[ML 06 Complexity and SVM]]
$$\text{VC (linear): }h=p+1\qquad h\le\min\big(\lceil R^2/\rho^2\rceil,p\big)+1$$
$$\min\tfrac12\lVert w\rVert^2\ \text{ s.t. }\ y_i(w^Tx_i+b)\ge1\qquad\text{margin width}=\frac{2}{\lVert w\rVert}$$
$$\text{Soft: }\min\tfrac12\lVert w\rVert^2+C\sum_i\xi_i\ \text{ s.t. }\ y_i(w^Tx_i+b)\ge1-\xi_i,\ \xi_i\ge0$$
$$\text{Dual: }\max_\alpha\sum_i\alpha_i-\tfrac12\sum_{i,j}y_iy_j\alpha_i\alpha_jK(x_i,x_j)\ \text{ s.t. }\ \alpha_i\ge0,\ \sum_i\alpha_iy_i=0$$
$$w=\sum_i\alpha_iy_ix_i\qquad f(z)=\sum_i\alpha_iy_iK(x_i,z)+b\qquad\varepsilon_{LOO}\le\frac{\#SV}{N}$$
$$K_{\text{poly}}=(x^Ty+1)^d\qquad K_{\text{RBF}}=\exp(-\lVert x-y\rVert^2/\sigma^2)$$
$$\text{Regularised covariance: }\Sigma\leftarrow\Sigma+\lambda I$$

## Regularisation → [[ML 07 Regularisation]]
$$\tilde J=J+\alpha\Omega(\theta)$$
$$\text{L2: }w\leftarrow(1-\epsilon\alpha)w-\epsilon\nabla_wJ\qquad\text{L1: }\nabla_w\tilde J=\alpha\operatorname{sign}(w)+\nabla_wJ$$
$$\text{Dropout at test: }w_{\text{new}}=p_{\text{keep}}\,w\qquad\text{Adversarial: }x+\epsilon\operatorname{sign}(\nabla_xJ)$$

## Networks and SGD → [[DL 01 Feed-forward networks and SGD]]
$$f(x)=w^T\max\{0,W^Tx+c\}+b\qquad\#\text{params per layer}=n_{in}n_{out}+n_{out}$$
$$\theta^*=\theta-\epsilon\,\frac1k\sum_{i=1}^k\nabla_\theta L(x^{(i)},y^{(i)},\theta)$$

## Losses → [[DL 02 Loss functions and maximum likelihood]]
$$\theta_{ML}=\arg\max_\theta\sum_i\log p_{\text{model}}(x^{(i)};\theta)$$
$$D_{KL}(p\Vert q)=\mathbb E_p[\log p-\log q]\qquad H(p,q)=H(p)+D_{KL}(p\Vert q)$$
$$\sigma(z)=\frac1{1+e^{-z}}\qquad\sigma'=\sigma(1-\sigma)\qquad\operatorname{softmax}(z)_k=\frac{e^{z_k}}{\sum_{k'}e^{z_{k'}}}$$
$$L_{CE}=-t\log y-(1-t)\log(1-y)\qquad L_{CE}=-\sum_kt_k\log y_k\qquad\frac{\partial L_{CE}}{\partial z}=y-t$$

## Backprop → [[DL 03 Backpropagation]]
$$\bar n_N=1\qquad\bar n_i=\sum_{n_j\in\text{Children}(n_i)}\bar n_j\frac{\partial n_j}{\partial n_i}$$
$$z=wx+b,\ y=\sigma(z),\ L=\tfrac12(y-t)^2:\qquad\bar y=y-t,\quad\bar z=\bar y\sigma'(z),\quad\bar w=\bar zx,\quad\bar b=\bar z$$
$$y=xW+b:\qquad\bar W=x^T\bar y,\quad\bar b=\textstyle\sum\bar y,\quad\bar x=\bar yW^T$$

## Optimisers → [[DL 04 Optimisers]]
$$\text{EWMA: }S_t=\rho S_{t-1}+(1-\rho)y_t\qquad\hat S_t=\frac{S_t}{1-\rho^t}$$
$$\text{Momentum: }v\leftarrow\rho v+(1-\rho)\nabla_\theta,\ \ \theta\leftarrow\theta-\epsilon v$$
$$\text{RMSProp: }r\leftarrow\rho r+(1-\rho)\nabla_\theta^2,\ \ \theta\leftarrow\theta-\epsilon\frac{\nabla_\theta}{\sqrt{r+\delta}}$$
$$\text{Adam: }\theta\leftarrow\theta-\epsilon\frac{\hat v}{\sqrt{\hat r+\delta}},\quad\hat v=\frac{v}{1-\rho_1^i},\ \hat r=\frac{r}{1-\rho_2^i}$$
