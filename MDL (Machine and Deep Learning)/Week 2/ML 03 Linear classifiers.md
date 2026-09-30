---
tags: [MDL, ML, exam]
week: 2
slides: "week3a_LinearClassifiers"
lecturer: "David Tax"
lab: "Assignment 4"
---
# ML 03 · Linear classifiers and bias-variance

[[ML 02 Density-based classification|← ML 02]] · [[00 MDL Index|↑ Index]] · [[ML 04 Nonlinear classifiers|ML 04 →]]

## Big picture
Density estimation is hard in high dimensions. So skip it: **assume a form for the [[Decision boundary|decision boundary]], define a loss, optimise the parameters**. Four [[Linear discriminant|linear classifiers]] share the same model and differ only in the loss.

$$g(x)=w^Tx+w_0,\qquad \text{classify } y_1 \text{ if } g(x)\ge 0,\ \ y_2 \text{ otherwise}$$

- $w$ is perpendicular to the boundary; $w_0$ shifts it away from the origin.
- **Homogeneous coordinates**: append a 1, $\tilde x=\begin{pmatrix}x\\1\end{pmatrix}$, $\tilde w=\begin{pmatrix}w\\w_0\end{pmatrix}$, so $g(x)=\tilde w^T\tilde x$.

| Classifier | What it optimises | Solution |
|---|---|---|
| Nearest mean / LDA | likelihood under Gaussian classes | closed form, see [[ML 02 Density-based classification]] |
| Perceptron | $-\sum$ of misclassified margins | iterative |
| Fisher | between / within scatter ratio | $w=\Sigma_W^{-1}(\mu_A-\mu_B)$ |
| Least squares | squared error to labels ±1 | $\hat w=(X^TX)^{-1}X^Ty$ |
| Logistic | likelihood of labels | gradient ascent |

## Two ways to minimise a cost $J(\theta)$
1. Set $\partial J/\partial\theta=0$ and solve (usually impossible).
2. **[[Gradient descent]]**: $\theta_{t+1}=\theta_t-\rho\,\dfrac{\partial J}{\partial\theta}$, with [[Learning rate|learning rate]] $\rho$.

## Perceptron
Labels $y_i\in\{+1,-1\}$. Loss counts only misclassified points, weighted by how wrong they are:
$$J(w)=\sum_{\text{misclassified }x_i}-y_i\,w^Tx_i$$
Gradient step:
$$w(t+1)=w(t)+\rho_t\sum_{\text{misclassified }x_i}y_i\,x_i$$
- Separable data: converges to *a* separating line (not a unique or best one).
- Non-separable data: never stops updating.
- Can be trained one sample at a time or in batches. It is the ancestor of the [[Feed-forward network|neural network]].

## Fisher linear discriminant
Project onto a direction $w$ and make the classes as separated as possible relative to their spread:
$$J_F(w)=\frac{\lvert w^T\mu_A-w^T\mu_B\rvert^2}{w^T\Sigma_Aw+w^T\Sigma_Bw}=\frac{w^T\Sigma_{\text{between}}\,w}{w^T\Sigma_W\,w}$$
with $\Sigma_{\text{between}}=(\mu_A-\mu_B)(\mu_A-\mu_B)^T$ and $\Sigma_W=\Sigma_A+\Sigma_B$. Setting the derivative to zero:
$$w=\Sigma_W^{-1}(\mu_A-\mu_B)$$
Same direction as [[LDA]], reached without assuming Gaussians. Differences: Fisher's $w_0$ is still free to choose; both need $\Sigma_W^{-1}$, so both break with too little data.

## Least squares
Treat classification as regression on labels $\pm1$:
$$J(w)=E\big[\lvert y-w^Tx\rvert^2\big]\ \Rightarrow\ \hat w=R_x^{-1}E[xy],\quad R_x=E[xx^T]$$
From samples, with $X$ the $N\times d$ data matrix (objects in rows):
$$\hat w=(X^TX)^{-1}X^Ty$$
Derivation (Assignment 4, 1.1a): $\nabla_w\lVert Xw-y\rVert^2=2X^T(Xw-y)=0\Rightarrow X^TXw=X^Ty$.

$X^TX$ is invertible only if the columns of $X$ are linearly independent: need $N\ge d$ and data that spans all $d$ directions. With an intercept (column of ones), the data must not lie in a lower-dimensional affine subspace.

Slide example: four 2D points with $X^TX=\begin{pmatrix}15&0\\0&2\end{pmatrix}$, $X^Ty=\begin{pmatrix}7\\0\end{pmatrix}$, so $\hat w=\begin{pmatrix}7/15\\0\end{pmatrix}$.

## Logistic classifier
Model the log-odds as linear:
$$\ln\frac{p(y_1\mid x)}{p(y_2\mid x)}=\beta_0+\beta^Tx\quad\Rightarrow\quad p(y_1\mid x)=\frac{1}{1+e^{-(\beta_0+\beta^Tx)}},\quad p(y_2\mid x)=\frac{1}{1+e^{\beta_0+\beta^Tx}}$$
Fit by maximising the log-likelihood with gradient ascent, starting from $\beta=0$:
$$\beta_{\text{new}}=\beta_{\text{old}}+\eta\,\frac{\partial\ln L}{\partial\beta},\qquad \frac{\partial \ln L}{\partial\beta_j}=\sum_{i\in y_1}(x_i)_j-\sum_{i=1}^{N}p(y_1\mid x_i)\,(x_i)_j$$
Read the gradient as "(target − prediction) × input", summed over objects. Same thing as [[DL 02 Loss functions and maximum likelihood]] with a [[Sigmoid|sigmoid]] output and [[Cross-entropy|cross-entropy]].

Boundary is linear ($\beta_0+\beta^Tx=0$); $\lVert\beta\rVert$ sets how steep the sigmoid is across it.

## Bias-variance dilemma
A classifier depends on the training set $D$ it happened to get: $g(x;D)$. Average the [[Squared error loss|squared error]] over training sets:
$$E_D\big[(g(x;D)-E[y\mid x])^2\big]=\underbrace{E_D\big[(g(x;D)-E_D[g(x;D)])^2\big]}_{\text{variance}}+\underbrace{\big(E_D[g(x;D)]-E[y\mid x]\big)^2}_{\text{bias}^2}$$
Derivation trick: add and subtract $E_D[g(x;D)]$, expand the square, the cross term averages to zero.
- **Variance**: how much the classifier changes between training sets.
- **Bias**: how far the *average* classifier is from the truth.
- Simple model (LDA): low variance, high bias. Flexible model ([[k-nearest neighbours|k-NN]]): high variance, low bias.
- Simple models are stable and need less data; complex models only pay off with enough data.

![[learning_and_feature_curves.png]]

**[[Feature curve]]**: error vs complexity/number of features is U-shaped for the [[True error and apparent error|true error]]. The minimum shifts right as the training set grows.

## Likely exam questions
- Compute $\hat w=(X^TX)^{-1}X^Ty$ for a tiny dataset, with and without intercept.
- Do one [[Perceptron|perceptron]] update for a given $w$, $\rho$ and misclassified point.
- Write down the [[Fisher linear discriminant|Fisher criterion]] and its solution; relate it to LDA.
- Derive the logistic [[Posterior probability|posterior]] from the linear log-odds assumption.
- Derive or explain the [[Bias-variance tradeoff|bias-variance decomposition]]; label LDA and k-NN on it.
- Slide "things to think about": do perceptron and [[Least squares|least squares]] depend on class densities? Same complexity? How to make a multi-class perceptron / Fisher? (Answers in [[MDL Practice questions]].)

## Books
PRML 4.1 (4.1.1 two classes, 4.1.3 least squares, 4.1.4 Fisher, 4.1.7 perceptron), 4.3.2 ([[Logistic regression|logistic regression]]), 3.2 (bias-variance), 3.1.1 (least-squares normal equations). DL book 5.4 (bias and variance), 5.7.1.

## Flashcards
#flashcards/MDL/Week2

Linear discriminant::$g(x)=w^Tx+w_0$, assign $y_1$ if $g(x)\ge0$

Geometric meaning of $w$ and $w_0$::$w$ is perpendicular to the decision boundary; $w_0$ shifts the boundary away from the origin

How do you absorb the bias into the weight vector?::Homogeneous coordinates, append a 1 to $x$ so $g(x)=\tilde w^T\tilde x$

Two ways to minimise a cost $J(\theta)$::Set the derivative to zero and solve, or follow the gradient $\theta_{t+1}=\theta_t-\rho\,\partial J/\partial\theta$

Perceptron loss::$J(w)=\sum_{\text{misclassified }x_i}-y_i\,w^Tx_i$

Perceptron update::$w\leftarrow w+\rho\sum_{\text{misclassified }x_i}y_i\,x_i$

Perceptron behaviour on separable vs non-separable data::Separable, it converges to some separating boundary. Non-separable, it keeps updating forever.

Fisher criterion::$J_F=\dfrac{\lvert w^T\mu_A-w^T\mu_B\rvert^2}{w^T\Sigma_Aw+w^T\Sigma_Bw}$, between-class over within-class scatter along $w$

Fisher solution::$w=\Sigma_W^{-1}(\mu_A-\mu_B)$ with $\Sigma_W=\Sigma_A+\Sigma_B$

Fisher vs LDA::Same direction $w$. LDA assumes Gaussian classes with equal covariance; Fisher only optimises the criterion and leaves $w_0$ free.

Least-squares solution::$\hat w=(X^TX)^{-1}X^Ty$

Gradient of $\lVert Xw-y\rVert^2$::$2X^T(Xw-y)$; setting it to zero gives the normal equations $X^TXw=X^Ty$

When is $X^TX$ invertible?::When the columns of $X$ are linearly independent, which needs $N\ge d$ and data spanning all $d$ directions

Logistic model assumption::The log-odds are linear, $\ln\dfrac{p(y_1\mid x)}{p(y_2\mid x)}=\beta_0+\beta^Tx$

Logistic posterior::$p(y_1\mid x)=\dfrac{1}{1+\exp(-(\beta_0+\beta^Tx))}$

How is the logistic classifier trained?::Maximise the log-likelihood by gradient ascent, $\beta\leftarrow\beta+\eta\,\partial\ln L/\partial\beta$, starting from $\beta=0$

Bias-variance decomposition::$E_D[(g-E[y\mid x])^2]=E_D[(g-E_D[g])^2]+(E_D[g]-E[y\mid x])^2$, variance plus bias squared

Define variance and bias of a classifier::Variance, how much the classifier changes across training sets. Bias, how far the average classifier is from the true output.

Bias and variance of LDA vs k-NN::LDA has low variance and high bias. k-NN has high variance and low bias.

Shape of the feature curve::Apparent error keeps dropping with complexity; true error is U-shaped. The minimum moves right with more training data.

Which loss does each linear classifier use?::Perceptron, misclassified margins. Least squares, squared error. Logistic and LDA, likelihood. Fisher, scatter ratio.
