---
tags: [MDL, ML, exam]
---
# ML 06 · Complexity and support vector classifiers

Back to [[00 MDL Index]] · previous [[ML 05 Evaluation]] · next [[ML 07 Regularisation]]
Slides: `week5a_Complexity` · Lab: Assignment 7

## Big picture
**Complexity = flexibility** = the ability to fit any data distribution. Choose it to match the training set size. The number of parameters is a bad measure of it; the VC dimension is a proper one; the SVM is the linear classifier designed to have a small VC dimension, and the kernel trick makes it nonlinear.

## Complexity and the learning curve
- More complex classifier: lower training error, higher test error on small sets, lower asymptotic error.
- Complex is good with enough data; with little data you overtrain. **Choose the complexity according to the available training set size.**

## Regularising Gaussian classifiers
With $N\le p$ objects (per class for QDA) the covariance estimate is singular and the classifier is undefined (the error peaks there; "QDC crashes"). Fix: add artificial noise to the diagonal,
$$\Sigma_i\leftarrow\Sigma_i+\lambda I$$
- The inverse now exists. Example: $\begin{pmatrix}5&0\\0&0\end{pmatrix}\to\begin{pmatrix}5+\lambda&0\\0&\lambda\end{pmatrix}$.
- $\lambda\to\infty$ with equal priors gives the **nearest mean classifier**.
- LDA shows the same peak; there the pseudo-inverse is typically used instead.

General form of regularisation: minimise training error plus a penalty on flexibility,
$$\varepsilon_A(\theta)+\lambda\,\Omega(\theta)$$
More in [[ML 07 Regularisation]].

## Number of parameters is not complexity
- 1D classifier with $N$ thresholds: $N$ parameters, more thresholds = more complex. Fine.
- $f(x)=\operatorname{sign}(\sin(\omega x))$: **one** parameter, yet by tuning $\omega$ it can separate almost any labelling of points on a line.

## VC dimension
**VC dimension $h$**: the largest number of points that the classifier can **shatter**, meaning it can realise *every* possible labelling of them (for at least one placement of the points).
- Linear classifier in $p$ dimensions: $h=p+1$. (A line in 2D shatters 3 points, not 4: XOR.)
- $\operatorname{sign}(\sin(\omega x))$: $h=\infty$.
- Known for very few classifiers.

Use: bound the true error from the apparent error. With probability at least $1-\eta$,
$$\varepsilon\le\varepsilon_A+\frac{E(N)}{2}\Big(1+\sqrt{1+\frac{4\varepsilon_A}{E(N)}}\Big),\qquad E(N)=4\,\frac{h\big(\ln(2N/h)+1\big)-\ln(\eta/4)}{N}$$
Do not memorise. Know: small $h$ (relative to $N$) → true error close to apparent error. The bound is very loose because it assumes the worst case (random labels); real data is nicely clustered.

## Support vector classifier
Assume separable data and constrain the weights so every training point has output at least 1 in magnitude (**canonical hyperplane**):
$$w^Tx_i+b\ge+1\ \text{ for } y_i=+1,\qquad w^Tx_i+b\le-1\ \text{ for } y_i=-1\qquad\Longleftrightarrow\qquad y_i(w^Tx_i+b)\ge1$$

For such a hyperplane with margin $\rho$ on data inside a sphere of radius $R$:
$$h\le\min\Big(\Big\lceil\frac{R^2}{\rho^2}\Big\rceil,\ p\Big)+1$$
So to get a small VC dimension: fewer dimensions, smaller radius, or **larger margin**. The distance from the boundary to each margin plane is $\rho=1/\lVert w\rVert$, so the band is $2/\lVert w\rVert$ wide. Maximising the margin = minimising $\lVert w\rVert$:

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2\quad\text{s.t.}\quad y_i(w^Tx_i+b)\ge1\ \ \forall i$$

![[svm_margin.png]]

### Dual form (Lagrange multipliers $\alpha_i$)
At a constrained optimum the gradient of the objective is parallel to the gradient of the constraint: $\dfrac{\partial J}{\partial\theta}=\lambda\dfrac{\partial f}{\partial\theta}$. Applying this and eliminating $w,b$:
$$\max_\alpha\ \sum_i\alpha_i-\tfrac12\sum_{i,j}y_iy_j\alpha_i\alpha_j\,x_i^Tx_j\quad\text{s.t.}\quad\alpha_i\ge0,\ \ \sum_i\alpha_iy_i=0$$
$$w=\sum_i\alpha_iy_ix_i,\qquad f(z)=\sum_i\alpha_iy_i\,x_i^Tz+b$$
- A quadratic programming problem with one unique solution.
- The solution is written in terms of **objects, not features**.
- Most $\alpha_i=0$. Objects with $\alpha_i>0$ lie on the margin: the **support vectors**. Removing any other object changes nothing.
- Leave-one-out bound: $\varepsilon_{\text{LOO}}\le\dfrac{\#\text{support vectors}}{N}$
- Because it depends on objects, it does well in high-dimensional spaces with few samples.

### Problem 1: classes overlap → slack variables
$$\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2+C\sum_i\xi_i\quad\text{s.t.}\quad y_i(w^Tx_i+b)\ge1-\xi_i,\ \ \xi_i\ge0$$
- $\xi_i$ = how far object $i$ violates its margin.
- $C$ trades training error against margin width. Large $C$: few violations, narrow margin, outliers have a big influence (more complex). Small $C$: wide margin, more violations (simpler).
- $C$ must be set beforehand, usually by cross-validation.

### Problem 2: the boundary is linear → kernel trick
The dual and the classifier only use **inner products** $x_i^Tx_j$. Map the data with $\Phi$ and every inner product becomes $\Phi(x_i)^T\Phi(x_j)$. Define a **kernel** $K(x,y)=\Phi(x)^T\Phi(y)$ and never compute $\Phi$:
$$f(z)=\sum_i\alpha_iy_i\,K(x_i,z)+b$$

Worked example. $\Phi(x)=(x_1^2,\ x_2^2,\ \sqrt2\,x_1x_2)$:
$$\Phi(x)^T\Phi(y)=x_1^2y_1^2+x_2^2y_2^2+2x_1x_2y_1y_2=(x_1y_1+x_2y_2)^2=(x^Ty)^2$$
A 3D inner product for the price of a 2D one.

| Kernel | $K(x,y)$ | Parameter |
|---|---|---|
| Linear | $x^Ty$ | |
| Polynomial | $(x^Ty+1)^d$ | degree $d$ |
| RBF / Gaussian | $\exp\!\big(-\lVert x-y\rVert^2/\sigma^2\big)$ | width $\sigma$ |

- The kernel implicitly maps to a (usually very) high-dimensional space; RBF to an infinite-dimensional one.
- Computational cost does not change apart from evaluating $K$.
- Small $\sigma$ = very flexible (overfits); large $\sigma$ = nearly linear. Same role as the Parzen width $h$.
- Special kernels exist for strings, images, invariances.

### SVM pros and cons
| + | − |
|---|---|
| generalises well, especially high-dim with few samples | quadratic optimisation is expensive |
| unique solution for given kernel and $C$ | kernel and $C$ must be tuned |
| kernel gives adjustable complexity and engineered representations | struggles with heavily overlapping classes |
| no density model assumed; solid theory (without slack) | sensitive to feature scaling |

## Likely exam questions
- Define VC dimension; give it for a linear classifier; explain why parameter count fails ($\sin\omega x$).
- Find the maximum-margin line and the support vectors for 3–4 points by geometry (Assignment 7, solved in [[MDL Practice questions]]).
- Write the primal SVM problem; explain why minimising $\lVert w\rVert$ maximises the margin.
- What are support vectors, and what is the LOO bound?
- Role of $C$; role of kernel width.
- Verify a kernel equals an inner product after a feature map (polynomial example; RBF via Taylor series).
- Why can an SVM be faster than Parzen at test time? (Only the support vectors are evaluated, not all training points.)
- Effect of $\lambda$ in $\Sigma+\lambda I$; limit $\lambda\to\infty$.

## Books
PRML 7.1 (maximum margin classifiers; 7.1.1 overlapping classes), 6.1–6.2 (dual representations, constructing kernels), Appendix E (Lagrange multipliers), 7.1.5 (computational learning theory, VC). DL book 5.7.2 (SVM), 5.2 (capacity, VC dimension).

## Flashcards
#flashcards/MDL/Week5

What is the complexity of a classifier?::Its flexibility, the ability to fit any data distribution

How should you choose complexity?::According to the available training set size. Complex needs lots of data, simple for small sets.

Regularised covariance::$\Sigma\leftarrow\Sigma+\lambda I$, which makes the inverse exist

QDA with $\lambda\to\infty$ and equal priors::The nearest mean classifier

General regularised objective::$\varepsilon_A(\theta)+\lambda\,\Omega(\theta)$, training error plus a penalty on flexibility

Why is the number of parameters a poor complexity measure?::$\operatorname{sign}(\sin(\omega x))$ has one parameter but can separate almost any labelling

VC dimension::The largest number of points $h$ that the classifier can shatter, i.e. realise every possible labelling

VC dimension of a linear classifier in $p$ dimensions::$h=p+1$

What does a small VC dimension give you?::A true error close to the apparent error (tight bound)

Why is the VC bound not practical?::It is very loose, it assumes the worst case of randomly labelled objects

Canonical hyperplane constraints::$y_i(w^Tx_i+b)\ge1$ for all training objects

VC bound for a canonical hyperplane::$h\le\min(\lceil R^2/\rho^2\rceil,\,p)+1$, so a larger margin $\rho$ gives a smaller $h$

Margin width of an SVM::$2/\lVert w\rVert$

Hard-margin SVM problem::$\min\tfrac12\lVert w\rVert^2$ subject to $y_i(w^Tx_i+b)\ge1$

SVM weight vector from the dual::$w=\sum_i\alpha_iy_ix_i$, with $\alpha_i\ge0$ and $\sum_i\alpha_iy_i=0$

What are support vectors?::The training objects with $\alpha_i>0$. They lie on the margin and alone determine the classifier.

Leave-one-out bound for the SVM::$\varepsilon_{LOO}\le\dfrac{\#\text{support vectors}}{N}$

Soft-margin SVM::$\min\tfrac12\lVert w\rVert^2+C\sum_i\xi_i$ subject to $y_i(w^Tx_i+b)\ge1-\xi_i$, $\xi_i\ge0$

Role of $C$::Trades training errors against margin width. Large $C$, few violations and narrow margin. Small $C$, wide margin. Set by cross-validation.

What is the kernel trick?::Replace every inner product $x_i^Tx_j$ by a kernel $K(x_i,x_j)=\Phi(x_i)^T\Phi(x_j)$, without ever computing $\Phi$

Kernelised SVM classifier::$f(z)=\sum_i\alpha_iy_iK(x_i,z)+b$

Polynomial and RBF kernel::$(x^Ty+1)^d$ and $\exp(-\lVert x-y\rVert^2/\sigma^2)$

Kernel for $\Phi(x)=(x_1^2,x_2^2,\sqrt2x_1x_2)$::$K(x,y)=(x^Ty)^2$

Why does the SVM work well in high dimensions?::The solution depends on (few) objects, not on the features, and no density is estimated

Three disadvantages of the SVM::Expensive quadratic optimisation, kernel and $C$ must be tuned, trouble with heavily overlapping classes

Why can an SVM be faster than Parzen at test time?::It evaluates kernels only on the support vectors, Parzen on all training objects
