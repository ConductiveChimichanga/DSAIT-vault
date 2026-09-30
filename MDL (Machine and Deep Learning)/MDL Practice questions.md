---
tags: [MDL, exam, practice]
---
# MDL practice questions

Back to [[00 MDL Index]]. Answers are folded: click the callout to open it. Questions marked **(lab)** are the pen-and-paper exercises from your assignments; the answers are mine, not an official key, so compare them with the lab solutions.

## Bayes decision theory → [[ML 01 Bayes decision theory]]

**1. (lab A1 1.1a)** Class $\omega_1$: Gaussian, $\mu_1=0$, $\sigma_1^2=\tfrac12$. Class $\omega_2$: Gaussian, $\mu_2=1$, $\sigma_2^2=\tfrac12$. Equal priors. Where is the decision boundary?
> [!success]- Answer
> Equal priors and equal variances, so set the densities equal: $(x-0)^2=(x-1)^2\Rightarrow x=\tfrac12$. The midpoint of the means.

**2. (lab A1 1.1b–d)** $\omega_1$ uniform on $[0,1]$, equal priors. Where is the boundary if $\omega_2$ is uniform on (b) $[2,3]$, (c) $[0.5,1.5]$, (d) $[0.5,2.5]$?
> [!success]- Answer
> (b) Anywhere in the gap $1<x<2$; both densities are zero there so any threshold gives zero error.
> (c) Both densities equal 1 on the overlap $[0.5,1]$, so the posteriors tie on the whole interval. $x<0.5\to\omega_1$, $x>1\to\omega_2$; any threshold in $[0.5,1]$ is optimal (Bayes error $=\tfrac12\cdot0.5=0.25$).
> (d) $p(x\mid\omega_2)=0.5$ on $[0.5,2.5]$, which is below $p(x\mid\omega_1)=1$ on the overlap. So $\omega_1$ wins on all of $[0,1]$ and the boundary is $x=1$.

**3. (lab A1 1.2a)** Same Gaussians as question 1, equal priors, loss matrix $L=\begin{pmatrix}0&0.5\\1&0\end{pmatrix}$. Boundary?
> [!success]- Answer
> The slides define $\lambda_{ji}$ = cost of assigning an object from class $j$ to class $i$, so $\lambda_{12}=0.5$ and $\lambda_{21}=1$: missing a true $\omega_2$ is the expensive mistake. Assign $\omega_1$ if $\lambda_{12}\,p(\omega_1\mid x)>\lambda_{21}\,p(\omega_2\mid x)$. Boundary: $0.5\,p(x\mid\omega_1)=1\cdot p(x\mid\omega_2)$.
> With $\sigma^2=\tfrac12$ the densities are $\propto e^{-(x-\mu)^2}$: $-(x-1)^2=\ln0.5-x^2\Rightarrow2x-1=-\ln2\Rightarrow x=\dfrac{1-\ln2}{2}\approx0.153$.
> The boundary moves toward $\omega_1$: we say "$\omega_2$" more often because missing it is expensive.

**3b. (lab A1 1.2b)** Same loss matrix, $\omega_1$ uniform on $[0,1]$, $\omega_2$ uniform on $[0.5,2.5]$, equal priors. Boundary?
> [!success]- Answer
> On the overlap $[0.5,1]$: $\lambda_{12}\,p(x\mid\omega_1)=0.5\cdot1=0.5$ and $\lambda_{21}\,p(x\mid\omega_2)=1\cdot0.5=0.5$. The risks are equal on the whole overlap, so any threshold in $[0.5,1]$ is optimal. Without costs the boundary was at $x=1$ (question 2d); the cost exactly cancels class 1's density advantage.

**4.** Two classes with $p(x\mid y_1)p(y_1)$ and $p(x\mid y_2)p(y_2)$ sketched. How do you read off the Bayes error?
> [!success]- Answer
> It is the area under the lower of the two curves: $\int\min[\cdot,\cdot]\,dx$.

## Density-based classifiers → [[ML 02 Density-based classification]]

**5.** A class has two training points $(-1,1)^T$ and $(0,1)^T$. Estimate $\hat\mu$ and $\hat\Sigma$ (ML). Can you build a QDA classifier?
> [!success]- Answer
> $\hat\mu=(-0.5,1)^T$. Centred points $(\mp0.5,0)^T$, so $\hat\Sigma=\tfrac12\Big[\begin{pmatrix}0.25&0\\0&0\end{pmatrix}+\begin{pmatrix}0.25&0\\0&0\end{pmatrix}\Big]=\begin{pmatrix}0.25&0\\0&0\end{pmatrix}$.
> $\det\hat\Sigma=0$: no inverse, the Gaussian density is undefined, QDA fails. Fix with shared covariance (LDA), $\sigma^2I$ (nearest mean) or $\hat\Sigma+\lambda I$.

**6.** Show that the k-NN classifier is a majority vote.
> [!success]- Answer
> $\hat p(x\mid y_m)=\dfrac{k_m}{n_mV_k}$, $\hat p(y_m)=\dfrac{n_m}{n}$. Product: $\dfrac{k_m}{nV_k}$. $n$ and $V_k$ are the same for all classes, so the largest posterior is the largest $k_m$.

**7.** What is the k-NN error when $k=N$?
> [!success]- Answer
> Every object gets the label of the largest class, so the error is the prior of the smaller class, $\min(p(y_1),p(y_2))$.

## Linear classifiers → [[ML 03 Linear classifiers]]

**8. (lab A4 1.2)** Inputs $X=(-2,-1,0,3)^T$, outputs $Y=(1,1,2,3)^T$. (a) Least-squares line without intercept. (b) With intercept. (c) Minimum polynomial degree for an exact fit?
> [!success]- Answer
> (a) $w=\dfrac{\sum x_iy_i}{\sum x_i^2}=\dfrac{-2-1+0+9}{4+1+0+9}=\dfrac{6}{14}=\dfrac37$.
> (b) $\bar x=0$, $\bar y=\tfrac74$. Slope $=\dfrac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}=\dfrac{6}{14}=\dfrac37$ (unchanged because $\bar x=0$). Intercept $=\bar y-\text{slope}\cdot\bar x=\dfrac74$.
> (c) With intercept: 4 points with distinct $x$ are fitted exactly by degree 3 (4 coefficients). Without intercept: never, because the curve must pass through $(0,0)$ but the data has $y=2$ at $x=0$.

**9.** Perceptron with $w=(1,1)^T$, learning rate $\rho=0.1$. The only misclassified object is $x=(2.5,-2.8)^T$ with $y=+1$. New $w$?
> [!success]- Answer
> Check: $w^Tx=-0.3<0$ but $y=+1$, so it is misclassified. $w\leftarrow w+\rho\,y\,x=(1+0.25,\ 1-0.28)=(1.25,\ 0.72)^T$. (This is the slide example.)

**10.** Slide "things to think about".
> [!success]- Answer
> *Are least squares/Fisher and the perceptron dependent on the class densities?* Neither estimates a density. Least squares/Fisher depend on the data through means and covariances (all points matter, outliers too); the perceptron depends only on the misclassified points.
> *Same complexity?* Both are linear, so the same model class: VC dimension $p+1$.
> *Multi-class perceptron / Fisher?* Train one-vs-rest (one discriminant per class, pick the largest output) or one-vs-one with voting. Multi-class Fisher: maximise between-class over within-class scatter, giving up to $C-1$ projection directions (PRML 4.1.6).

## Nonlinear classifiers → [[ML 04 Nonlinear classifiers]]

**11.** A node holds 8 objects of class A and 2 of class B. Gini impurity and entropy?
> [!success]- Answer
> Gini $=0.8\cdot0.2+0.2\cdot0.8=0.32$. Entropy $=-0.8\ln0.8-0.2\ln0.2=0.500$ nats ($0.722$ bits).

**12.** An AdaBoost weak learner has weighted error $\varepsilon=0.2$. Its weight? What if $\varepsilon=0.5$?
> [!success]- Answer
> $\alpha=\tfrac12\ln\dfrac{0.8}{0.2}=\tfrac12\ln4\approx0.693$. For $\varepsilon=0.5$: $\alpha=0$, a coin flip gets no say.

## Evaluation → [[ML 05 Evaluation]]

**13.** TP = 40, FN = 10, FP = 20, TN = 130. Error, recall, precision, specificity?
> [!success]- Answer
> Error $=30/200=0.15$. Recall $=40/50=0.80$. Precision $=40/60=0.67$. Specificity $=130/150=0.87$.

**14. (lab A6)** Why do the learning curves of a simple and a complex classifier intersect?
> [!success]- Answer
> With little data the complex one overfits (high variance) and is worse. With lots of data its variance vanishes and its lower bias wins (lower asymptotic error). So somewhere they cross; the best classifier depends on the training set size.

**15. (lab A6)** How do the bias and variance of the cross-validation estimate change with the number of folds?
> [!success]- Answer
> More folds: each classifier trains on more data, so less pessimistic bias. Each test fold is smaller, so individual fold estimates are noisier, and LOO estimates are highly correlated across folds. With a much larger dataset all of these effects shrink.

## Complexity and SVM → [[ML 06 Complexity and SVM]]

**16. (lab A7 1.1a)** Class 1: $(0,1)^T$, $(0,3)^T$. Class 2: $(2,0)^T$. Maximum-margin classifier and number of support vectors?
> [!success]- Answer
> The closest pair across classes is $(0,1)$ and $(2,0)$. The boundary is their perpendicular bisector, through $(1,0.5)$, normal direction $(-2,1)$.
> Solve $w=c(-2,1)$ with $w^T(0,1)+b=+1$ and $w^T(2,0)+b=-1$: $c=\tfrac25$, $b=\tfrac35$, so $w=(-0.8,\ 0.4)^T$. Margin width $2/\lVert w\rVert=\sqrt5$. Check $(0,3)$: $1.2+0.6=1.8\ge1$, not on the margin.
> **2 support vectors.**

**17. (lab A7 1.1b)** Move $(0,1)^T$ to $(0,-1)^T$. What changes?
> [!success]- Answer
> Class 1 now spans the segment from $(0,-1)$ to $(0,3)$ on the line $x_1=0$; its closest point to $(2,0)$ is $(0,0)$. Boundary: $x_1=1$, i.e. $w=(-1,0)^T$, $b=1$. Both class-1 points give $+1$ and $(2,0)$ gives $-1$: **3 support vectors**, margin width 2.

**18. (lab A7 1.2)** Why is the SVM sensitive to feature scaling?
> [!success]- Answer
> The margin is a Euclidean distance. Rescaling one feature changes which points are closest and hence the support vectors and the boundary direction, so a test point can change sides. (LDA/Fisher with a full covariance is invariant to such rescaling.)

**19. (lab A7 2.1)** Find $\phi$ with $\phi(x)^T\phi(\chi)=\exp(-(x-\chi)^2)$ for scalar $x,\chi$. Dimensionality?
> [!success]- Answer
> $e^{-(x-\chi)^2}=e^{-x^2}e^{-\chi^2}e^{2x\chi}=e^{-x^2}e^{-\chi^2}\sum_{k=0}^\infty\dfrac{(2x\chi)^k}{k!}$.
> So $\phi_k(x)=e^{-x^2}\sqrt{\dfrac{2^k}{k!}}\,x^k$ for $k=0,1,2,\dots$ **Infinite-dimensional.**

**20. (lab A7 2.2)** Kernelise the nearest mean classifier. What does it become with a Gaussian kernel?
> [!success]- Answer
> $\lVert x-\mu_C\rVert^2=x^Tx-\dfrac{2}{N_C}\sum_ix^Tx_i^C+\dfrac{1}{N_C^2}\sum_{i,j}x_i^{C\,T}x_j^C$. Replace every inner product by $K$.
> For the Gaussian kernel $K(x,x)=1$ for every class, so: assign to the class with the largest $\dfrac{1}{N_C}\sum_iK(x,x_i^C)-\dfrac{1}{2N_C^2}\sum_{i,j}K(x_i^C,x_j^C)$. The first term is a Parzen density estimate of the class at $x$; the second is a constant per class. So it is a Parzen classifier with a class-dependent offset.

**21. (lab A7 2.3c)** Why can an SVM be much faster than Parzen at test time?
> [!success]- Answer
> Both evaluate a sum of kernels, but the SVM sums only over the support vectors; Parzen sums over every training object.

## Neural networks → [[DL 01 Feed-forward networks and SGD]]

**22.** How many parameters in a fully connected net 784 → 100 → 10?
> [!success]- Answer
> $(784\cdot100+100)+(100\cdot10+10)=78\,500+1\,010=79\,510$.

**23.** Verify that $W=\begin{pmatrix}1&1\\1&1\end{pmatrix}$, $c=(0,-1)^T$, $w=(1,-2)^T$, $b=0$ solves XOR.
> [!success]- Answer
> See the table in [[DL 01 Feed-forward networks and SGD]]: outputs $0,1,1,0$.

## Losses → [[DL 02 Loss functions and maximum likelihood]]

**24.** Logits $z=(2,1,0)$, true class is the first. Softmax, cross-entropy loss, gradient w.r.t. $z$?
> [!success]- Answer
> $e^z=(7.389,2.718,1)$, sum $11.107$. $y=(0.665,0.245,0.090)$. $L=-\ln0.665=0.408$. $\partial L/\partial z=y-t=(-0.335,\ 0.245,\ 0.090)$.

**25.** Target $t=1$, logit $z=-6$. Compare $dL/dz$ for squared error and cross-entropy with a sigmoid output.
> [!success]- Answer
> $y=\sigma(-6)=0.0025$. SE: $(y-t)\,y(1-y)=-0.9975\cdot0.0025\approx-0.0025$. CE: $y-t=-0.9975$. Cross-entropy gives a gradient about 400 times larger on this badly wrong prediction.

## Backprop and optimisers → [[DL 03 Backpropagation]], [[DL 04 Optimisers]]

**26.** $x=2,w=3,b=4,t=5$, ReLU activation, $L=\tfrac12(y-t)^2$, learning rate 0.1. One gradient descent step.
> [!success]- Answer
> Forward $z=10,y=10,L=12.5$. Backward $\bar y=5,\bar z=5,\bar w=10,\bar b=5$. Update $w=2,b=3.5$. New $y=7.5$, $L=3.125$.

**27.** Same model with $t=20$ instead. One step.
> [!success]- Answer
> Forward $z=10,y=10,L=50$. $\bar y=-10,\bar z=-10,\bar w=-20,\bar b=-10$. $w=3+2=5$, $b=4+1=5$. New $y=15$, $L=12.5$.

**28.** EWMA with $\rho=0.9$, $S_0=0$, observations $y_1=10$, $y_2=20$. Give $S_1,S_2$ with and without bias correction.
> [!success]- Answer
> $S_1=0.1\cdot10=1$, $S_2=0.9\cdot1+0.1\cdot20=2.9$. Corrected: $\hat S_1=1/(1-0.9)=10$, $\hat S_2=2.9/(1-0.81)=15.26$.

## Regularisation → [[ML 07 Regularisation]]

**29.** $w=2$, $\nabla_wJ=0.5$, $\epsilon=0.1$, L2 coefficient $\alpha=0.5$. One update.
> [!success]- Answer
> $w\leftarrow(1-\epsilon\alpha)w-\epsilon\nabla_wJ=0.95\cdot2-0.05=1.85$.

**30.** A unit is kept with probability $p=0.8$ during dropout training and its outgoing weight is $w=0.5$. Weight at test time?
> [!success]- Answer
> $w_{\text{test}}=p\,w=0.4$, so that the expected input to the next layer matches training.
