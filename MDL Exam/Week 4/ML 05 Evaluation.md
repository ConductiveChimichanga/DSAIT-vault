---
tags: [MDL, ML, exam]
---
# ML 05 · Classifier evaluation

Back to [[00 MDL Index]] · previous [[ML 04 Nonlinear classifiers]] · next [[ML 06 Complexity and SVM]]
Slides: `week4b_evaluation` · Lab: Assignment 6

## Big picture
The error on the training set is **not** a good measure of the true error, and a single number is not all we want. This lecture is about estimating performance honestly and reading the curves that diagnose a classifier.

## What each classifier optimises

| Classifier | Optimises |
|---|---|
| QDA, LDA, nearest mean, logistic | maximum likelihood |
| Decision tree | impurity / Gini |
| Neural network | MSE or maximum likelihood (cross-entropy) |
| k-NN | nothing explicit |

**Surrogate loss**: the classification error (0-1 loss) cannot be optimised directly (not differentiable), so we optimise a convenient stand-in. The optimum of the surrogate is generally not the optimum of the true error.

## Definitions
- **True error** $\varepsilon$: error on the whole (unseen) distribution.
- **Apparent error** $\varepsilon_A$: error on the training set. Optimistically biased.
- **Overfitting**: the gap between the two.
- **Bayes error**: the floor no classifier gets under.
- **Asymptotic error**: what a given classifier reaches with infinite data (≥ Bayes error).

## Train/test trade-off
- Large training set → good classifier. Large test set → reliable, unbiased error estimate.
- Same set for both → optimistic bias.
- Small independent test set → unbiased but high variance estimate.
- 50/50 split is a common default, not necessarily good.

A test error estimate is a binomial proportion, so its standard deviation is roughly $\sqrt{\varepsilon(1-\varepsilon)/N_{\text{test}}}$: quadruple the test set to halve the uncertainty.

Sources of variation in a measured error (Assignment 6): which training set you drew, which test set you drew, and randomness inside the training algorithm.

## Cross-validation
Split the data in $n$ parts. Train on $n-1$, test on the remaining one, rotate $n$ times, average:
$$\hat\varepsilon=\frac1n\sum_{i=1}^{n}\hat\varepsilon_i$$
- **Leave-one-out (LOO)**: $n=N$. Nearly unbiased (trains on $N-1$ objects) but expensive and the estimate has high variance.
- **10-fold** is the usual compromise.
- Fewer folds → each classifier sees less data → pessimistic bias. More folds → less bias, more compute.
- **Bootstrapping**: resample with replacement to get many train sets.

## Learning curves and feature curves
![[learning_and_feature_curves.png]]

**Learning curve**: error vs training set size, for train and test.
- True error decreases, apparent error increases, they meet at the asymptotic error.
- A more complex classifier has a lower training error, a **higher** test error on small sets and a **lower** asymptotic error. So the curves of a simple and a complex classifier **cross**: there is no single best classifier.
- Use it to see the amount of overtraining, whether more data would help, and how classifiers compare.

**Feature curve**: error vs number of features / complexity at fixed training size. Apparent error keeps dropping; true error is U-shaped (curse of dimensionality). For Parzen, the width $h$ plays the role of (inverse) complexity.

Rules from the slides: more complex classifiers and larger feature sets need larger training sets; small training sets need simpler classifiers or fewer features.

## Confusion matrix
$c_{ij}$ = number of objects of true class $i$ labelled as class $j$. Shows *which* errors are made; two classifiers with the same overall error can have very different matrices.

Slide example (rows = true class): A: 8, 2, 0 · B: 6, 23, 1 · C: 4, 1, 15. Per-class errors 2/10 = 0.20, 7/30 = 0.23, 5/20 = 0.25. Overall error 14/60 = 0.233; the unweighted mean of the class errors is 0.228.

## Two-class measures
Let A be the target (positive) class.

| | predicted A | predicted B |
|---|---|---|
| **true A** | TP | FN |
| **true B** | FP | TN |

$$\text{error}=\frac{FP+FN}{N}\qquad\text{accuracy}=1-\text{error}$$
$$\text{sensitivity}=\text{recall}=\text{TPR}=\frac{TP}{TP+FN}\qquad\text{specificity}=\frac{TN}{TN+FP}$$
$$\text{precision}=\frac{TP}{TP+FP}\qquad\text{FPR}=1-\text{specificity}=\frac{FP}{FP+TN}$$

In the slide notation: $\varepsilon_A$ = part of class A sent to B, $\eta_A$ = part classified correctly, $p(A)=\eta_A+\varepsilon_A$; recall $=\eta_A/p(A)$, precision $=\eta_A/(\eta_A+\varepsilon_B)$.

## Reject option
- **Ambiguity reject**: refuse objects near the boundary (posteriors about equal).
- **Outlier reject**: refuse objects far from all training data (low $p(x)$).
- **Reject curve**: error $\varepsilon_r$ vs rejected fraction $r$. Rejecting lowers the error but rejection has a cost too.
- Optimal amount: total cost $c=c_r\,r+c_\varepsilon\,\varepsilon$ is a straight line in the $(r,\varepsilon)$ plane; slide it until it touches the reject curve.

## ROC curve
![[roc.png]]
Sweep the decision threshold $d$ in $S(x)-d=0$ and plot the two class errors against each other (or TPR vs FPR).
- Each threshold is one **operating point**.
- Use when priors or misclassification costs are unknown or will change, and to compare classifiers.
- **AUC**: area under the curve. Perfect = 1.0, random = 0.5. **Insensitive to class priors.**
- Any point on the line between two classifiers' operating points can be realised by randomly using one a fraction $\alpha$ of the time and the other $1-\alpha$.

## Likely exam questions
- True or false: "training error is a good estimate of true error" (false) and "a good estimate of the true error is all we are after" (false: cost, per-class errors, reject, ROC matter).
- Sketch a learning curve for a simple and a complex classifier and explain the crossing.
- Compute precision, recall, specificity and error from a confusion matrix.
- Explain k-fold vs LOO vs hold-out; bias and variance of each estimate.
- What is a surrogate loss and why is it needed?
- What does AUC measure and why is it prior-independent?

## Books
PRML 1.3 (model selection, cross-validation), 1.5.3 (reject option), 1.5.1. DL book 5.3 (validation sets, cross-validation), 11.1 (performance metrics, precision/recall).

## Flashcards
#flashcards/MDL/Week4

Apparent error vs true error::Apparent, error on the training set (optimistically biased). True, error on the unseen distribution.

What is a surrogate loss?::A convenient loss optimised instead of the classification error, which cannot be optimised directly

What do LDA, QDA, nearest mean and logistic optimise?::Maximum likelihood

Large training set vs large test set::Large training set gives a good classifier. Large test set gives a reliable error estimate.

Error estimate from a small independent test set::Unbiased but unreliable (large variance)

n-fold cross-validation::Split the data in $n$ parts, train on $n-1$, test on the rest, rotate $n$ times, average the errors

Leave-one-out::Cross-validation with $n=N$. Nearly unbiased, but expensive and high variance.

What is a learning curve?::Error (train and test) plotted against training set size

Learning curve behaviour::True error decreases, apparent error increases, both converge to the asymptotic error

Complex vs simple classifier on a learning curve::Complex has lower training error, higher test error for small sets and lower asymptotic error, so the curves cross

What is a feature curve?::Error against number of features or complexity at a fixed training set size; the true error is U-shaped

Confusion matrix entry $c_{ij}$::Number of objects of true class $i$ classified as class $j$

Recall (sensitivity)::$\dfrac{TP}{TP+FN}$, the fraction of the target class that is found

Precision::$\dfrac{TP}{TP+FP}$, the fraction of objects assigned to the class that really belong to it

Specificity::$\dfrac{TN}{TN+FP}$, the performance on objects outside the target class

Ambiguity reject vs outlier reject::Ambiguity, reject objects near the boundary (equal posteriors). Outlier, reject objects far from all training data.

What is a reject curve?::Error $\varepsilon_r$ against rejected fraction $r$. Rejecting lowers the error but has its own cost.

How is an ROC curve made?::Vary the classifier threshold and plot true positive rate against false positive rate (or the two class errors)

When is ROC analysis useful?::When priors or costs are unknown or changing, and to compare or combine classifiers

AUC values::1.0 for a perfect classifier, 0.5 for random. Insensitive to class priors.

Standard deviation of a test error estimate::About $\sqrt{\varepsilon(1-\varepsilon)/N_{test}}$

Three sources of variation in a measured error::The training set drawn, the test set drawn, randomness in the training algorithm
