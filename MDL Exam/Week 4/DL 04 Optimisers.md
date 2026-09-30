---
tags: [MDL, DL, exam]
---
# DL 04 · Optimisers: momentum, RMSProp, Adam

Back to [[00 MDL Index]] · previous [[DL 03 Backpropagation]]
Source: **Assignment 5 only**. I have no lecture slides for the Optimization lecture, so treat this as a core summary and check it against the slides.

## Big picture
Plain [[Stochastic gradient descent|SGD]] uses only the current mini-batch gradient. It zig-zags in valleys that are steep in one direction and flat in another, and the gradient is noisy. The three optimisers fix this with **running averages** of past gradients.

![[optimizers.png]]

## Exponentially weighted moving average (EWMA)
$$S_t=\rho\,S_{t-1}+(1-\rho)\,y_t$$
- $\rho\in[0,1)$ is the decay. It averages over roughly the last $\dfrac1{1-\rho}$ values ($\rho=0.9$ → about 10).
- Larger $\rho$ = smoother but slower to react.
- Starting from $S_0=0$ biases the first values toward zero. **Bias correction**: $\hat S_t=\dfrac{S_t}{1-\rho^t}$.

In the formulas below $\nabla_\theta$ is the mini-batch gradient, $\epsilon$ the [[Learning rate|learning rate]], $\delta$ a small constant for numerical stability, and squares/roots are element-wise.

## SGD with momentum
Average the **gradient**:
$$v_i=\rho\,v_{i-1}+(1-\rho)\,\nabla_\theta,\qquad\theta'=\theta-\epsilon\,v_i$$
Directions that keep the same sign accumulate; directions that flip sign cancel. Damps the zig-zag and carries through flat regions and noise.

## RMSProp
Average the **squared gradient** and divide by its root:
$$r_i=\rho\,r_{i-1}+(1-\rho)\,\nabla_\theta^2,\qquad\theta'=\theta-\epsilon\,\frac{\nabla_\theta}{\sqrt{r_i+\delta}}$$
A per-parameter learning rate: parameters with consistently large gradients take smaller steps, parameters with small gradients take larger ones.

## Adam = momentum + RMSProp + bias correction
$$v_i=\rho_1v_{i-1}+(1-\rho_1)\nabla_\theta,\qquad\hat v_i=\frac{v_i}{1-\rho_1^{\,i}}$$
$$r_i=\rho_2r_{i-1}+(1-\rho_2)\nabla_\theta^2,\qquad\hat r_i=\frac{r_i}{1-\rho_2^{\,i}}$$
$$\theta'=\theta-\epsilon\,\frac{\hat v_i}{\sqrt{\hat r_i+\delta}}$$
Typical defaults: $\rho_1=0.9$, $\rho_2=0.999$, $\delta=10^{-8}$.

Useful fact: on the very first step $\hat v_1=\nabla_\theta$ and $\hat r_1=\nabla_\theta^2$, so the step is $\approx\epsilon\cdot\operatorname{sign}(\nabla_\theta)$. [[Adam]]'s step size is about $\epsilon$ regardless of the gradient's scale.

| | Keeps average of | Effect |
|---|---|---|
| SGD | nothing | noisy, zig-zags |
| Momentum | gradient (1st moment) | smooths direction, builds speed |
| RMSProp | squared gradient (2nd moment) | rescales step per parameter |
| Adam | both, bias-corrected | both effects, robust default |

> [!note] Other forms you may see
> The DL book writes [[Momentum|momentum]] as $v\leftarrow\alpha v-\epsilon g$, $\theta\leftarrow\theta+v$ (no $1-\rho$ factor), and puts $\delta$ outside the square root. Same idea, differently scaled. Use the form from the course.

## Likely exam questions
- Compute 2–3 [[Exponentially weighted moving average|EWMA]] steps by hand, with and without bias correction.
- One update step of momentum / [[RMSProp]] / Adam for given numbers.
- Why is bias correction needed and when does it stop mattering? ($\rho^t\to0$ after enough steps.)
- What does each optimiser fix about plain SGD?

## Books
DL book 8.3 (8.3.1 SGD, 8.3.2 momentum), 8.5 (8.5.2 RMSProp, 8.5.3 Adam). UDL 6.3 (momentum), 6.4 (Adam). D2L chapter 12.

## Flashcards
#flashcards/MDL/Week4

EWMA::$S_t=\rho\,S_{t-1}+(1-\rho)\,y_t$

Roughly how many values does an EWMA average over?::About $1/(1-\rho)$, e.g. 10 for $\rho=0.9$

EWMA bias correction::$\hat S_t=\dfrac{S_t}{1-\rho^t}$, because starting at $S_0=0$ biases early values toward zero

SGD with momentum::$v_i=\rho v_{i-1}+(1-\rho)\nabla_\theta$, $\theta'=\theta-\epsilon v_i$

What does momentum fix?::It averages out directions that flip sign (zig-zag, noise) and accumulates consistent directions

RMSProp::$r_i=\rho r_{i-1}+(1-\rho)\nabla_\theta^2$, $\theta'=\theta-\epsilon\dfrac{\nabla_\theta}{\sqrt{r_i+\delta}}$

What does RMSProp do?::Gives each parameter its own step size, smaller where gradients are consistently large

Adam first moment::$v_i=\rho_1v_{i-1}+(1-\rho_1)\nabla_\theta$, $\hat v_i=v_i/(1-\rho_1^i)$

Adam second moment::$r_i=\rho_2r_{i-1}+(1-\rho_2)\nabla_\theta^2$, $\hat r_i=r_i/(1-\rho_2^i)$

Adam update::$\theta'=\theta-\epsilon\dfrac{\hat v_i}{\sqrt{\hat r_i+\delta}}$

Adam in words::Momentum plus RMSProp plus bias correction

Typical Adam hyperparameters::$\rho_1=0.9$, $\rho_2=0.999$, $\delta=10^{-8}$

Size of Adam's very first step::About $\epsilon$ per parameter, since $\hat v_1/\sqrt{\hat r_1}=\operatorname{sign}(\nabla_\theta)$

Purpose of $\delta$ in RMSProp and Adam::Numerical stability, it avoids division by zero
