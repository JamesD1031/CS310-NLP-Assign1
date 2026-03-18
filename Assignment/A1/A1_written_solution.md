# Assignment 1 Written Solution

## 1. When \(k = 2\), softmax cross-entropy reduces to BCE

Let the logits be \(z_1, z_2\). The softmax outputs are

\[
\hat{y}_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}},
\qquad
\hat{y}_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2}}.
\]

For binary classification, let the scalar label be \(y \in \{0,1\}\), and write the one-hot target as

\[
\mathbf{t} = [t_1, t_2] = [1-y,\; y].
\]

The softmax cross-entropy loss is

\[
\mathcal{L}
= -\sum_{i=1}^{2} t_i \log \hat{y}_i
= -(1-y)\log \hat{y}_1 - y \log \hat{y}_2.
\]

Now define

\[
a = z_2 - z_1.
\]

Then

\[
\hat{y}_2
= \frac{e^{z_2}}{e^{z_1}+e^{z_2}}
= \frac{1}{1 + e^{-(z_2-z_1)}}
= \sigma(a),
\]

where \(\sigma(\cdot)\) is the sigmoid function. Also,

\[
\hat{y}_1 = 1 - \hat{y}_2 = 1 - \sigma(a).
\]

Substitute these into the loss:

\[
\mathcal{L}
= -(1-y)\log(1-\sigma(a)) - y\log(\sigma(a)).
\]

Reordering the terms gives

\[
\mathcal{L}
= -\Big[y\log(\sigma(a)) + (1-y)\log(1-\sigma(a))\Big].
\]

This is exactly the binary cross-entropy loss:

\[
\mathcal{L}_{\text{BCE}}
= -\Big[y\log p + (1-y)\log(1-p)\Big],
\quad \text{with } p=\sigma(a).
\]

Therefore, when \(k=2\), softmax cross-entropy is equivalent to BCE.

## 2. Gradient with respect to logit \(z_i\)

For general \(k\), define

\[
\hat{y}_j = \frac{e^{z_j}}{\sum_{\ell=1}^{k} e^{z_\ell}},
\qquad
\mathcal{L} = -\sum_{j=1}^{k} y_j \log \hat{y}_j,
\]

where \(\mathbf{y}\) is the one-hot target vector.

First compute the derivative of the softmax output:

\[
\frac{\partial \hat{y}_j}{\partial z_i}
= \hat{y}_j(\delta_{ij} - \hat{y}_i),
\]

where \(\delta_{ij}\) is the Kronecker delta.

Now apply the chain rule:

\[
\frac{\partial \mathcal{L}}{\partial z_i}
= \sum_{j=1}^{k} \frac{\partial \mathcal{L}}{\partial \hat{y}_j}
\frac{\partial \hat{y}_j}{\partial z_i}.
\]

Since

\[
\frac{\partial \mathcal{L}}{\partial \hat{y}_j}
= -\frac{y_j}{\hat{y}_j},
\]

we get

\[
\frac{\partial \mathcal{L}}{\partial z_i}
= \sum_{j=1}^{k}
\left(-\frac{y_j}{\hat{y}_j}\right)
\hat{y}_j(\delta_{ij} - \hat{y}_i).
\]

Cancel \(\hat{y}_j\):

\[
\frac{\partial \mathcal{L}}{\partial z_i}
= -\sum_{j=1}^{k} y_j(\delta_{ij} - \hat{y}_i).
\]

Expand the sum:

\[
\frac{\partial \mathcal{L}}{\partial z_i}
= -\sum_{j=1}^{k} y_j\delta_{ij}
+ \sum_{j=1}^{k} y_j\hat{y}_i.
\]

Because \(\sum_{j=1}^{k} y_j = 1\) for a one-hot label vector,

\[
\sum_{j=1}^{k} y_j\delta_{ij} = y_i,
\qquad
\sum_{j=1}^{k} y_j\hat{y}_i = \hat{y}_i.
\]

So the final result is

\[
\boxed{
\frac{\partial \mathcal{L}}{\partial z_i}
= \hat{y}_i - y_i
}
\]

for every class \(i = 1, \dots, k\).
