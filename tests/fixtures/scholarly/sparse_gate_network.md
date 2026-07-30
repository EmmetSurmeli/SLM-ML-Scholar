# Sparse Gate Networks for Sequence Classification
Authors: Mira Chen, Pavel Ortiz
Year: 2025
Venue: Open Methods Workshop
Keywords: sparse gating, sequence classification, reproducibility
Identifier: 10.1234/localml.sgn

## Abstract
We propose Sparse Gate Networks (SGN) for compact sequence classification. In
our experiments, SGN achieves 91.2% accuracy on the SpiralBench dataset.

## Introduction
Our contribution is a gated encoder whose selected features remain inspectable.
Prior work [1] studies dense encoders, while Doe and Roe (2022) study pruning.

## Method
We assume that examples are independent and identically distributed. Let x_i be
the input vector. We define g as the feature gate.

$$
g = \sigma(W_g x_i + b_g) \tag{1}
$$

where W_g is the gate projection matrix and b_g denotes the gate bias. The
hidden state is

\[
h_i = g \odot x_i. \tag{2}
\]

where h_i denotes the gated representation. The objective is
$L = -\sum_i y_i \log p_i$.

## Algorithm
Input: training examples and labels
Output: trained SGN parameters
1. Initialize the gate projection.
2. Compute gated representations.
3. Minimize the objective with Adam.
4. Return the lowest-validation-loss checkpoint.

## Experiments
We use the SpiralBench dataset with 1,200 training examples, 200 validation
examples, and 400 test examples. Features are standardized using training-set
means. We use Adam optimizer with learning rate 0.001, batch size 32, weight
decay 0.01, dropout 0.10, 20 epochs, random seed 7, and 5 runs. The hidden
dimension is 64 and the model has 2 layers. Accuracy and F1 are reported against
the DenseNet baseline.

Table 1: Test results
| System | Accuracy | F1 |
|---|---:|---:|
| DenseNet | 88.0% | 0.874 |
| SGN | 91.2% | 0.907 |

## Ablation
Removing the feature gate reduces accuracy to 86.4%. Replacing the gate with a
constant mask produces 84.9% accuracy.

## Limitations
The method is restricted to fixed-length vectors and does not interpret raw
images. Training requires a preprocessing pass. Future work will evaluate
variable-length data.

## Appendix: Sensitivity
For the sensitivity experiment, the learning rate is 0.0005. Here q represents
the quantile level.

$$
q = W_q h_i. \tag{3}
$$

In Equation 3, q represents the query vector.

## References
[1] A. Author. Dense Encoders. 2021. Open Archive.
[2] Doe and Roe. Structured Pruning. 2022. Methods Notes.
