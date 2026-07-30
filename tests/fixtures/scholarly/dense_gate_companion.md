# Dense and Sparse Gates: A Controlled Comparison
Authors: Nia Brooks, Omar Lee
Year: 2025

## Abstract
We compare dense and sparse gates for tabular prediction on the SpiralBench
dataset.

## Method and Experimental Setup
The DenseGate model uses a 3-layer encoder with hidden dimension 64. We use SGD
optimizer with learning rate 0.01 and batch size 64. The SpiralBench dataset is
split into 1,100 training, 300 validation, and 400 test examples. Accuracy is
the primary metric and SGN is the baseline.

## Results
DenseGate achieves 90.1% accuracy on the test split. SGN achieves 90.8% under
this companion protocol. These values use one run and no uncertainty is
reported.

## Component Analysis
Without the second dense layer, DenseGate achieves 87.3% accuracy.

## Limitations and Future Work
The comparison uses a different split from prior SGN experiments. Future work
will compare equal compute budgets and multiple random seeds.

## References
[1] M. Chen and P. Ortiz. Sparse Gate Networks. 2025. Open Methods Workshop.
