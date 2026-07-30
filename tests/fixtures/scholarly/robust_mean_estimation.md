# Robust Mean Estimation under Contamination
Authors: Lena Morris, Arun Shah
Year: 2024

## Abstract
We study robust location estimation when observations include bounded
contamination. We show that the trimmed estimator reduces mean squared error in
the authored GaussianShift dataset.

## Assumptions and Theory
Suppose that observations x_i are independent with finite variance. Under the
assumption that the contamination fraction satisfies $\epsilon \leq 0.1$, let
\mu denote the population mean and let n be the sample count.

$$
\hat{\mu}_\tau = \frac{1}{n-2k}\sum_{i=k+1}^{n-k} x_{(i)} \tag{1}
$$

where k denotes the trimming count and \tau represents the trim fraction. The
estimator holds when $2k < n$.

## Procedure
1. Sort the observations.
2. Remove k values from each tail.
3. Average the retained observations.

## Evaluation
The GaussianShift dataset contains 2,000 samples. We compare the TrimmedMean
method with the SampleMean baseline. Mean squared error and calibration error
are reported over 10 trials.

System,Mean squared error,Calibration error
SampleMean,0.42,0.18
TrimmedMean,0.25,0.11

Our results indicate that TrimmedMean achieves mean squared error 0.25 with a
95% confidence interval of [0.22, 0.28].

## Discussion and Limitations
The estimator is sensitive to asymmetric contamination and requires the trim
fraction to be selected. We do not evaluate high-dimensional observations.

## References
[1] R. Example. Robust Statistics. 2020. Open Notes.
