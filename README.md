# HC-DBSCAN
We propose a constrained hyperparameter optimization method for density-based spatial clustering
of applications with noise (DBSCAN) called HC-DBSCAN. Transforming prior knowledge into
constraints can improve clustering quality through constrained clustering. Previous works on con-
strained clustering mainly use instance-level constraints since adding cluster-level ones to the model
is not straightforward. Recently a few studies have started to deal with cluster-level constraints via
optimization models. However, to apply constrained clustering to DBSCAN through optimization,
existing evaluation metrics are computationally too expensive due to its non-convex clusters support.
In addition, we want to consider noise instances not assigned to any clusters. As such, we introduce
an evaluation metric called penalized Davies-Bouldin score with computational cost O(N ) that can
efficiently handle a noise cluster. Due to its intrinsic nature, the penalized Davies-Bouldin score is
adequate for evaluating convex clusters despite its low computational cost. We combine penalized
Davies-Bouldin score with suitable constraints so that the proposed method supports non-convex
clusters. We empirically show that the penalized Davies-Bouldin score surrogates NMI, an evaluation
metric that uses true label, in a restricted feasible region, although it assumes a convex-shape cluster.
We formulate a constrained hyperparameter optimization problem for DBSCAN with a Bayesian
optimization model. We develop an algorithm using the Alternating direction method of multiplier
Bayesian Optimization (ADMMBO) to solve the optimization problem. We customize ADMMBO to
our problem. HC-DSBCAN deals with the integer type hyperparameter, and we design the constraint
function to consider whether the result violates the constraint and how much. Finally, we implement
the Bayesian inference in ADMMBO to match the characteristics of hyperparameter optimization
for DBSCAN. Our algorithm performs well and shows good clustering performance compared with
benchmark constrained methods for DBSCAN on simulated and real datasets.
