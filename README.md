# HC-DBSCAN
We propose a constrained hyperparameter optimization method for density-based spatial clustering
of applications with noise (DBSCAN) called HC-DBSCAN.

We formulate a constrained hyperparameter optimization problem for DBSCAN with a Bayesian
optimization model. We develop an algorithm using the Alternating direction method of multiplier
Bayesian Optimization (ADMMBO) to solve the optimization problem. We customize ADMMBO to
our problem. HC-DSBCAN deals with the integer type hyperparameter, and we design the constraint
function to consider whether the result violates the constraint and how much. Finally, we implement
the Bayesian inference in ADMMBO to match the characteristics of hyperparameter optimization
for DBSCAN. 

Our algorithm performs well and shows good clustering performance compared with
benchmark constrained methods for DBSCAN on simulated and real datasets.


## Requirements
Pytorch

GPytorch

Scikit-learn

Pandas

Numpy

matplotlib

UMAP

DBCV
