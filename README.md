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
This list represents our experiment settings with HC-DBSCAN. We recommend to import equal or higher version of each package to utilize the HC-DBSCAN method.
* Python 3.9.6
* Pytorch 1.9.0
* GPytorch 1.5.1
* scikit-learn 1.0.2
* pandas 1.3.2
* numpy 1.20.3
* matplotlib 3.4.2
* UMAP 0.5.2
* scipy 1.7.1

We also provide whole experiments results and process. So if you want to check the entire process, the following packages are required.

* imageio 2.16.1
* torchvision 0.10.0
* DBCV 


## Development

To run the unit test for MNIST dataset,

> python HC-DBSCAN_MNIST.py

This code returns a clustering results of HC-DBSCAN and benchmarks with a constraint on the number of clusters and 5 cannot-link constraints.

If you want to run the HC-DBSCAN for other dataset (Ex:dry_bean),

> python HC-DBSCAN_benchmarks.py --data_name=dry_bean

We provide an experimental environment for all datasets presented in our paper, "Constrained Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using Hyperparameter Optimization".
