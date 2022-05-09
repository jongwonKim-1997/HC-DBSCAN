#
# author: Jongwon Kim (pioneer0517@postech.ac.kr)
# last updated: July 02, 2023
#


#%%
from tokenize import String
from pandas.core.construction import is_empty_data

from torch.functional import _return_counts
#from numpy.core.fromneric import _alen_dispathcer
#import numpy as np
#from numpy.lib.type_check import _asfarray_dispatcher
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.kernels import ScaleKernel, MaternKernel
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from DBCV import DBCV
import sklearn.cluster
import sklearn.metrics
import umap
from sklearn.decomposition import PCA
from scipy.stats import norm
#import hdbscan
import argparse
# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The ExponentiatedQuadratic kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#%%

import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



import pandas as pd
import imageio
#%%
from functools import reduce
import math

import torch

from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels import Kernel

class MaternKernel2(Kernel):

    has_lengthscale = True

    def __init__(self, nu=2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel2, self).__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        x1 = torch.round(x1)
        x2 = torch.round(x2)
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
#            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1).div(self.lengthscale)
            x2_ = (x2).div(self.lengthscale)
            
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )
#%%


from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels import Kernel


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class RBFKernel2(Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        x1 = torch.round(x1)
        x2 = torch.round(x2)
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
              x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )
#%%
start_date = "final"
class AdvancedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,n_hyp):
        super(AdvancedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims = torch.tensor([0])  ) * MaternKernel2(active_dims = torch.tensor([1]) ) )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
# %%
import torchvision.datasets as dsets
import torchvision.transforms as transforms
#%%
device = 'cpu'
# 0. Import Data
def import_data(data='mnist', size=1000):
    # MNIST

    if data == "mnist":
        mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                                train=True, # True를 지정하면 훈련 데이터로 다운로드
                                transform=transforms.ToTensor(), # 텐서로 변환
                                download=True)

      
        batch_size = 60000
        data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
                            
        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device).numpy()
            Y = Y.to(device).numpy()
        train_data = X.reshape(len(X),-1)
        train_labels = Y 
    # REUTERS   
    if data =='reuters':
        train_data = pd.read_csv("./data/20_newsgroup.csv")
        train_data = train_data.dropna()
        train_data['labels'] = pd.Categorical(train_data.copy()['labels']).codes
        train_labels = train_data['labels']
        train_data = train_data.drop(['labels'],axis=1)

    if data =="cifar10":
        mnist_train = dsets.CIFAR10(root='MNIST_data/', # 다운로드 경로 지정
                                train=True, # True를 지정하면 훈련 데이터로 다운로드
                                transform=transforms.ToTensor(), # 텐서로 변환
                                download=True)

        batch_size = 50000
        data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
                            
        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device).numpy()
            Y = Y.to(device).numpy()
        train_data = X.reshape(len(X),-1)
        train_labels = Y 

    if data =="FashionMNIST":
        mnist_train = dsets.FashionMNIST(root='FashionMNIST/', # 다운로드 경로 지정
                                train=True, # True를 지정하면 훈련 데이터로 다운로드
                                transform=transforms.ToTensor(), # 텐서로 변환
                                download=True)

        batch_size = 60000
        data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
                            
        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device).numpy()
            Y = Y.to(device).numpy()
        train_data = X.reshape(len(X),-1)
        train_labels = Y 

    if data =='iris':
        train_data = pd.read_csv("./data/iris.csv")
        train_data = train_data.dropna()
        train_labels = train_data['iris']
        train_data['iris'] = pd.Categorical(train_data.copy()['iris']).codes
        train_labels = train_data['iris']
        train_data = train_data.drop(['iris'],axis=1)
    if data =='Cell237':
        train_data = pd.read_csv("./data/Cell237.csv")
        train_data = train_data.dropna()
        train_labels = train_data['class']
        train_data['class'] = pd.Categorical(train_data.copy()['class']).codes
        train_labels = train_data['class']
        train_data = train_data.drop(['class'],axis=1)


    if data =='seeds':
        train_data = pd.read_csv("./data/seeds.csv")
        train_data = train_data.dropna()
        train_labels = train_data['seeds']
        train_data = train_data.drop(['seeds'],axis=1)

    if data =='table_1':
        train_data = pd.read_csv("./data/abalone.csv")
        train_data = train_data.dropna()
        train_labels = train_data['Rings']
        train_data = train_data.drop(['Rings'],axis=1)

    if data =='table_2':
        train_data = pd.read_csv("./data/Data_Cortex_Nuclear.csv")
        train_data = train_data.dropna()
        train_labels = train_data['class']
        train_data['class'] = pd.Categorical(train_data.copy()['class']).codes
        train_labels = train_data['class']
        train_data = train_data.drop(['class'],axis=1)

    if data =='table_3':
        train_data = pd.read_csv("./data/Dry_Bean_Dataset.csv")
        train_data = train_data.dropna()
        train_labels = train_data['Class']
        train_data['Class'] = pd.Categorical(train_data.copy()['Class']).codes
        train_labels = train_data['Class']
        train_data = train_data.drop(['Class'],axis=1)

    if data =='table_4':
        train_data = pd.read_csv("./data/Faults.csv")
        train_data = train_data.dropna()
        train_labels = train_data['class']
        train_data['class'] = pd.Categorical(train_data.copy()['class']).codes
        train_labels = train_data['class']
        train_data =  train_data.drop(['class'],axis=1)

    if data =='table_5':    
        train_data = pd.read_csv("./data/Frogs_MFCCs.csv")
        train_data = train_data.dropna()
        train_labels = train_data['Species']
        train_data['Species'] = pd.Categorical(train_data.copy()['Species']).codes
        train_labels = train_data['Species']
        train_data = train_data.drop(['Species','Family','Genus'],axis=1)

    if data == 'toy1':

        im = imageio.imread('test_2.png')
        #im = imageio.imread('test_smile_face.png')
        im_data = np.sum(im,axis=2)
        im_data_list = []
        for i in range(400):
            for j in range(400):
                if im_data[i,j] !=765:
                    im_data_list.append([i,j])
        train_data = np.array(im_data_list)
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy2':

        im = imageio.imread('test_smile_face.png')
        im_data = np.sum(im,axis=2)
        im_data_list = []
        for i in range(400):
            for j in range(400):
                if im_data[i,j] !=765:
                    im_data_list.append([i,j])
        train_data = np.array(im_data_list)
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy3':
        train_data = pd.read_csv("./data/Compound.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy4':
        train_data = pd.read_csv("./data/pathbased.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy5':
        train_data = pd.read_csv("./data/Aggregation.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy6':
        train_data = pd.read_csv("./data/spiral.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])


        

   #print(train_data.shape)
    size = min(size,len(train_data))
    rnd_idx = np.random.RandomState(seed=2022).permutation(len(train_data))[:size]
    raw_data = train_data
    raw_data = np.array(raw_data)
    train_data = raw_data[rnd_idx]
    raw_labels =np.array(train_labels)
    
   #print(raw_labels.shape)

    train_labels = raw_labels[rnd_idx]
    
    train_labels = train_labels.reshape(-1)

    return train_data, train_labels

#%%
def embedding_data(train_data ,embedding = 'umap',n_components=2):


    # UMAP
    if embedding == 'umap':
        train_data = umap.UMAP(n_components = n_components,n_neighbors=10, min_dist=0.001).fit_transform(train_data)

    # PCA 
    if embedding == 'pca':
        pca = PCA(n_components=n_components)
        train_data = pca.fit_transform(train_data)

    return train_data

# 1. Clustering Method
def clustering(clustering_method = 'dbscan',hyp_dict=None,lambda1 =1):
    if clustering_method == 'kmeans':
        return sklearn.cluster.KMeans(**hyp_dict)
    if clustering_method == 'dbscan':
        return sklearn.cluster.DBSCAN(**hyp_dict)
    if clustering_method == 'hdbscan':
        hyp_dict['min_cluster_size'] = int(hyp_dict['min_cluster_size'])+2
        hyp_dict['cluster_selection_epsilon'] = float(hyp_dict['cluster_selection_epsilon'])
        hdbscanner = hdbscan.HDBSCAN(**hyp_dict)
        hdbscanner.predict = hdbscanner.fit_predict
        return hdbscanner

# 3. Clustering Metric
def metric(train_data, labels,train_labels=None, metric_method = 'davies_bouldin_score',noise=True,lambda1=1):
    n_labels = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    non_noise_rate = sum(labels!=-1)/n_labels
    non_noise_idx = (labels!=-1)
    if n_clusters<=1:
        return 10

    if metric_method == 'davies_bouldin_score':
        if noise ==True:
            return -np.exp(-sklearn.metrics.davies_bouldin_score(train_data[non_noise_idx],labels[non_noise_idx])) *  non_noise_rate
        else :
            return -np.exp(-sklearn.metrics.davies_bouldin_score(train_data[non_noise_idx],labels[non_noise_idx]))
    elif metric_method == 'silhouette_score':
        if noise ==True:
            return -sklearn.metrics.silhouette_score(train_data, labels, metric='euclidean')  * non_noise_rate
        else :
            return -sklearn.metrics.silhouette_score(train_data, labels, metric='euclidean')
    elif metric_method == 'normalized_mutual_info_score':
#        if noise == True:
#            return -sklearn.metrics.normalized_mutual_info_score(train_labels[non_noise_idx],labels[non_noise_idx]) * non_noise_rate
#        else : 
        return -sklearn.metrics.normalized_mutual_info_score(train_labels[non_noise_idx],labels[non_noise_idx])
    elif metric_method =='metrics.calinski_harabasz_score(X, labels)':
        if noise ==True:
            return -sklearn.metrics.calinski_harabasz_score(train_data[non_noise_idx], labels[non_noise_idx])  * non_noise_rate
        else :
            return -sklearn.metrics.calinski_harabasz_score(train_data[non_noise_idx], labels[non_noise_idx])
    elif metric_method == 'DBCV':
        return DBCV(train_data,labels)
    elif metric_method == 'normalized_mutual_info_score_20':
        
        rnd_idx = np.random.RandomState(seed=1).permutation(len(train_labels))[: int(len(train_labels)*0.2) ]
        if noise == True:
            return -sklearn.metrics.normalized_mutual_info_score(train_labels[rnd_idx][labels[rnd_idx]!=-1],labels[rnd_idx][labels[rnd_idx]!=-1]) + lambda1 * non_noise_rate
        else : 
            return -sklearn.metrics.normalized_mutual_info_score(train_labels[rnd_idx][labels[rnd_idx]!=-1],labels[rnd_idx][labels[rnd_idx]!=-1])
    elif metric_method == 'normalized_mutual_info_score_40':
        rnd_idx = np.random.RandomState(seed=2).permutation(len(train_labels))[: int(len(train_labels)*0.4) ]
        if noise == True:
            return -sklearn.metrics.normalized_mutual_info_score(train_labels[rnd_idx][labels[rnd_idx]!=-1],labels[rnd_idx][labels[rnd_idx]!=-1]) + lambda1 * non_noise_rate
        else : 
            return -sklearn.metrics.normalized_mutual_info_score(train_labels[rnd_idx][labels[rnd_idx]!=-1],labels[rnd_idx][labels[rnd_idx]!=-1])

    else:
        raise NameError('HiThere')
        return "Your do wrong"


def round_1(x):
    for idx_ in integer_var:
        tmp = tf.round(tf.slice(x,[0,idx_],[-1,1]))
        columns = [idx_]
        columns = tf.convert_to_tensor(columns)
        rows = tf.range(tf.shape(x)[0], dtype=columns.dtype)
        ii, jj = tf.meshgrid(rows, columns, indexing='ij')
        x = tf.tensor_scatter_nd_update(x,tf.stack([ii, jj], axis=-1),tmp)
    return x

def predict_with_optimized_hyps(X_train, U_train, X_test,n_hyp,hyp_setting):
    U_train = U_train.reshape(-1)
    train_x = torch.tensor(X_train)
    train_y = torch.tensor(U_train)
    

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = AdvancedGPModel(train_x, train_y, likelihood,n_hyp).double()
    hypers = {
        'covar_module.base_kernel.kernels.0.lengthscale' : torch.tensor(hyp_setting[0]),
        'covar_module.base_kernel.kernels.1.lengthscale' : torch.tensor(hyp_setting[1]),
        'likelihood.noise_covar.noise': torch.tensor(hyp_setting[2]),
        'covar_module.outputscale': torch.tensor(hyp_setting[3]),
    }
    model.initialize(**hypers)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()    

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses = []
    iteration_n = 200
    for i in range(iteration_n):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        losses.append(loss.item())

        ##print('Iter {}/{} - Loss: {}   outputscale: {} lengthscale_xy: {} lengthscale_z: {}.  noise: {}'.format(
        #     i + 1, iteration_n, loss.item(),
        #     model.covar_module.outputscale.item(),
        #     model.covar_module.base_kernel.kernels[0].lengthscale.item(),
        #     model.covar_module.base_kernel.kernels[1].lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()
    
    m_eval = model.eval()
    X_test = torch.tensor(X_test)
    temp_posterior = m_eval(X_test)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mu = temp_posterior.mean.detach().numpy()
        sigma = temp_posterior.variance.detach().numpy()**0.5
        hyp_setting = [model.covar_module.base_kernel.kernels[0].lengthscale.item(),
            model.covar_module.base_kernel.kernels[1].lengthscale.item(),
            model.likelihood.noise_covar.noise.item(),
            model.covar_module.outputscale.item()]
    return mu, sigma, sigma , hyp_setting

# Constrained Clustering
def ADMMBO(train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,initial_index=0):
    hyp_setting = np.array([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],1,1])
    hyp_setting_F = hyp_setting 
    hyp_setting_C = hyp_setting
    bounds = bounds
    hyp_key = hyp_dict.keys()
    n_hyp = len(bounds)
    bounds = np.array(bounds)


    n_constraint = len(constraint_function_list)

    
    # Initialize
    X_init = []
    
    for lower, upper in bounds:
        X_init.append((np.random.RandomState(seed=initial_index*10+5).uniform((lower),(upper),n_init)))
    X_init = np.array(X_init).T
    X_train = np.empty((n_init+(alpha+n_constraint*beta)*n_iter,n_hyp))
    X_n = 0
    F_n = 0
    C_n = 0
    X_train[:n_init] = X_init
    Y_train =[]
    X_n = n_init
    n_labels = len(np.unique(train_labels))
    label_max = max(np.unique(train_labels,return_counts=True)[1])


    F_train = np.empty((n_init+(alpha+n_constraint*beta)*n_iter))
    NMI_train = np.empty((n_init+(alpha+n_constraint*beta)*n_iter))
    C_train = np.empty((n_constraint,n_init+(alpha+n_constraint*beta)*n_iter))
    real_C_train = np.empty((n_constraint,n_init+(alpha+n_constraint*beta)*n_iter))

    for hyp_set in X_train[:n_init] :
        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = hyp_set[idx_]
        cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

        F_train[F_n] = score
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
            score2 = 10
        else:
            score2 = metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
        NMI_train[F_n] = score2
        F_n +=1
        # Constraints 

        for con_idx in range(n_constraint):

            C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

            real_C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)


        C_n +=1


    # Change minimum value 
    f_idx_list = []
    for i in range(ADMMBO_dict['n_iter']):
        for j in range(alpha):
            f_idx_list.append(j+i*(alpha+beta*n_constraint))
    f_idx_list = np.array(f_idx_list)
    f_idx_list = f_idx_list + n_init
    f_idx_list = np.concatenate([f_idx_list, np.array([i for i in range(n_init)])])


    c_idx_list =[]
    for c_idx in range(n_constraint):
        c_idx_list1 = []
        for i in range(ADMMBO_dict['n_iter']):
            for j in range(alpha+beta*(c_idx),alpha+beta*(c_idx+1)):
                c_idx_list1.append(j+i*(alpha+beta*n_constraint))
        c_idx_list1 = np.array(c_idx_list1)
        c_idx_list1 = c_idx_list1 + n_init
        c_idx_list1 = np.concatenate([c_idx_list1, np.array([i for i in range(n_init)])])
        c_idx_list.append(c_idx_list1)

    ############################################

    U_train = []
    u_min= 100*M
    h_min_list = [100*M for _ in range(n_constraint)]
    H_train = []
    Z_train = []

    X = []
    
    for lower, upper in bounds:
        X.append((np.random.uniform((lower),(upper),1)))
    X = np.array(X).T

    Z = []
    
    for lower, upper in bounds:
        Z.append((np.random.uniform((lower),(upper),n_constraint)))
    Z = np.array(Z).T

    Y = np.zeros(shape=Z.shape)

    ###################################################################
    for iter_temp in range(n_iter):

        for _ in range(alpha):

            
            
            X_test = []
    
            for lower, upper in bounds:
                X_test.append((np.random.uniform((lower),(upper),n_test)))
            X_test = np.array(X_test).T


            # Gaussian Process with u_list

            hyp_setting_F_post = hyp_setting_F

            mu, sigma, Sigma,hyp_setting_F = predict_with_optimized_hyps(X_train[:X_n], F_train[:X_n], X_test,n_hyp,hyp_setting_F)

            #print(np.sqrt(np.sum(np.square(hyp_setting_F- hyp_setting_F_post))))

            L1_list =[]
            mu_H = mu
            for con_idx in range(n_constraint):
                mu_H = mu_H + rho*( np.sqrt(np.sum(np.square(X_test-Z[con_idx]+Y[con_idx]/rho),axis=1)))/2
            
            for index in range(len(mu)):
                mean = mu_H[index]
                std = sigma[index]

                L1 = ((u_min-mean)/std)*norm.cdf((u_min-mean)/std)+norm.pdf((u_min-mean)/std)
                L1_list.append(L1)
            ##print(U_train[:X_n])


            # if iter_temp%10 ==0:
            #     plt.scatter(X_test[:,0],X_test[:,0],alpha= L1_list)
            #    #print(L1)

            next_x  = X_test[np.argmax(L1_list)]
            X_train[X_n] = next_x
            X_n +=1
            X = next_x 
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = next_x[idx_]
            cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                score = 10
            else:
                score = metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

            F_train[F_n] = score
            if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                score2 = 10
            else:
                score2 = metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
            NMI_train[F_n] = score2
            F_n +=1
            # Constraints 
            u_min = min(u_min,min(score + rho*( np.sqrt(np.sum(np.square(X_test-Z[con_idx]+Y[con_idx]/rho),axis=1)))/2))
            for con_idx in range(n_constraint):

                C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

                real_C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)


            C_n +=1

        # Search new point z
        

        next_z =[]
        if constraint == "Hard":
            for c_idx in range(n_constraint):         
                c_idx_list1 = c_idx_list[c_idx]       
                for _ in range(beta):
                    if acquisition_function=='EI':
                        CZ_train = C_train[c_idx]
                        #H_train = (CZ_train>0).astype(int) + rho/(2*M)*np.sqrt(np.sum(np.square(X-X_train+Y[c_idx]/rho),axis=1))
                        
                        h_min = h_min_list[c_idx]
                        
                        L2_list =[]
                        X_test = []
                        for lower, upper in bounds:
                            X_test.append((np.random.uniform((lower),(upper),n_test)))
                        X_test = np.array(X_test).T


                        mu_z, sigma_z, Sigma_z,hyp_setting_C = predict_with_optimized_hyps(X_train[:X_n], CZ_train[:X_n], X_test,n_hyp,hyp_setting_C)
                        
                        for index in range(len(mu_z)):
                            
                            mean = mu_z[index]
                            std = sigma_z[index]

                            q = rho/(2*M)*np.sqrt(np.sum(np.square((X- X_test[index] + Y[c_idx]/rho))))


                            theta = norm.cdf(mean/std)

                            if h_min -q < 0 :
                                L2 = 0
                            elif h_min -q < 1 :
                                L2 = max(0,h_min-q)*(1-theta)
                            else :
                                L2 = max(0,h_min-q)*(1-theta) + max(0,h_min-q-1)*theta

                            
                            L2_list.append(L2)

                        next_z_temp = X_test[np.argmax(L2_list)]
                        


                        # Evaluate new value
                        for idx_, key in enumerate(hyp_key):
                            hyp_dict[key] = next_z_temp[idx_]

                        cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

                        cluster_data = cluster.fit(train_data)
                        labels = cluster_data.labels_
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score = 10
                        else:
                            score = metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

                        F_train[F_n] = score
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score2 = 10
                        else:
                            score2 = metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
                        NMI_train[F_n] = score2

                        # Constraints 

                        for con_idx in range(n_constraint):

                            C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

                            real_C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)


                        h_min_list[c_idx] = min( min( rho/(2*M)*np.sqrt(np.sum(np.square(X-X_test+Y[c_idx]/rho),axis=1)) + int(constraint_function_list[c_idx](cluster_data)>0) ) , h_min_list[c_idx])

                        X_train[X_n] = next_z_temp

                        F_n +=1
                        C_n +=1
                        X_n +=1
                    if acquisition_function=='GP-UCB':
                        CZ_train = C_train[c_idx]
                        H_train = (CZ_train>0).astype(int) + rho/(2*M)*np.sqrt(np.sum(np.square(X-X_train+Y[c_idx]/rho),axis=1))
                        h_min = min(H_train[c_idx_list1[c_idx_list1<=X_n]])
                        L2_list =[]
                        X_test = []
                            
                        for lower, upper in bounds:
                            X_test.append((np.random.uniform((lower),(upper),n_test)))
                        X_test = np.array(X_test).T

                        mu_z, sigma_z, Sigma_z,hyp_setting_C = predict_with_optimized_hyps(X_train[:X_n], CZ_train[:X_n], X_test,n_hyp,hyp_setting_C)
                        
                        for index in range(len(mu_z)):
                            
                            mean = mu_z[index]
                            std = sigma_z[index]

                            q = rho/(2*M)*np.sqrt(np.sum(np.square((X- X_test[index] + Y[c_idx]/rho))))

                            L2 = mean - std * 0.1 + q
                            L2_list.append(L2)

                        next_z_temp = X_test[np.argmin(L2_list)]
                        


                        # Evaluate new value
                        for idx_, key in enumerate(hyp_key):
                            hyp_dict[key] = next_z_temp[idx_]

                        cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

                        cluster_data = cluster.fit(train_data)
                        labels = cluster_data.labels_
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score = 10
                        else:
                            score = metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

                        F_train[F_n] = score
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score2 = 10
                        else:
                            score2 = metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
                        NMI_train[F_n] = score2

                        # Constraints 

                        for con_idx in range(n_constraint):

                            C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

                            real_C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

                        X_train[X_n] = next_z_temp
 
                        F_n +=1
                        C_n +=1
                        X_n +=1
                next_z.append(next_z_temp)
                Z_train.append(next_z)
        elif constraint == "Soft":
            for c_idx in range(len(C_train)):
                c_idx_list1 = c_idx_list[c_idx]    
                for _ in range(beta):
                    CZ_train = C_train[c_idx]
                    if (max(C_train[c_idx][:C_n]) - min(C_train[c_idx][:C_n]) == 0):
                        #CZ_train = C_train[c_idx]
                        H_train = CZ_train+ rho/(2*M)*np.sqrt(np.sum(np.square(X-X_train+Y[c_idx]/rho),axis=1))
                    else :
                        #CZ_train = C_train[c_idx]  /(max(C_train[c_idx][:C_n]  ) - min(C_train[c_idx][:C_n]  ))
                        H_train = CZ_train + rho/(2*M)*np.sqrt(np.sum(np.square(X-X_train+Y[c_idx]/rho),axis=1))

                    L2_list =[]

                    X_test = []
                        
                    for lower, upper in bounds:
                        X_test.append((np.random.uniform((lower),(upper),n_test)))
                    X_test = np.array(X_test).T
                    mu_z, sigma_z, Sigma_z,hyp_setting_C = predict_with_optimized_hyps(X_train[:X_n], CZ_train[:X_n], X_test,n_hyp,hyp_setting_C)

                    q = rho/(2*M)*np.sqrt(np.sum(np.square(X-X_test+Y[c_idx]/rho),axis=1))

                    mean_H = mu_z + q
                    
                    h_min = h_min_list[c_idx]
                    if acquisition_function == "EI":
                        for index in range(len(mu_z)):
                            mean = mean_H[index]
                            std = sigma_z[index]
                            L2 = ((h_min-mean)/std)*norm.cdf((h_min-mean)/std)+norm.pdf((h_min-mean)/std)
                            L2_list.append(L2)
                        
                        next_z_temp = X_test[np.argmax(L2_list)]
                    elif acquisition_function == "GP-UCB":
                        for index in range(len(mu_z)):
                            mean = mean_H[index]
                            std = sigma_z[index]
                            L2 = mean - std * 0.1
                            L2_list.append(L2)
                        
                        next_z_temp = X_test[np.argmin(L2_list)]
                    # Evaluate new value
                    for idx_, key in enumerate(hyp_key):
                        hyp_dict[key] = next_z_temp[idx_]

                    cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

                    cluster_data = cluster.fit(train_data)
                    labels = cluster_data.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                        score = 10
                    else:
                        score = metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

                    F_train[F_n] = score
                    if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                        score2 = 10
                    else:
                        score2 = metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
                    NMI_train[F_n] = score2

                    # Constraints 

                    for con_idx in range(n_constraint):

                        C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

                        real_C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)


                    h_min_list[c_idx] = min( min( constraint_function_list[c_idx](cluster_data) +  q) , h_min_list[c_idx])

                    X_train[X_n] = next_z_temp

                    F_n +=1
                    C_n +=1
                    X_n +=1
        
                
                next_z.append(next_z_temp)

        R = X - next_z
        S = -rho*(next_z - np.array(Z))
        if np.sqrt(np.sum(np.square(R))) > 10 * np.sqrt(np.sum(np.square(S))):
            rho = rho*2
        elif np.sqrt(np.sum(np.square(S))) > 10 * np.sqrt(np.sum(np.square(R))):
            rho = rho/2
        
        Z = np.array(next_z)

        Y = Y + rho*(X-Z)
        Z_train.append(Z)
        Y_train.append(Y)

    return X_train, F_train, C_train, real_C_train, NMI_train,Y_train



# Random Search
def RS_(train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,initial_index=0):
    bounds = bounds
    hyp_key = hyp_dict.keys()
    n_hyp = len(bounds)



    X_init = []
    
    for lower, upper in bounds:
        X_init.append((np.random.RandomState(seed=initial_index*10+5).uniform((lower),(upper),n_iter)))
    X_init = np.array(X_init).T
    X_train = X_init
    F_train = []
    
    C_train = []
    n_constraint = len(constraint_function_list)
    for _ in range(n_constraint):
        C_train.append([])
    NMI_train =[]
    real_C_train1 =[]
    real_C_train2 =[]
    hyp_key = hyp_dict.keys()

    for X_val in X_train:

        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = X_val[idx_]

        cluster = clustering(clustering_method=clustering_method,hyp_dict = hyp_dict)
        
        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        
        # Constraints 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels,metric_method=metric_method)

        for con_idx in range(n_constraint):

            C_score = constraint_function_list[con_idx](cluster_data)
            C_train[con_idx].append(C_score)
        F_train = np.append(F_train, score)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels,metric_method='normalized_mutual_info_score')
        NMI_train = np.append(NMI_train, score)
    C_train = np.array(C_train)
    real_C_train = C_train
    return X_train, F_train, C_train, real_C_train,NMI_train, F_train


def Grid_(train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,initial_index=0):
    hyp_setting = [bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],1,1]

    bounds = bounds
    hyp_key = hyp_dict.keys()
    n_hyp = len(bounds)


    lower, upper = bounds[1]
    y = [i for i in range(int(lower),int(upper))]

    lower, upper = bounds[0]
    x = np.linspace(lower,upper,(int(n_iter/len(y))+1))
    xx, yy = np.meshgrid(x,y)
    mesh_XY = np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)],axis=1)

    X_train = mesh_XY
    X_train = X_train[np.random.permutation(len(X_train))][:n_iter]
    n_constraint = len(constraint_function_list)
    F_train = []
    C_train = []
    n_constraint = len(constraint_function_list)
    for _ in range(n_constraint):
        C_train.append([])

    NMI_train =[]
    hyp_key = hyp_dict.keys()
    for X_val in X_train:

        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = X_val[idx_]

        cluster = clustering(clustering_method=clustering_method,hyp_dict = hyp_dict)
        
        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        
        # Constraints 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels,metric_method=metric_method)


        for con_idx in range(n_constraint):

            C_score = constraint_function_list[con_idx](cluster_data)
            C_train[con_idx].append(C_score)

        F_train = np.append(F_train, score)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score2 = 10
        else:
            score2 = metric(train_data,labels,train_labels = train_labels,metric_method = 'normalized_mutual_info_score',noise=False)
        NMI_train = np.append(NMI_train, score2)
    C_train = np.array(C_train)
    real_C_train = C_train
    return X_train, F_train, C_train, real_C_train,NMI_train, F_train

# BO
def BO_(train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,initial_index=0):
    hyp_setting = [bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],1,1]
    hyp_setting_F = hyp_setting 
    hyp_setting_C = hyp_setting
    hyp_key = hyp_dict.keys()
    n_hyp = len(bounds)

    # Initialize
    X_init = []
    
    for lower, upper in bounds:
        X_init.append((np.random.RandomState(seed=initial_index*10+5).uniform((lower),(upper),n_init)))
    X_init = np.array(X_init).T
    X_train = X_init
    F_train = []
    C_train = []
    n_constraint = len(constraint_function_list)
    NMI_train =[]
    for _ in range(n_constraint):
        C_train.append([])
    for hyp_set in X_train :
        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = hyp_set[idx_]
        cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)
    
        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels,metric_method = metric_method)

        F_train.append(score)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score2 = 10
        else:
            score2 = metric(train_data,labels,train_labels = train_labels,metric_method = 'normalized_mutual_info_score',noise=False)
        NMI_train.append(score2)
        # Constraints 
        for con_idx in range(n_constraint):

            C_score = constraint_function_list[con_idx](cluster_data)
            C_train[con_idx].append(C_score)


    F_train = np.array(F_train)
    F_train = np.reshape(F_train,(-1,1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ############################################################

    X = []
    
    for lower, upper in bounds:
        X.append((np.random.uniform((lower),(upper),1)))
    X = np.array(X).T

    ###################################################################
    for iter_temp in range(n_iter):
        X_test = []
    
        for lower, upper in bounds:
            X_test.append((np.random.uniform((lower),(upper),n_test)))
        X_test = np.array(X_test).T

        # Gaussian Process with u_list

        mu, sigma, Sigma, hyp_setting_F = predict_with_optimized_hyps(X_train, np.reshape(F_train,(-1)), X_test,n_hyp,hyp_setting_F)

        L1_list = []

        for index in range(len(mu)):
            mean = mu[index]
            std = sigma[index]
            u_min = min(F_train)
            L1 = (((u_min-mean)/std)*norm.cdf((u_min-mean)/std)+norm.pdf((u_min-mean)/std))
            L1_list.append(L1)

        next_x  = X_test[np.argmax(L1_list)]
        next_x = np.reshape(next_x,(1,-1))
        X_train = np.concatenate([X_train,next_x])
        
        X = next_x 
        
        # Evaluate new value
        C_train1 = []
        C_train2 = []
        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = next_x[0][idx_]

        cluster = clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        
        # Constraints 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels,metric_method = metric_method )

        for con_idx in range(n_constraint):

            C_score = constraint_function_list[con_idx](cluster_data)
            C_train[con_idx].append(C_score)

        
        F_train = np.append(F_train, score)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score2 = 10
        else:
            score2 = metric(train_data,labels,train_labels = train_labels,metric_method = 'normalized_mutual_info_score',noise=False)
        NMI_train = np.append(NMI_train,score2)
        
    C_train = np.array(C_train)
    real_C_train = C_train
    return X_train, F_train, C_train, real_C_train,NMI_train, F_train


#%%
def figure_print(X_train_list = None, F_train_list = None,C_train_list = None,real_C_train_list = None,NMI_train_list = None,train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,iter_n=10,initial_index=0,print_fig = False):
    
    bounds = bounds
    # 최종적으로 가장 좋은 NMI값과 Objective function값을 가지는 hyperparaemter 값과 objective function value
    X_val_list = []
    F_val_list = []
    res = []
    if print_fig:
        fig = plt.figure()
    regret = 0
    # 가장 낮은 Objective function값과 그 때의 NMI값을 저장한다. 
    for idx in range(iter_n):
        X_train = X_train_list[idx]
        F_train = F_train_list[idx]
        NMI_train = NMI_train_list[idx]
        best_list = []
        nmi_list = []
        for idx in range(len(F_train)):
            best_list.append(F_train[np.argmin(F_train[:(idx+1)])])
            nmi_list.append(NMI_train[np.argmin(F_train[:(idx+1)])])
        if print_fig:
            plt.plot(range(n_init,len(nmi_list)),nmi_list[n_init:]) 
        X_val = X_train[np.argmin(F_train)]
        F_val = min(F_train)
        X_val_list.append(X_val)
        F_val_list.append(F_val)
        regret += sum(nmi_list)
    

    if print_fig:
        
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        my_file = "/images/"+data_name+"_"+str(n_max)+"_"+str(n_min)+"_"+hyperparamter_optimization+ '_1.svg'
        plt.savefig(my_path+my_file)
        



    n_constraints = len(constraint_function_list)
    dbs_list = []
    nmi_list = []
    noise_list = []
    cluster_list =[]
    hyp_key = hyp_dict.keys()

    #
    if ADMMBO_dict['metric_method'] == "normalized_mutual_info_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            dbs_list.append(metric(train_data,labels))
       #print(data_name + constraint+hyperparamter_optimization)
       #print("davies_bouldin_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(dbs_list),np.var(dbs_list),np.min(dbs_list)))
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write(data_name + constraint+"\n")
            f.write("davies_bouldin_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(dbs_list),np.var(dbs_list),np.min(dbs_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("\n")
            f.close()
        nmi_list = F_val_list

        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)

    elif ADMMBO_dict['metric_method'] == "davies_bouldin_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            
            constarint_list = []
            for con_idx in range(n_constraints):
                constarint_list.append(constraint_function_list[con_idx](cluster_data)<=0)
            result1 = reduce((lambda x, y: x and y),            constarint_list)
            if result1:
                nmi_list.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
        if len(nmi_list)==0:
            nmi_list.append(100)
       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print("davies_bouldin_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write(data_name + constraint+ "_"+ hyperparamter_optimization)
            f.write("\n")
            f.write("davies_bouldin_score : ")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()

        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)

    elif ADMMBO_dict['metric_method'] == "silhouette_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            nmi_list.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)


       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print("silhouette_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write(data_name + constraint+ "_"+ hyperparamter_optimization)
            f.write("\n")
            f.write("silhouette_score : ")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()
        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)

    else : 
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            nmi_list.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print(str(ADMMBO_dict['metric_method'] )+" :")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")

        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write(str(ADMMBO_dict['metric_method'] )+" :")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()
        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)
    X_val = X_val_list[np.argmin(F_val_list)]
    cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

    cluster_data = cluster.fit(train_data)
    labels = cluster_data.labels_
    if print_fig:
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        f = open(my_path+"/"+"log_1003.txt", 'a')
        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = X_val[idx_]
            f.write(key+" : "+ str(hyp_dict[key] ) +"\n")
        f.write("nber of labels : "+ str(len(np.unique(labels))) +"\n")
        f.write("regret:{:.3f}".format(regret))
        f.write("\n")
        f.close()


   #print("labels : ")
   #print(np.unique(labels,return_counts=True))
    
    res.append(regret)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    res.append(n_clusters)
    bounds = bounds
    X_val_list = []
    F_val_list = []
    if print_fig:
        plt.close(fig)

    if print_fig:
        fig = plt.figure()
    plus_constraints  =0
    for _ in range(n_constraints):
        plus_constraints += (np.array(C_train_list)[:,_]>0).astype('int')*100
    F_train_list = np.array(F_train_list) + plus_constraints
    NMI_train_list =np.array(NMI_train_list) + plus_constraints
   #print(plus_constraints)
    regret = 0
    nmi_list_list =[]
    for idx in range(iter_n):
        X_train = X_train_list[idx]
        NMI_train = NMI_train_list[idx]
        F_train = F_train_list[idx]
        best_list = []
        nmi_list = []
        for idx in range(len(F_train)):
            best_list.append(F_train[np.argmin(F_train[:(idx+1)])])
            nmi_list.append(NMI_train[np.argmin(F_train[:(idx+1)])])
        if print_fig:
            plt.plot(range(n_init,len(nmi_list)),nmi_list[n_init:]) 
        X_val = X_train[np.argmin(F_train)]
        F_val = min(F_train)
        X_val_list.append(X_val)
        F_val_list.append(F_val)
        regret += sum(nmi_list)
        nmi_list_list.append(nmi_list)
    if print_fig:
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        my_file = "/images/"+data_name+"_"+str(n_max)+"_"+str(n_min)+"_"+hyperparamter_optimization+ '_2.svg'
        plt.savefig(my_path+my_file)
    nmi_list_return = nmi_list_list

    dbs_list = []
    nmi_list = []
    nmi_list2 = []
    nmi_list3 = []
    noise_list = []
    cluster_list =[]
    hyp_key = hyp_dict.keys()

    if ADMMBO_dict['metric_method'] == "normalized_mutual_info_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            constarint_list = []
            for con_idx in range(n_constraints):
                constarint_list.append(constraint_function_list[con_idx](cluster_data)<=0)
            # result1 means this hyperparameter is feasible.
            result1 = reduce((lambda x, y: x and y),  constarint_list)
            if result1:
                nmi_list.append(-sklearn.metrics.normalized_mutual_info_score(train_labels,labels))
                nmi_list2.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=True))
                nmi_list3.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
        if len(nmi_list)==0:
            nmi_list.append(100)
        if len(nmi_list2)==0:
            nmi_list2.append(100)
        if len(nmi_list3)==0:
            nmi_list3.append(100)
       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print("davies_bouldin_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write("davies_bouldin_score : ")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()

        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(nmi_list2))
        res.append(np.var(nmi_list2))
        res.append(np.min(nmi_list2))
        res.append(np.mean(nmi_list3))
        res.append(np.var(nmi_list3))
        res.append(np.min(nmi_list3))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)


    elif ADMMBO_dict['metric_method'] == "davies_bouldin_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            constarint_list = []
            for con_idx in range(n_constraints):
                constarint_list.append(constraint_function_list[con_idx](cluster_data)<=0)
            result1 = reduce((lambda x, y: x and y),            constarint_list)
            if result1:
                nmi_list.append(-sklearn.metrics.normalized_mutual_info_score(train_labels,labels))
                nmi_list2.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=True))
                nmi_list3.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
        if len(nmi_list)==0:
            nmi_list.append(100)
        if len(nmi_list2)==0:
            nmi_list2.append(100)
        if len(nmi_list3)==0:
            nmi_list3.append(100)
       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print("davies_bouldin_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write("davies_bouldin_score : ")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()

        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(nmi_list2))
        res.append(np.var(nmi_list2))
        res.append(np.min(nmi_list2))
        res.append(np.mean(nmi_list3))
        res.append(np.var(nmi_list3))
        res.append(np.min(nmi_list3))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)
        
    elif ADMMBO_dict['metric_method'] == "silhouette_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            nmi_list.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)


       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print("silhouette_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write("\n")
            f.write("silhouette_score : ")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()
        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)
    else : 
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            nmi_list.append(metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
       #print(data_name +"_"+ constraint + "_"+ hyperparamter_optimization)
       #print(str(ADMMBO_dict['metric_method'] )+" :")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list)))
       #print("")
       #print("normalized_mutual_info_score : ")
       #print("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list)))
       #print("")
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write(str(ADMMBO_dict['metric_method'] )+" :")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(F_val_list),np.var(F_val_list),np.min(F_val_list))+"\n")
            f.write("normalized_mutual_info_score : "+"\n")
            f.write("mean:{:.3f},variance:{:.3f},min:{:.3f}".format(np.mean(nmi_list),np.var(nmi_list),np.min(nmi_list))+"\n")
            f.close()
        res.append(np.mean(F_val_list))
        res.append(np.var(F_val_list))
        res.append(np.min(F_val_list))
        res.append(np.mean(nmi_list))
        res.append(np.var(nmi_list))
        res.append(np.min(nmi_list))
        res.append(np.mean(noise_list))
        res.append(np.var(noise_list))
        res.append(np.min(noise_list))
        res.append(cluster_list)

    X_val = X_val_list[np.argmin(F_val_list)]
    cluster = clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

    cluster_data = cluster.fit(train_data)
    labels = cluster_data.labels_
    if print_fig:
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        f = open(my_path+"/"+"log_1003.txt", 'a')
        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = X_val[idx_]
            f.write(key+" : "+ str(hyp_dict[key] ) +"\n")

   #print("labels : ")
   #print(np.unique(labels,return_counts=True))
        f.write("nber of labels : "+ str(len(np.unique(labels))) +"\n")
        f.write("regret:{:.3f}".format(regret))
        f.write("\n")
        f.close()
    
    res.append(regret)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    res.append(n_clusters)
    if print_fig:
        plt.close(fig)
        fig = plt.figure()
        plt.scatter(show_data[:,0],show_data[:,1],c=labels)
        black_idx = (labels==-1)
        plt.scatter(show_data[black_idx,0],show_data[black_idx,1],c='black')
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        my_file = "/images/"+data_name+"_"+str(n_max)+"_"+str(n_min)+"_"+hyperparamter_optimization+ '_3.svg'
        plt.savefig(my_path+my_file)
        plt.close(fig)
 
    return res, nmi_list_return

#%%
def iterate_experiment(ADMMBO_dict= None,exp_fun = None,iter_n=10):
    start = time.time()
    X_train_list = []
    F_train_list = []
    C_train_list = []
    real_C_train_list = []
    NMI_train_list = []
    Y_train_list = []
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    f = open(my_path+"/"+"log_1003.txt", 'a')
    f.write("\n")
    f.write(ADMMBO_dict['data_name']+ "\n")
    f.write("n_max : "+  str(ADMMBO_dict['n_max'])+ "\n")
    f.write( "n_min : "+   str(ADMMBO_dict['n_min']) +"\n")
    f.write( "ele_max : "+  str(ADMMBO_dict['ele_max']) + "\n" )
    f.close()
    
    for _ in range(iter_n):
        ADMMBO_dict['seed'] = _**2+1
        ADMMBO_dict['initial_index'] = _
        X_train, F_train, C_train,real_C_train,NMI_train,Y_train = exp_fun(**ADMMBO_dict)
        X_train_list.append(X_train)
        F_train_list.append(F_train)
        C_train_list.append(C_train)
        Y_train_list.append(Y_train)
        real_C_train_list.append(real_C_train)
        NMI_train_list.append(NMI_train)

    end = time.time()
   #print(end-start)
    

    return X_train_list, F_train_list, C_train_list, real_C_train_list,NMI_train_list,Y_train_list

# %%

train_data0, train_labels = import_data(data='mnist',size=3000)
show_data = embedding_data(train_data = train_data0,n_components = 2 )
#%%
color_list = ['lightcoral','pink','r','y','g','c','b','m','green','navy']

# %%

for i in range(10):
    idx = (train_labels==i)
    plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01,color=color_list[i])
    plt.scatter(0,0,label=i)
plt.legend()
#%%
for i in range(10):
    idx = (train_labels==i)
    plt.scatter(show_data[idx,0],show_data[idx,1],label=i,color=color_list[i],alpha=0.5)

plt.legend()
plt.savefig("mnist_preview.svg")
#%%
idx_list = [2,2,2,9,2,9,2,2,8,1]
for i in range(10):
    idx = (train_labels==i)
    plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01,color=color_list[i])
    plt.scatter(show_data[idx,0][idx_list[i]],show_data[idx,1][idx_list[i]],label=i,color=color_list[i],s=40)
plt.legend()
plt.savefig("mnist_pointview.svg")
#%%
np.where(train_labels==9)[0][3]
#%%
print(np.where(train_labels==4)[0][2])
print(np.where(train_labels==9)[0][1])
print(np.where(train_labels==7)[0][2])
print(np.where(train_labels==8)[0][8])
print(np.where(train_labels==5)[0][9])
print(np.where(train_labels==3)[0][9])
#%%
idx=4
loca=80
plt.scatter(show_data[loca,0],show_data[loca,1],label=idx,color=color_list[idx],s=40)
idx=9
loca=35
plt.scatter(show_data[loca,0],show_data[loca,1],label=idx,color=color_list[idx],s=40)
idx=7
loca=13
plt.scatter(show_data[loca,0],show_data[loca,1],label=idx,color=color_list[idx],s=40)
idx=8
loca=127
plt.scatter(show_data[loca,0],show_data[loca,1],label=idx,color=color_list[idx],s=40)
idx=5
loca=141
plt.scatter(show_data[loca,0],show_data[loca,1],label=idx,color=color_list[idx],s=40)
idx=3
loca=59
plt.scatter(show_data[loca,0],show_data[loca,1],label=idx,color=color_list[idx],s=40)
plt.legend()
plt.xlim(0,18)
plt.ylim(0,10)
for i in range(10):
    idx = (train_labels==i)
    plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01,color=color_list[i])
plt.savefig("mnist_pointview.svg")
#%%
start_date = '2022-04-30'
my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
os.makedirs(my_path, exist_ok=True) 
os.makedirs(my_path+"/images", exist_ok=True) 

f = open(my_path+"/"+"log_1003.txt", 'w')
f.write("2023-09-25-16:00"+ "\n")
output_csv = []
DBCV_list = []


n_labels_list = [10,5,5,3,7,3]
for data_idx, data_name in enumerate(['toy1','toy2','toy3','toy4','toy5','toy6']):
    X_train_list = []
    F_train_list = []
    C_train_list = []
    # Data Load and embedding

    train_data0, train_labels = import_data(data=data_name,size=5000)
    show_data = embedding_data(train_data = train_data0,n_components = 2 )
    if train_data0.shape[1]>20 : 
        train_data = embedding_data(train_data = train_data0,n_components =  int(np.sqrt(train_data0.shape[1]))+1 )
    else : 
        train_data = train_data0
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    n_labels = len(np.unique(train_labels)) 
    label_max = max(np.unique(train_labels,return_counts=True)[1])
    #fig = plt.figure()
    # plt.scatter(show_data[:,0],show_data[:,1],c=train_labels)
    # plt.title("true_label_"+data_name)
    # my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    # my_file = "/images/true_label_"+data_name+'.svg'
    # plt.savefig(my_path+my_file)
    # plt.close(fig)
    pd.DataFrame(show_data).to_csv(my_path +"/" +data_name+"_show_data.csv") 
    pd.DataFrame(train_data).to_csv(my_path +"/"+ data_name+"_train_data.csv") 


    bounds = np.array( [[0.001,0.5],[2,2*np.log(len(train_data))]])
    integer_var = [1]
    hyp_dict = {
        "cluster_selection_epsilon" : 0.5,
        "min_cluster_size" : 5
    }
    hyp_dict = {
        "eps" : 0.5,
        "min_samples" : 5
    }

    fig = plt.figure()
    plt.scatter(train_data[:,0],train_data[:,1],c='gray')
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/true_label_"+data_name+str(data_idx)+'raw+'+'.svg'
    plt.savefig(my_path+my_file)
    plt.close(fig)
   
    n_labels = n_labels_list[data_idx]
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(n_labels - n_clusters , n_clusters -n_labels)

        return C_score
    
    constraint_function_list = [constraint_function1]
    ADMMBO_dict = { 
        "data_name" : data_name, 
        "train_data" : train_data, 
        "show_data" : show_data, 
        "train_labels" : train_labels, 
        "rho" : 10, 
        "M" : 10, 

         "n_max" : n_labels, 
        "n_min" : n_labels, 
        "ele_max" : label_max, 
        "n_init" : 10, 
        "n_iter" : 20, 
        "n_test" : 500, 
        "str_cov" : 'se', 
        "str_initial_method_bo" : 'uniform', 
        "seed" : 0, 
        "clustering_method" : 'dbscan', 
        "metric_method" : 'davies_bouldin_score', 
        "hyp_dict" : hyp_dict, 
        "bounds" : bounds, 
        "integer_var" : [1], 
        "hyperparamter_optimization" : "ADMMBO", 
        "constraint":'Hard',
        "acquisition_function":"EI",
        "alpha" : 2,
        "beta" : 5,
        "constraint_function_list" :constraint_function_list
          ,
        'initial_index':0
    }


    # Grid
    ADMMBO_dict['hyperparamter_optimization'] = "ADMMBO"
    X_train, F_train, C_train,real_C_train,NMI_train, Y_train = ADMMBO(**ADMMBO_dict)
    X_train_list.append(X_train)
    F_train_list.append(F_train)
    C_train_list.append(C_train)
    fig = plt.figure()
    plt.scatter(X_train[:,0],X_train[:,1])
    plt.show()
    plt.close(fig)
    ADMMBO_dict['hyperparamter_optimization'] = "RS"
    ADMMBO_dict['n_iter'] = ADMMBO_dict['n_iter'] *( ADMMBO_dict['alpha']+ ADMMBO_dict['beta']) +  ADMMBO_dict['n_init']
    X_train, F_train, C_train,real_C_train,NMI_train, Y_train = RS_(**ADMMBO_dict)
    X_train_list.append(X_train)
    F_train_list.append(F_train)
    C_train_list.append(C_train)
    fig = plt.figure()
    plt.scatter(X_train[:,0],X_train[:,1])
    plt.show()
    plt.close(fig)
    ADMMBO_dict['hyperparamter_optimization'] = "Grid"
    X_train, F_train, C_train,real_C_train,NMI_train, Y_train = Grid_(**ADMMBO_dict)
    X_train_list.append(X_train)
    F_train_list.append(F_train)
    C_train_list.append(C_train)
    fig = plt.figure()
    plt.scatter(X_train[:,0],X_train[:,1])
    plt.show()
    plt.close(fig)
    ADMMBO_dict['hyperparamter_optimization'] = "BO"
    ADMMBO_dict['n_iter'] = ADMMBO_dict['n_iter'] -  ADMMBO_dict['n_init']
    X_train, F_train, C_train,real_C_train,NMI_train, Y_train = BO_(**ADMMBO_dict)
    X_train_list.append(X_train)
    F_train_list.append(F_train)
    C_train_list.append(C_train)
    fig = plt.figure()
    plt.scatter(X_train[:,0],X_train[:,1])
    plt.show()
    plt.close(fig)

    for idx in range(len(X_train_list)):

        X_train = X_train_list[idx]
        F_train = F_train_list[idx]
        C_train = C_train_list[idx]

        #plt.scatter(X_train[:,0],X_train[:,1],alpha=C_train<=0)
        #plt.figure()
        #plt.scatter(X_train[:,0],X_train[:,1])
        hyp_key = hyp_dict.keys()
        #X_val =[0.02737924,  9.2487109]
        if len(F_train[(C_train[0].reshape(-1)<=0)]) ==0:
            if len(X_train[np.where(F_train == max(F_train))][0]) !=1 :
                X_val = X_train[np.where(F_train == max(F_train)  )][0]
            else : 
                X_val = X_train[np.where(F_train == max(F_train)  )]
            f.write('there is no feasible solution')
        else : 
            if len(X_train[np.where(F_train == min(F_train[(C_train[0].reshape(-1)<=0)  ]))][0]) !=1:
                X_val = X_train[np.where(F_train == min(F_train[(C_train[0].reshape(-1)<=0)  ]))][0]
            else : 
                X_val = X_train[np.where(F_train == min(F_train[(C_train[0].reshape(-1)<=0)  ]))]

        #X_val = X_train[282]

        for idx_, key in enumerate(hyp_key):
            hyp_dict[key] = X_val[idx_]

        cluster = clustering(clustering_method=ADMMBO_dict['clustering_method'],hyp_dict = hyp_dict)

        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        
        f.write(str(X_val))
        f.write('\n')
        f.write(str(np.unique(labels,return_counts=True)))
        f.write('\n')

        f.write("Davies_bouldin_score : ")
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1 or len(set(labels))==len(labels):
            score = 10
        else:
            score = metric(train_data,labels,train_labels = train_labels,metric_method = 'davies_bouldin_score',noise=False)

        f.write(str(score))
        f.write('\n')

        fig = plt.figure()
        plt.scatter(train_data[:,0],train_data[:,1],c=labels,alpha=0.9)
        black_idx = (labels==-1)
        plt.scatter(train_data[black_idx,0],train_data[black_idx,1],c='gray')
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        my_file = "/images/true_label_"+data_name+str(data_idx)+str(idx)+'.svg'
        plt.savefig(my_path+my_file)
        plt.close(fig)


f.close()
#%%

start_date ="2022-04-27-Matern"
class AdvancedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,n_hyp):     
        super(AdvancedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,active_dims = torch.tensor([0])  ) * MaternKernel2(nu=2.5,active_dims = torch.tensor([1]) ) )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
os.makedirs(my_path, exist_ok=True) 
os.makedirs(my_path+"/images", exist_ok=True) 
f = open(my_path+"/"+"log_1003.txt", 'w')
f.write("2022-01-26-16:00"+ "\n")

def constraint_function79(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[80]==labels[35])+1

    return C_score

def constraint_function94(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[35]==labels[13])+1

    return C_score

def constraint_function38(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[127]==labels[141])+1

    return C_score

def constraint_function35(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[127]==labels[59])+1

    return C_score

def constraint_function58(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[59]==labels[141])+1

    return C_score    

constraint_function_list = [constraint_function1,constraint_function79, constraint_function94,constraint_function38,constraint_function35,constraint_function58]

f.close()
output_csv = []
for data_name in ['mnist']:
#for data_name in ["mnist"]:
#for data_name in ['mnist','table_4','table_5']:
    

    # Data Load and embedding
    train_data0, train_labels = import_data(data=data_name,size=3000)
    show_data = embedding_data(train_data = train_data0,n_components = 2 )
    if data_name != "reuters":
        if train_data0.shape[1]>10 : 
            train_data = embedding_data(train_data = train_data0,n_components =  round(np.sqrt(train_data0.shape[1]+1)) )

        else : 
            train_data =  embedding_data(train_data = train_data0,n_components =  round(np.sqrt(train_data0.shape[1]+1)) )

    else : 
        train_data = train_data0
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    n_labels = len(np.unique(train_labels)) 
    label_max = max(np.unique(train_labels,return_counts=True)[1])
    fig = plt.figure()
    plt.scatter(show_data[:,0],show_data[:,1],c=train_labels)
    plt.title("true_label_"+data_name)
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/true_label_"+data_name+'.svg'
    plt.savefig(my_path+my_file)
    plt.close(fig)
    pd.DataFrame(show_data).to_csv(my_path +"/" +data_name+"_show_data.csv") 
    pd.DataFrame(train_data).to_csv(my_path +"/"+ data_name+"_train_data.csv") 

    hyp_dict = {
        "cluster_selection_epsilon" : 0.5,
        "min_cluster_size" : 5
    }

    hyp_dict = {
        "eps" : 0.5,    
        "min_samples" : 5
    }
    #bounds = np.array( [[0.00001,0.5],[2,max(10,train_data.shape[1]*int(np.log10(train_data.shape[0]))) ]])
    bounds = np.array( [[0.0001,1],[2,train_data.shape[1]*np.round(np.log10(train_data.shape[0])) ]])
    integer_var = [1]

    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = non-positive value
        C_score = - min(n_labels - n_clusters , n_clusters -n_labels)

        return C_score
    
    ADMMBO_dict = { 
        "data_name" : data_name, 
        "train_data" : train_data, 
        "show_data" : show_data, 
        "train_labels" : train_labels, 
        "rho" : 10, 
        "M" : 100, 

        
        "n_max" : n_labels, 
        "n_min" : n_labels, 
        "ele_max" : label_max, 
        "n_init" : 10, 
        "n_iter" : 5, 
        "n_test" : 500, 
        "str_cov" : 'se', 
        "str_initial_method_bo" : 'uniform', 
        "seed" : 0, 
        "clustering_method" : 'dbscan', 
        "metric_method" : 'davies_bouldin_score', 
        "hyp_dict" : hyp_dict, 
        "bounds" : bounds, 
        "integer_var" : [1], 
        "hyperparamter_optimization" : "ADMMBO", 
        "constraint":'Soft',
        "acquisition_function":"EI",
        "alpha" : 2,
        "beta" : 2,
        "constraint_function_list" :constraint_function_list,
        'initial_index':0
    }

    #  # ADMMBO
    label_bounds = [[n_labels+int(n_labels*0.1),max(n_labels-int(n_labels*0.1),1)]]

    label_condition = ["normal"]



    ele_bounds = [int(label_max*1.5),label_max*2,max(label_max/2,int(len(train_data)/n_labels))]
    ele_condition = ["normal","loosen","tight"]
    ele_condition = ["normal"]
    nmi_list_list = []
    nmi_list_list2 = []
    nmi_list_list3 = []
    for constraint_i in ['Hard','Soft']:
        for ele_idx in range(1):
            ADMMBO_dict["ele_max"] = ele_bounds[ele_idx]
            for label_idx in range(1):

                ADMMBO_dict["n_max"] = label_bounds[label_idx][0]
                ADMMBO_dict["n_min"] = label_bounds[label_idx][1]

                def constraint_function1(cluster_data):
                    labels = cluster_data.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    # Feasible solution = negative value
                    C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

                    return C_score

                constraint_function_list = [constraint_function1,constraint_function79, constraint_function94,constraint_function38,constraint_function35,constraint_function58]

                ADMMBO_dict['constraint'] = constraint_i
                ADMMBO_dict['hyperparamter_optimization'] = "ADMMBO_" + label_condition[label_idx] + ele_condition[ele_idx] + ADMMBO_dict['constraint']
                X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,ADMMBO)

                output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
                res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
                output_csv.append(["ADMMBO"] + output + res)
                if label_idx ==0:
                    nmi_list_list.append(nmi_list)
                elif label_idx ==1:
                    nmi_list_list2.append(nmi_list)
                else :
                    nmi_list_list3.append(nmi_list)
                # ADMMBO_dict['constraint'] = "Soft"
                # X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list,Y_train_list  = iterate_experiment(ADMMBO_dict,ADMMBO)
                
                # output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
                # res = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = real_C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)

                # output_csv.append(["ADMMBO"] + output + res)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0
    hyp_dict = {
        "eps" : 0.5,
        "min_samples" : 5
    }
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score
    constraint_function_list = [constraint_function1,constraint_function79, constraint_function94,constraint_function38,constraint_function35,constraint_function58]
    ADMMBO_dict["constraint_function_list"] =constraint_function_list


    ADMMBO_dict['n_iter'] = ADMMBO_dict['n_iter'] *(ADMMBO_dict['alpha']+ADMMBO_dict['beta']*len(constraint_function_list))  +  ADMMBO_dict['n_init']

    # RS
    ADMMBO_dict['hyperparamter_optimization'] = "RS"
    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,RS_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["RS"] + output + res)    
    nmi_list_list.append(nmi_list)



    # Grid
    ADMMBO_dict['hyperparamter_optimization'] = "Grid"
    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0

    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,Grid_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[0], ele_condition[0]]
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["Grid"] + output + res)      

    nmi_list_list.append(nmi_list)


    # BO
    ADMMBO_dict['hyperparamter_optimization'] = "BO"
    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0

    ADMMBO_dict['n_iter'] = ADMMBO_dict['n_iter'] -  ADMMBO_dict['n_init']
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,BO_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["BO"] + output + res)
    nmi_list_list.append(nmi_list)

    nmi_const_list_list = np.array(nmi_list_list)


    nmi_val_list_list = nmi_list_list[0:2]
    #############################################################################


    nmi_val_list_list = np.array(nmi_val_list_list)
    # Number of violation for each algorithms
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_constraint.svg'
    plt.figure()
    label_list = ['HC-DBSCAN (Hard)', 'HC-DBSCAN (Soft)',"RS","Grid","BO"]
    color_list = ['red','orange','blue','yellow','green']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_const_list_list):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        plt.plot(nmi_list_mean,label=label_list[idx],color=color_list[idx])
        plt.plot(nmi_list_mean+nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(nmi_list_mean-nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    # NMI value for each algorithms in which clustering satisfies the constraints.
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_nmi.svg'
    plt.figure()
    label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
    color_list = ['red','orange','blue','yellow']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_val_list_list):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        origin_len = len(nmi_list_mean)
        origin_index = (nmi_list_mean <5)
        x_axis = np.array([ i for i in range(origin_len)])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)


    ####################################################################################################################

    ######################################################################################################################

    
my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
col_names = ["HPO",	"data_name",	"constraint",	"n_clsuters",	"n_elements",	"mean_1",	"var_1",	"min_1",	"mean,_2",	"var_2",	"min_2",	"mean_noise"	,"var_noise",	"min_noise",	"n_cluster1","regret","n_cluster_draw_1",	"mean_4"	,"var_4"	,"min_4",	"nmi_mean"	,"nmi_var",	"nmi_min", "nmi_mean_penalty_noise"	,"nmi_var_penalty_noise",	"nmi_min_penalty_noise", "nmi_mean_no_noise"	,"nmi_var_no_noise",	"nmi_min_no_noise", 	"mean_noise2"	,"var_noise2",	"min_noise2"	,"n_cluster2"	,"regret2","n_cluster_draw_1"]
output_csv = pd.DataFrame(output_csv)
output_csv.columns = col_names
output_csv.to_csv(my_path + "/" +start_date+".csv")

#%%
nmi_val_list_list = np.array(nmi_val_list_list)
#%%
# Number of violation for each algorithms
my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
my_file = "/images/"+data_name+ 'final_constraint.svg'
plt.figure()
label_list = ['HC-DBSCAN (Hard)', 'HC-DBSCAN (Soft)',"RS","Grid","BO"]
color_list = ['red','orange','blue','yellow','green']
plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

for idx,nmi_list in enumerate(nmi_const_list_list):
    nmi_list_mean = nmi_list.mean(axis=0)
    nmi_list_std = nmi_list.std(axis=0)
    plt.plot(nmi_list_mean,label=label_list[idx],color=color_list[idx])
    plt.plot(nmi_list_mean+nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.plot(nmi_list_mean-nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
plt.legend(loc='upper right')
plt.savefig(my_path + my_file)

# NMI value for each algorithms in which clustering satisfies the constraints.
my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
my_file = "/images/"+data_name+ 'final_nmi.svg'
plt.figure()
label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
color_list = ['red','orange','blue','yellow']
plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

for idx,nmi_list in enumerate(nmi_val_list_list):
    nmi_list_mean = nmi_list.mean(axis=0)
    nmi_list_std = nmi_list.std(axis=0)
    origin_len = len(nmi_list_mean)
    origin_index = (nmi_list_mean <5)
    x_axis = np.array([ i for i in range(origin_len)])
    plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
    plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
plt.legend(loc='upper right')
plt.savefig(my_path + my_file)


####################################################################################################################

######################################################################################################################


my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
col_names = ["HPO",	"data_name",	"constraint",	"n_clsuters",	"n_elements",	"mean_1",	"var_1",	"min_1",	"mean,_2",	"var_2",	"min_2",	"mean_noise"	,"var_noise",	"min_noise",	"n_cluster1","regret","n_cluster_draw_1",	"mean_4"	,"var_4"	,"min_4",	"nmi_mean"	,"nmi_var",	"nmi_min", "nmi_mean_penalty_noise"	,"nmi_var_penalty_noise",	"nmi_min_penalty_noise", "nmi_mean_no_noise"	,"nmi_var_no_noise",	"nmi_min_no_noise", 	"mean_noise2"	,"var_noise2",	"min_noise2"	,"n_cluster2"	,"regret2","n_cluster_draw_1"]
output_csv = pd.DataFrame(output_csv)
output_csv.columns = col_names
output_csv.to_csv(my_path + "/" +start_date+".csv")



#%%
ADMMBO_dict['n_iter'] = int((ADMMBO_dict['n_iter'] -  ADMMBO_dict['n_init'])/7)+ADMMBO_dict['n_init']
# %%




#==================================================================================================================================================================================================
#%%



start_date ="2022-04-27-Matern"
class AdvancedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,n_hyp):     
        super(AdvancedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,active_dims = torch.tensor([0])  ) * MaternKernel2(nu=2.5,active_dims = torch.tensor([1]) ) )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
os.makedirs(my_path, exist_ok=True) 
os.makedirs(my_path+"/images", exist_ok=True) 
f = open(my_path+"/"+"log_1003.txt", 'w')
f.write("2022-01-26-16:00"+ "\n")

def constraint_function79(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[80]==labels[35])+1

    return C_score

def constraint_function94(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[35]==labels[13])+1

    return C_score

def constraint_function38(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[127]==labels[141])+1

    return C_score

def constraint_function35(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[127]==labels[59])+1

    return C_score

def constraint_function58(cluster_data):
    labels = cluster_data.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Feasible solution = negative value
    C_score = - np.double(labels[59]==labels[141])+1

    return C_score    

constraint_function_list = [constraint_function1,constraint_function79, constraint_function94,constraint_function38,constraint_function35,constraint_function58]

f.close()
output_csv = []
for data_name in ['mnist']:
#for data_name in ["mnist"]:
#for data_name in ['mnist','table_4','table_5']:
    

    # Data Load and embedding
    train_data0, train_labels = import_data(data=data_name,size=3000)
    show_data = embedding_data(train_data = train_data0,n_components = 2 )
    if data_name != "reuters":
        if train_data0.shape[1]>10 : 
            train_data = embedding_data(train_data = train_data0,n_components =  round(np.sqrt(train_data0.shape[1]+1)) )

        else : 
            train_data =  embedding_data(train_data = train_data0,n_components =  round(np.sqrt(train_data0.shape[1]+1)) )

    else : 
        train_data = train_data0
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    n_labels = len(np.unique(train_labels)) 
    label_max = max(np.unique(train_labels,return_counts=True)[1])
    fig = plt.figure()
    plt.scatter(show_data[:,0],show_data[:,1],c=train_labels)
    plt.title("true_label_"+data_name)
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/true_label_"+data_name+'.svg'
    plt.savefig(my_path+my_file)
    plt.close(fig)
    pd.DataFrame(show_data).to_csv(my_path +"/" +data_name+"_show_data.csv") 
    pd.DataFrame(train_data).to_csv(my_path +"/"+ data_name+"_train_data.csv") 

    hyp_dict = {
        "cluster_selection_epsilon" : 0.5,
        "min_cluster_size" : 5
    }

    hyp_dict = {
        "eps" : 0.5,    
        "min_samples" : 5
    }
    #bounds = np.array( [[0.00001,0.5],[2,max(10,train_data.shape[1]*int(np.log10(train_data.shape[0]))) ]])
    bounds = np.array( [[0.0001,1],[2,train_data.shape[1]*np.round(np.log10(train_data.shape[0])) ]])
    integer_var = [1]

    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = non-positive value
        C_score = - min(n_labels - n_clusters , n_clusters -n_labels)

        return C_score
    
    ADMMBO_dict = { 
        "data_name" : data_name, 
        "train_data" : train_data, 
        "show_data" : show_data, 
        "train_labels" : train_labels, 
        "rho" : 10, 
        "M" : 100, 

        
        "n_max" : n_labels, 
        "n_min" : n_labels, 
        "ele_max" : label_max, 
        "n_init" : 20, 
        "n_iter" : 10, 
        "n_test" : 500, 
        "str_cov" : 'se', 
        "str_initial_method_bo" : 'uniform', 
        "seed" : 0, 
        "clustering_method" : 'dbscan', 
        "metric_method" : 'davies_bouldin_score', 
        "hyp_dict" : hyp_dict, 
        "bounds" : bounds, 
        "integer_var" : [1], 
        "hyperparamter_optimization" : "ADMMBO", 
        "constraint":'Soft',
        "acquisition_function":"EI",
        "alpha" : 2,
        "beta" : 5,
        "constraint_function_list" :constraint_function_list,
        'initial_index':0
    }

    #  # ADMMBO
    label_bounds = [[n_labels+int(n_labels*0.1),max(n_labels-int(n_labels*0.1),1)],[int(n_labels*0.5)+int(n_labels*0.1),max(int(n_labels*0.5)-int(n_labels*0.1),1)],[int(n_labels*1.5)+int(n_labels*0.1),max(int(n_labels*1.5)-int(n_labels*0.1),1)]]

    label_condition = ["normal","less","high"]



    ele_bounds = [int(label_max*1.5),label_max*2,max(label_max/2,int(len(train_data)/n_labels))]
    ele_condition = ["normal","loosen","tight"]
    ele_condition = ["normal"]
    nmi_list_list = []
    nmi_list_list2 = []
    nmi_list_list3 = []
    for constraint_i in ['Hard','Soft']:
        for ele_idx in range(1):
            ADMMBO_dict["ele_max"] = ele_bounds[ele_idx]
            for label_idx in range(3):

                ADMMBO_dict["n_max"] = label_bounds[label_idx][0]
                ADMMBO_dict["n_min"] = label_bounds[label_idx][1]

                def constraint_function1(cluster_data):
                    labels = cluster_data.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    # Feasible solution = negative value
                    C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

                    return C_score

                constraint_function_list = [constraint_function1,constraint_function79, constraint_function94,constraint_function38,constraint_function35,constraint_function58]

                ADMMBO_dict['constraint'] = constraint_i
                ADMMBO_dict['hyperparamter_optimization'] = "ADMMBO_" + label_condition[label_idx] + ele_condition[ele_idx] + ADMMBO_dict['constraint']
                X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,ADMMBO)

                output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
                res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
                output_csv.append(["ADMMBO"] + output + res)
                if label_idx ==0:
                    nmi_list_list.append(nmi_list)
                elif label_idx ==1:
                    nmi_list_list2.append(nmi_list)
                else :
                    nmi_list_list3.append(nmi_list)
                # ADMMBO_dict['constraint'] = "Soft"
                # X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list,Y_train_list  = iterate_experiment(ADMMBO_dict,ADMMBO)
                
                # output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
                # res = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = real_C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)

                # output_csv.append(["ADMMBO"] + output + res)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0
    hyp_dict = {
        "eps" : 0.5,
        "min_samples" : 5
    }
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    ADMMBO_dict['n_iter'] = ADMMBO_dict['n_iter'] *7 +  ADMMBO_dict['n_init']

    # RS
    ADMMBO_dict['hyperparamter_optimization'] = "RS"
    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,RS_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["RS"] + output + res)    
    nmi_list_list.append(nmi_list)



    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[1][0]
    ADMMBO_dict["n_min"] = label_bounds[1][1]
    label_idx = 1
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score
    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,RS_)
    
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["RS"] + output + res)
    nmi_list_list2.append(nmi_list)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[2][0]
    ADMMBO_dict["n_min"] = label_bounds[2][1]
    label_idx = 2
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,RS_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["RS"] + output + res)
    nmi_list_list3.append(nmi_list)




    # Grid
    ADMMBO_dict['hyperparamter_optimization'] = "Grid"
    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]

    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,Grid_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[0], ele_condition[0]]
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["Grid"] + output + res)      

    nmi_list_list.append(nmi_list)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[1][0]
    ADMMBO_dict["n_min"] = label_bounds[1][1]
    label_idx = 1
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,Grid_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["Grid"] + output + res)
    nmi_list_list2.append(nmi_list)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[2][0]
    ADMMBO_dict["n_min"] = label_bounds[2][1]
    label_idx = 2
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,Grid_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["Grid"] + output + res)
    nmi_list_list3.append(nmi_list)


    # BO
    ADMMBO_dict['hyperparamter_optimization'] = "BO"
    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[0][0]
    ADMMBO_dict["n_min"] = label_bounds[0][1]
    label_idx = 0
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]

    ADMMBO_dict['n_iter'] = ADMMBO_dict['n_iter'] -  ADMMBO_dict['n_init']
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,BO_)
    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["BO"] + output + res)
    nmi_list_list.append(nmi_list)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[1][0]
    ADMMBO_dict["n_min"] = label_bounds[1][1]
    label_idx = 1
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,BO_)

    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["BO"] + output + res)
    nmi_list_list2.append(nmi_list)

    ADMMBO_dict["ele_max"] = ele_bounds[0]
    ADMMBO_dict["n_max"] = label_bounds[2][0]
    ADMMBO_dict["n_min"] = label_bounds[2][1]
    label_idx = 2
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score

    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,BO_)

    output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
    
    res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
    output_csv.append(["BO"] + output + res)
    nmi_list_list3.append(nmi_list)

    nmi_const_list_list = np.array(nmi_list_list)
    nmi_const_list_list2 = np.array(nmi_list_list2)
    nmi_const_list_list3 = np.array(nmi_list_list3)

    nmi_val_list_list = nmi_list_list[0:2]
    nmi_val_list_list2 = nmi_list_list2[0:2]
    nmi_val_list_list3 = nmi_list_list3[0:2]
#############################################################################
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

        return C_score
        
    ADMMBO_dict["constraint_function_list"] = [constraint_function1]
    ADMMBO_dict['n_iter'] = int((ADMMBO_dict['n_iter'] )/7)

    for constraint_i in ['Hard','Soft']:
        for ele_idx in range(1):
            ADMMBO_dict["ele_max"] = ele_bounds[ele_idx]
            for label_idx in range(3):

                ADMMBO_dict["n_max"] = label_bounds[label_idx][0]
                ADMMBO_dict["n_min"] = label_bounds[label_idx][1]

                def constraint_function1(cluster_data):
                    labels = cluster_data.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    # Feasible solution = negative value
                    C_score = - min(ADMMBO_dict["n_max"]  - n_clusters , n_clusters -ADMMBO_dict["n_min"] )

                    return C_score

                ADMMBO_dict["constraint_function_list"] = [constraint_function1]
                ADMMBO_dict['metric_method'] = "normalized_mutual_info_score"
                ADMMBO_dict['constraint'] = constraint_i
                ADMMBO_dict['hyperparamter_optimization'] = "ADMMBO_NMI_" + label_condition[label_idx] + ele_condition[ele_idx] + ADMMBO_dict['constraint']
                X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list, Y_train_list = iterate_experiment(ADMMBO_dict,ADMMBO)

                output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
                res, nmi_list = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)
                output_csv.append(["ADMMBO_NMI"] + output + res)
                if label_idx ==0:
                    nmi_val_list_list.append(nmi_list)
                elif label_idx ==1:
                    nmi_val_list_list2.append(nmi_list)
                else :
                    nmi_val_list_list3.append(nmi_list)
                # ADMMBO_dict['constraint'] = "Soft"
                # X_train_list, F_train_list, C_train_list,real_C_train_list,NMI_train_list,Y_train_list  = iterate_experiment(ADMMBO_dict,ADMMBO)
                
                # output = [ADMMBO_dict['data_name'],ADMMBO_dict['constraint'],label_condition[label_idx], ele_condition[ele_idx]]
                # res = figure_print(X_train_list = X_train_list, F_train_list = F_train_list,C_train_list = C_train_list,real_C_train_list = real_C_train_list,NMI_train_list = NMI_train_list, **ADMMBO_dict)

                # output_csv.append(["ADMMBO"] + output + res)
    ##############################################################################
    nmi_val_list_list = np.array(nmi_val_list_list)
    nmi_val_list_list2 = np.array(nmi_val_list_list2)
    nmi_val_list_list3 = np.array(nmi_val_list_list3)

    # Number of violation for each algorithms
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_constraint.svg'
    plt.figure()
    label_list = ['HC-DBSCAN (Hard)', 'HC-DBSCAN (Soft)',"RS","Grid","BO"]
    color_list = ['red','orange','blue','yellow','green']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')
    for idx,nmi_list in enumerate(nmi_const_list_list):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        plt.plot(nmi_list_mean,label=label_list[idx],color=color_list[idx])
        plt.plot(nmi_list_mean+nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(nmi_list_mean-nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    # NMI value for each algorithms in which clustering satisfies the constraints.
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_nmi.svg'
    plt.figure()
    label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
    color_list = ['red','orange','blue','yellow']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_val_list_list):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        origin_len = len(nmi_list_mean)
        origin_index = (nmi_list_mean <5)
        x_axis = np.array([ i for i in range(origin_len)])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)


    ####################################################################################################################

    
    # NMI value for each algorithms in which clustering satisfies the constraints.
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_low_nmi.svg'
    plt.figure()
    label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
    color_list = ['red','orange','blue','yellow']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_val_list_list2):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        origin_len = len(nmi_list_mean)
        origin_index = (nmi_list_mean <5)
        x_axis = np.array([ i for i in range(origin_len)])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_low_constraint.svg'
    plt.figure()
    label_list = ['HC-DBSCAN (Hard)', 'HC-DBSCAN (Soft)',"RS","Grid","BO"]
    color_list = ['red','orange','blue','yellow','green']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')
    for idx,nmi_list in enumerate(nmi_const_list_list2):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        plt.plot(nmi_list_mean,label=label_list[idx],color=color_list[idx])
        plt.plot(nmi_list_mean+nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(nmi_list_mean-nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_low_val.svg'
    plt.figure()
    label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
    color_list = ['red','orange','blue','yellow']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_val_list_list2):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        origin_len = len(nmi_list_mean)
        origin_index = (nmi_list_mean <5)
        x_axis = np.array([ i for i in range(origin_len)])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    ######################################################################################################################

    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_high_constraints.svg'
    plt.figure()
    label_list = ['HC-DBSCAN (Hard)', 'HC-DBSCAN (Soft)',"RS","Grid","BO"]
    color_list = ['red','orange','blue','yellow','green']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')
    for idx,nmi_list in enumerate(nmi_const_list_list3):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        plt.plot(nmi_list_mean,label=label_list[idx],color=color_list[idx])
        plt.plot(nmi_list_mean+nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(nmi_list_mean-nmi_list_std,alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_high_val.svg'
    plt.figure()
    label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
    color_list = ['red','orange','blue','yellow']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_val_list_list3):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        origin_len = len(nmi_list_mean)
        origin_index = (nmi_list_mean <5)
        x_axis = np.array([ i for i in range(origin_len)])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)

    
    # NMI value for each algorithms in which clustering satisfies the constraints.
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    my_file = "/images/"+data_name+ 'final_high_nmi.svg'
    plt.figure()
    label_list = ['Hard, Davies_bouldin_score', 'Soft, Davies_bouldin_score',"Hard, NMI","Soft, NMI"]
    color_list = ['red','orange','blue','yellow']
    plt.axvline(ADMMBO_dict['n_init'],linestyle='dashed')

    for idx,nmi_list in enumerate(nmi_val_list_list3):
        nmi_list_mean = nmi_list.mean(axis=0)
        nmi_list_std = nmi_list.std(axis=0)
        origin_len = len(nmi_list_mean)
        origin_index = (nmi_list_mean <5)
        x_axis = np.array([ i for i in range(origin_len)])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index],label=label_list[idx],color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] + nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
        plt.plot(x_axis[origin_index],nmi_list_mean[origin_index] - nmi_list_std[origin_index],alpha=0.3,linestyle='dotted',color=color_list[idx])
    plt.legend(loc='upper right')
    plt.savefig(my_path + my_file)
    
my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
col_names = ["HPO",	"data_name",	"constraint",	"n_clsuters",	"n_elements",	"mean_1",	"var_1",	"min_1",	"mean,_2",	"var_2",	"min_2",	"mean_noise"	,"var_noise",	"min_noise",	"n_cluster1","regret","n_cluster_draw_1",	"mean_4"	,"var_4"	,"min_4",	"nmi_mean"	,"nmi_var",	"nmi_min", "nmi_mean_penalty_noise"	,"nmi_var_penalty_noise",	"nmi_min_penalty_noise", "nmi_mean_no_noise"	,"nmi_var_no_noise",	"nmi_min_no_noise", 	"mean_noise2"	,"var_noise2",	"min_noise2"	,"n_cluster2"	,"regret2","n_cluster_draw_1"]
output_csv = pd.DataFrame(output_csv)
output_csv.columns = col_names
output_csv.to_csv(my_path + "/" +start_date+".csv")