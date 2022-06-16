#
# author: Jongwon Kim (pioneer0517@postech.ac.kr)
# last updated: June 02, 2022
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
#%%
from ADMMBO import kernel
from clustering import evaluation_metric as EM
from ADMMBO import gaussian_process as GP
from ADMMBO import plot




def main():

    device = 'cpu'


    start_date = '2022-06-02-simulated'
    my_path = r"c:/Users/user/Documents/GitHub/HC-DBSCAN/" + start_date

    os.makedirs(my_path, exist_ok=True) 
    os.makedirs(my_path+"/images", exist_ok=True) 

    f = open(my_path+"/"+"log_"+ start_date +".txt", 'w')
    f.write(start_date+ "\n")
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
        label_max = max(np.unique(train_labels,return_counts=True)[1])

        pd.DataFrame(show_data).to_csv(my_path +"/" +data_name+"_show_data.csv") 
        pd.DataFrame(train_data).to_csv(my_path +"/"+ data_name+"_train_data.csv") 

        fig = plt.figure()
        plt.scatter(train_data[:,0],train_data[:,1],c='gray')
        my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
        my_file = "/images/true_label_"+data_name+str(data_idx)+'raw+'+'.svg'
        plt.savefig(my_path+my_file)
        plt.close(fig)

        bounds = np.array( [[0.001,0.5],[2,2*np.log(len(train_data))]])
        integer_var = [1]




        # Define constraint function
        n_labels = n_labels_list[data_idx]
        def constraint_function1(cluster_data):
            labels = cluster_data.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # Feasible solution = negative value
            C_score = - min(n_labels - n_clusters , n_clusters -n_labels)

            return C_score
        
        constraint_function_list = [constraint_function1]

        hyp_dict = {
            "eps" : 0.5,
            "min_samples" : 5
        }
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


if __name__ == "__main__":
	main()