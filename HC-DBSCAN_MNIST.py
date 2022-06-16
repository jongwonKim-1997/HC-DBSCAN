#
# author: Jongwon Kim (pioneer0517@postech.ac.kr)
# last updated: June 02, 2022
#


from HCDBSCAN import core
from HCDBSCAN import preprocessing
from HCDBSCAN.clustering import DBSCAN

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time    
hyp_dict = {
    "eps" : 0.5,    
    "min_samples" : 5
}

def main():
    data_name = 'mnist'
    # Load and preprocess the MNIST dataset 
    train_data0, train_labels = preprocessing.import_data(data=data_name,size=3000)
    show_data = preprocessing.embedding_data(train_data = train_data0,n_components = 2 )
    if data_name != "reuters":
        if train_data0.shape[1]>10 : 
            train_data = preprocessing.embedding_data(train_data = train_data0,n_components =  round(np.sqrt(train_data0.shape[1]+1)) )

        else : 
            train_data =  preprocessing.embedding_data(train_data = train_data0,n_components =  round(np.sqrt(train_data0.shape[1]+1)) )

    else : 
        train_data = train_data0
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
        

    # Define constraint functions
    def constraint_CL(idx1,idx2):
        def constraint_function_CL(cluster_data):
            labels = cluster_data.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # Feasible solution = negative value
            C_score = - np.double(labels[idx1]==labels[idx2])+1
            return C_score
        return constraint_function_CL


    n_labels = len(np.unique(train_labels)) 
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(n_labels - n_clusters , n_clusters -n_labels)

        return C_score

    CL_idx_list =  [[80,35],[35,13],[127,141],[127,59],[59,141]]
    constraint_function_list = [constraint_function1] + [constraint_CL(idx1,idx2) for (idx1,idx2) in CL_idx_list]

    # Define HC-DBSCAN function's input parameters
    label_max = max(np.unique(train_labels,return_counts=True)[1])
    bounds = np.array( [[0.0001,1],[2,train_data.shape[1]*np.round(np.log10(train_data.shape[0])) ]])
    ADMMBO_dict = { 
        "data_name" : data_name, 
        "train_data" : train_data, 
        "show_data" : show_data, 
        "train_labels" : train_labels, 
        "rho" : 10, 
        "M" : 100, 

        
        "n_max" : 0, 
        "n_min" : 0, 
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

    X_train, F_train, C_train, real_C_train,NMI_train,Y_train = core.HC_DBSCAN(**ADMMBO_dict)

    best_hyperparameter = X_train[np.argmin(F_train)]
    
    hyp_key = hyp_dict.keys()
    for idx_, key in enumerate(hyp_key):
        hyp_dict[key] = best_hyperparameter[idx_]

    cluster = DBSCAN.clustering(clustering_method = ADMMBO_dict['clustering_method'], hyp_dict= hyp_dict)

    cluster_data = cluster.fit(train_data)
    labels = cluster_data.labels_

    # Plot the image
    color_list = ['lightcoral','pink','r','y','g','c','b','m','green','navy']
    fig = plt.figure()
    for i in range(10):
        idx = (train_labels==i)
        plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01,color=color_list[i])
    plt.title("MNIST dataset")
    plt.show()
    plt.close(fig)


    fig = plt.figure()
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
    plt.annotate(s='', xy=show_data[80], xytext=show_data[35], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[35], xytext=show_data[15], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[127], xytext=show_data[141], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[127], xytext=show_data[59], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[59], xytext=show_data[141], arrowprops=dict(arrowstyle='<->'))
    plt.legend()
    plt.xlim(0,18)
    plt.ylim(0,10)
    for i in range(10):
        idx = (train_labels==i)
        plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01,color=color_list[i])
    plt.show()
    plt.title("MNIST dataset with five CL constraints")
    plt.close(fig)

    fig = plt.figure()
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
    plt.annotate(s='', xy=show_data[80], xytext=show_data[35], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[35], xytext=show_data[15], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[127], xytext=show_data[141], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[127], xytext=show_data[59], arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='', xy=show_data[59], xytext=show_data[141], arrowprops=dict(arrowstyle='<->'))
    plt.legend()
    plt.xlim(0,18)
    plt.ylim(0,10)

    n_labels = len(np.unique(labels)) 
    for i in range(n_labels):
        idx = (labels==i)
        plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01)
    plt.show()
    plt.title("HC-DBSCAN result with MNIST dataset")
    plt.close(fig)

    
    fig = plt.figure()
    plt.legend()
    plt.xlim(0,18)
    plt.ylim(0,10)

    n_labels = len(np.unique(labels)) 
    for i in range(n_labels):
        idx = (labels==i)
        plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01)
    plt.show()
    plt.title("HC-DBSCAN result with MNIST dataset")
    plt.close(fig)
if __name__ == "__main__":
    main()
