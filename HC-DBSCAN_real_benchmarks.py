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




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#%%



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



import pandas as pd

#%%
from ADMMBO import kernel
from clustering import evaluation_metric as EM
from ADMMBO import gaussian_process as GP
from ADMMBO import plot

def HC_DBSCAN(data_name_list,n_labels_list):
    output_csv = []
    for data_name in data_name_list:
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
            "eps" : 0.5,    
            "min_samples" : 5
        }
        bounds = np.array( [[0.0001,1],[2,train_data.shape[1]*np.round(np.log10(train_data.shape[0])) ]])

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
    return output_csv


def main():
    device = 'cpu'

    train_data0, train_labels = import_data(data='mnist',size=3000)
    show_data = embedding_data(train_data = train_data0,n_components = 2 )

    color_list = ['lightcoral','pink','r','y','g','c','b','m','green','navy']

    for i in range(10):
        idx = (train_labels==i)
        plt.scatter(show_data[idx,0],show_data[idx,1],label=i,color=color_list[i],alpha=0.5)

    plt.legend()
    plt.savefig("mnist_preview.svg")

    idx_list = [2,2,2,9,2,9,2,2,8,1]
    for i in range(10):
        idx = (train_labels==i)
        plt.scatter(show_data[idx,0],show_data[idx,1],alpha=0.01,color=color_list[i])
        plt.scatter(show_data[idx,0][idx_list[i]],show_data[idx,1][idx_list[i]],label=i,color=color_list[i],s=40)
    plt.legend()
    plt.savefig("mnist_pointview1.svg")

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
    plt.savefig("mnist_pointview2.svg")
    #%%

    start_date = '2022-06-02-real'
    my_path = r"c:/Users/user/Documents/GitHub/HC-DBSCAN/" + start_date
    os.makedirs(my_path, exist_ok=True) 
    os.makedirs(my_path+"/images", exist_ok=True) 
    f = open(my_path+"/"+"log_"+ start_date +".txt", 'w')
    f.write(start_date+ "\n")

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

    n_labels=0
    def constraint_function1(cluster_data):
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Feasible solution = negative value
        C_score = - min(n_labels - n_clusters , n_clusters -n_labels)

        return C_score
    constraint_function_list = [constraint_function1,constraint_function79, constraint_function94,constraint_function38,constraint_function35,constraint_function58]

    f.close()

    data_name_list = []
    n_labels_list = []
    
    
            
    my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
    col_names = ["HPO",	"data_name",	"constraint",	"n_clsuters",	"n_elements",	"mean_1",	"var_1",	"min_1",	"mean,_2",	"var_2",	"min_2",	"mean_noise"	,"var_noise",	"min_noise",	"n_cluster1","regret","n_cluster_draw_1",	"mean_4"	,"var_4"	,"min_4",	"nmi_mean"	,"nmi_var",	"nmi_min", "nmi_mean_penalty_noise"	,"nmi_var_penalty_noise",	"nmi_min_penalty_noise", "nmi_mean_no_noise"	,"nmi_var_no_noise",	"nmi_min_no_noise", 	"mean_noise2"	,"var_noise2",	"min_noise2"	,"n_cluster2"	,"regret2","n_cluster_draw_1"]
    output_csv = pd.DataFrame(output_csv)
    output_csv.columns = col_names
    output_csv.to_csv(my_path + "/" +start_date+".csv")


if __name__ == "__main__":
    main()
