import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functools import reduce
from clustering import DBSCAN
from clustering import evaluation_metric


def figure_print(X_train_list = None, F_train_list = None,C_train_list = None,real_C_train_list = None,NMI_train_list = None,train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,iter_n=10,initial_index=0,print_fig = False,start_date="2022-06-16"):
    
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
    if metric_method == "normalized_mutual_info_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            dbs_list.append(evaluation_metric.metric(train_data,labels))
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

    elif 'metric_method' == "davies_bouldin_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

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
            result1 = reduce((lambda x, y: x and y), constarint_list)
            if result1:
                nmi_list.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
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

    elif metric_method == "silhouette_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            nmi_list.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
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

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            nmi_list.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
       
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write(str(metric_method )+" :")
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
    cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

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

    if metric_method == "normalized_mutual_info_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

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
                nmi_list2.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=True))
                nmi_list3.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
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


    elif metric_method == "davies_bouldin_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

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
                nmi_list2.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=True))
                nmi_list3.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
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
        
    elif 'metric_method' == "silhouette_score":
        for X_val in X_val_list:
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = X_val[idx_]

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            nmi_list.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
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

            cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_labels = len(labels)
            noise_rate = sum(labels==-1)/n_labels
            noise_list.append(noise_rate)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_list.append(n_clusters)
            nmi_list.append(evaluation_metric.metric(train_data,labels,train_labels=train_labels,metric_method='normalized_mutual_info_score',noise=False))
       
        if print_fig:
            my_path = r"c:/Users/user/Documents/GitHub/Constraint_DBC/" + start_date
            f = open(my_path+"/"+"log_1003.txt", 'a')
            f.write("With constraints : "+"\n")
            f.write(str(metric_method)+" :")
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
    cluster = DBSCAN.clustering(clustering_method=clustering_method,hyp_dict= hyp_dict)

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
