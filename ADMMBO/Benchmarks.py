
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
