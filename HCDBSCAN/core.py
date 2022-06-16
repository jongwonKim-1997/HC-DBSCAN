import numpy as np
from clustering import DBSCAN
from clustering import evaluation_metric 
from scipy.stats import norm
import time
import torch
import gpytorch
import kernel

def predict_with_optimized_hyps(X_train, U_train, X_test,n_hyp,hyp_setting):
    U_train = U_train.reshape(-1)
    train_x = torch.tensor(X_train)
    train_y = torch.tensor(U_train)
    

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = kernel.AdvancedGPModel(train_x, train_y, likelihood,n_hyp).double()
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
def HC_DBSCAN(train_data = None ,show_data = None,train_labels=None, rho = 0.5,M = 100, n_max = 12, n_min = 8, ele_max = 200,n_init = 5, n_iter = 10, n_test = 50, str_cov = 'se',str_initial_method_bo='uniform',seed=0,clustering_method='dbscan',metric_method = 'daivies_bouldin',hyp_dict = {"eps" : 0.5,"min_samples" : 5, "p" : 2 } , bounds = np.array([[0.1,2],[0.1,15],[0.1,5]]), integer_var = [0,1],constraint='hard',data_name ='mnist',hyperparamter_optimization ='ADMMBO',constraint_function_list = None,acquisition_function='EI',alpha=2,beta = 4,initial_index=0,start_date='2022-06-16'):
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
        cluster = DBSCAN.clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

        cluster_data = cluster.fit(train_data)
        labels = cluster_data.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
            score = 10
        else:
            score = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

        F_train[F_n] = score
        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
            score2 = 10
        else:
            score2 = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
        NMI_train[F_n] = score2
        F_n +=1
        # Constraints 

        for con_idx in range(n_constraint):

            C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)

            real_C_train[con_idx,C_n] = constraint_function_list[con_idx](cluster_data)


        C_n +=1


    # Change minimum value 
    f_idx_list = []
    for i in range(n_iter):
        for j in range(alpha):
            f_idx_list.append(j+i*(alpha+beta*n_constraint))
    f_idx_list = np.array(f_idx_list)
    f_idx_list = f_idx_list + n_init
    f_idx_list = np.concatenate([f_idx_list, np.array([i for i in range(n_init)])])


    c_idx_list =[]
    for c_idx in range(n_constraint):
        c_idx_list1 = []
        for i in range(n_iter):
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


            next_x  = X_test[np.argmax(L1_list)]
            X_train[X_n] = next_x
            X_n +=1
            X = next_x 
            for idx_, key in enumerate(hyp_key):
                hyp_dict[key] = next_x[idx_]
            cluster = DBSCAN.clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

            cluster_data = cluster.fit(train_data)
            labels = cluster_data.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                score = 10
            else:
                score = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

            F_train[F_n] = score
            if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                score2 = 10
            else:
                score2 = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
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

                        cluster = DBSCAN.clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

                        cluster_data = cluster.fit(train_data)
                        labels = cluster_data.labels_
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score = 10
                        else:
                            score =evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

                        F_train[F_n] = score
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score2 = 10
                        else:
                            score2 =evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
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

                        cluster = DBSCAN.clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

                        cluster_data = cluster.fit(train_data)
                        labels = cluster_data.labels_
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score = 10
                        else:
                            score = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

                        F_train[F_n] = score
                        if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                            score2 = 10
                        else:
                            score2 = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
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

                    cluster = DBSCAN.clustering(clustering_method = clustering_method,hyp_dict = hyp_dict)

                    cluster_data = cluster.fit(train_data)
                    labels = cluster_data.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                        score = 10
                    else:
                        score = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = metric_method)

                    F_train[F_n] = score
                    if n_clusters == 0 or len(set(labels)) == 1 or n_clusters==1:
                        score2 = 10
                    else:
                        score2 = evaluation_metric.metric(train_data,labels,train_labels = train_labels, metric_method = 'normalized_mutual_info_score')
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


def iterate_experiment(ADMMBO_dict= None,exp_fun = None,iter_n=10,start_date='2022-06-16'):
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
        X_train, F_train, C_train,real_C_train,NMI_train,Y_train = exp_fun(**ADMMBO_dict,start_date=start_date)
        X_train_list.append(X_train)
        F_train_list.append(F_train)
        C_train_list.append(C_train)
        Y_train_list.append(Y_train)
        real_C_train_list.append(real_C_train)
        NMI_train_list.append(NMI_train)

    end = time.time()
   #print(end-start)
    

    return X_train_list, F_train_list, C_train_list, real_C_train_list,NMI_train_list,Y_train_list


