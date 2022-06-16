import sklearn
import numpy as np


# 3. Clustering Evaluation Metric
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
        import DBCV
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
