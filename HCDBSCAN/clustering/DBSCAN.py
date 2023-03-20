import sklearn

import sklearn.cluster

def clustering(clustering_method = 'dbscan',hyp_dict=None,lambda1 =1):
    hyp_key = hyp_dict.keys()
    for idx_, key in enumerate(hyp_key):
        if key =='min_samples':
            hyp_dict[key] = int(hyp_dict[key])
    if clustering_method == 'kmeans':
        return sklearn.cluster.KMeans(**hyp_dict)
    if clustering_method == 'dbscan':
        return sklearn.cluster.DBSCAN(**hyp_dict)
