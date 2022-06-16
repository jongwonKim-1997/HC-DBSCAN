import sklearn

def clustering(clustering_method = 'dbscan',hyp_dict=None,lambda1 =1):
    if clustering_method == 'kmeans':
        return sklearn.cluster.KMeans(**hyp_dict)
    if clustering_method == 'dbscan':
        return sklearn.cluster.DBSCAN(**hyp_dict)
