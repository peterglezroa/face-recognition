"""
clustering.py
-------------
File that contains the clustering algorithms used for the project
"""
from sklearn.cluster import KMeans
import numpy as np

def load_kmeans_model(file_name:str) -> KMeans:
    """
    Load cluster with Pickle

    @returns the loaded model
    """
    with open(file_name, "rb") as f:
        return pickle.load(f)
    raise Exception(f"Could not load kmeans model {file_name}")

def kmeans_model(data:np.array, n_clusters:int, file_name:str=None) -> dict:
    """
    Clusters using kmeans. If a file_name is specified, then it saved the model
    with pickle.

    @returns dictionary with the clusters and the centroids
    """
    model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, verbose=1)
    model.fit(data)

    # Save as .pkl file
    if file_name is not None:
        with open(file_name, "wb") as f:
            pickle.dump(model, f)

    return {
        "clusters": model.labels_,
        "centroids": model.cluster_centers_
    }

def dbscan_model(data:np.array, file_name:str=None) -> dict:
    """
    Clusters using kmeans. If a file_name is specified, then it saved the model
    with pickle.

    @returns dictionary with the clusters and the centroids
    """
    # eps = distance between two samples for one to be considered as in the
    # neighborhood of the other
    # min_samples = number of samples in a neighborhood for a point to be
    # considered as a core point
    model = DBSCAN(eps=0.5, min_samples=5, verbose=1)
    model.fit(data)

    # Save as .pkl file
    if file_name is not None:
        with open(file_name, "wb") as f:
            pickle.dump(model, f)

    return {
        "clusters": model.labels_,
        "centroids": model.cluster_centers_
    }

def cluster_data(model:str, data:np.array, n_clusters:int, file_name:str=None) -> dict:
    """
    Clusters the data according to the specified clustering algorithm

    @returns a dictionary:
        'clusters': numpy array of the resulting clusters of the data
        'centroids': centroids for the clusters if applied
    """
    if model == "kmeans":
        return kmeans_model(data, n_clusters, file_name=file_name)
#    if model == "dbscan":
#    if model == "agglomerative":
#    if model == "optics":

    raise Exception("Not a valid cluster algorithm name")
