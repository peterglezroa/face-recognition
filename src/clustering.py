"""
clustering.py
-------------
File that contains the clustering algorithms used for the project
"""
from mvlearn import cluster as mv_cluster
from mvlearn.decomposition import GroupPCA
from sklearn import cluster as skl_cluster
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

CLUSTERING_MODELS = ["kmeans", "kmeans-pca", "agglomerative", "agglomerative-pca"]
MVCLUSTERING_MODELS = ["kmeans", "kmeans-pca"]

def load_kmeans_model(folder_name:str):
    """
    Load cluster with Pickle

    @returns the loaded model
    """
    with open(folder_name, "rb") as f:
        return pickle.load(f)
    raise Exception(f"Could not load kmeans model {folder_name}")

def calculate_optimal_distance(data:np.array) -> float:
    # Calculate optimal eps
    neigh = NearestNeighbors(n_neighbors = 2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
#    plt.plot(distances)
#    plt.show()
    return 0

def cluster_data(model_name:str, data:np.array, n_clusters:int, folder_name:str=None) -> list:
    """
    Clusters the data according to the specified clustering algorithm

    @returns a dictionary:
        'clusters': numpy array of the resulting clusters of the data
        'centroids': centroids for the clusters if applied
    """
    # Check if data is going to pass through pca
    pca = ("-pca" in model_name)
    if pca:
        model_name = model_name.split("-pca")[0]
        print("Applying pca to data...")
        # Random state to control random number and replication
        pca = PCA(n_components=min(100, n_clusters), random_state=42)
        pca.fit(data)
        print("Transforming data with pca...")
        data = pca.transform(data)

        # Save as .pkl file
        if folder_name is not None:
            print("Saving pca with pickle...")
            with open(os.path.join(folder_name, "pca.pkl"), "wb") as f:
                pickle.dump(model, f)


    # If n_clusters is known
    if model_name == "kmeans":
        print("Initializing KMeans...")
        model = skl_cluster.KMeans(n_clusters=n_clusters,
            n_init=10, max_iter=300)
    elif model_name == "agglomerative":
        print("Initializing Agglomerative...")
        model = skl_cluster.AgglomerativeClustering(n_clusters=n_clusters)

    # If n_clusters is unknown
    elif model_name == "dbscan":
        print("Initializing DBSCAN...")
        model = DBSCAN(eps=100, min_samples=2)
    elif model_name == "optics":
        print("Initializing OPTICS...")
        model = OPTICS(eps=30, min_samples=3)
    else: raise Exception("Not a valid cluster algorithm name")

    # PCA data if indicated
    print("Running clustering model...")
    clusters = model.fit_predict(data)
    print("Finished clustering.")

    # Save as .pkl file
    if folder_name is not None:
        print("Saving model with pickle...")
        with open(os.path.join(folder_name, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    return clusters

def mvc_cluster_data(model_name:str, data:np.array, n_clusters:int, folder_name:str=None) -> list:
    """
    """
    # Check if data is going to pass through pca
    pca = ("-pca" in model_name)
    if pca:
        model_name = model_name.split("-pca")[0]
        print("Applying pca to data...")
        # Random state to control random number and replication
        pca = GroupPCA(n_components=min(100,n_clusters), random_state=42)
        pca.fit(data)
        print("Transforming data with pca...")
        data = pca.transform(data)

        # Save as .pkl file
        if folder_name is not None:
            print("Saving pca with pickle...")
            with open(os.path.join(folder_name, "pca.pkl"), "wb") as f:
                pickle.dump(model, f)

    # If n_clusters is known
    if model_name == "kmeans":
        print("Initializing MultiView K Means...")
        model = mv_cluster.MultiviewKMeans(n_clusters=n_clusters, n_jobs=-1)
    else: raise Exception("Not a valid cluster algorithm name")

    # PCA data if indicated
    print("Performing Multi View clustering...")
    clusters = model.fit_predict(data)

    # Save as .pkl file
    if folder_name is not None:
        print("Saving model with pickle...")
        with open(os.path.join(folder_name, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    return clusters
