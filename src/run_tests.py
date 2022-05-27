"""
run_tests.py
------------
File that includes the functions to run all or specific models with specific
datasets and save the results on a csv file.
"""
import numpy as np
import os
import pandas as pd
from datetime import datetime
from itertools import combinations
from tensorflow.keras import Model

from clustering import cluster_data, mvc_cluster_data, CLUSTERING_MODELS, MVCLUSTERING_MODELS
from datasets import preprocess_dataframe, get_dataset_sample
from face_recognition import get_recognition_model, RECOGNITION_MODELS
from sklearn.metrics import normalized_mutual_info_score as nmi_score

RESULTS_PATH = "../results.csv"

def score_cluster(df:pd.DataFrame, cluster_col:str="clusters") -> float:
    """
    Receives a pandas dataframe with the clusters of each image and the true
    labels to give a score.

    @returns: Score
    """
    return nmi_score(df["label"], df["clusters"])

def cluster_model(rec_model_name:str, cluster_model_name:str, df:pd.DataFrame) -> pd.DataFrame:
    """
    Runs the model with the images of an specific dataset.

    @returns: a pandas dataframe with the calculated clusters.
    """
    # Get recognition model
    model = get_recognition_model(rec_model_name)
    
    # Prepare dataframe
    data = preprocess_dataframe(df)

    # Get embedings
    embedings = model.predict(data)

    # Cluster embedings
    df["clusters"] = cluster_data(
        cluster_model_name, embedings, len(df["label"].unique()))
    return df

def cluster_models_cc(rec_models:list, cluster_model:str, df:pd.DataFrame) -> pd.DataFrame:
    """
    Runs the a combination of the models with a singular view clustering.

    @returns: dictionary utilizing the different clustering algorithms
    """
    models = []
    for model_name in rec_models:
        models.append(get_recognition_model(model_name))
    
    # Prepare dataframe
    data = preprocess_dataframe(df)

    # Get embedings
    embedings = np.array(models[0].predict(data))
    models.pop(0) # remove model that we already predicted
    for model in models:
        embedings = np.append(embedings, model.predict(data), axis=1)

    # Cluster embedings
    df["clusters"] = cluster_data(cluster_model, embedings,
        len(df["label"].unique()))
    return df

def cluster_models_mvc(rec_models: list, cluster_model_name:str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the combination of the models with a multiview clustering algorithm.

    @returns: a pandas dataframe with the calculated clusters.
    """
    # Get recognition models
    models = []
    for model_name in rec_models:
        models.append(get_recognition_model(model_name))
    
    # Prepare dataframe
    data = preprocess_dataframe(df)

    # Get embedings
    embedings = []
    for model in models:
        embedings.append(model.predict(data))

    # Cluster embedings
    df["clusters"] = mvc_cluster_data(cluster_model_name, embedings,
        len(df["label"].unique()))
    return df

"""
TODO: MVEC
def cluster_models_ensemble()
TODO: JULE
def cluster_models_jules()
"""

def evaluate_all(dataset:str, people:int, file_name:str) -> None:
    """
    Runs all the proposed models to evaluate clustering for face recognition.
    Then it saves the results in the general file: face-recognition/results.csv.
    And also creates a file of all the labels in <file_name>
    """
    # DATASET ----------------------------------------------------------------
    df = get_dataset_sample(dataset, people)
    if people <= 0:
        people = len(df["label"].unique())
    data = preprocess_dataframe(df)

    # GET EMBEDDINGS AND SINGLE CLUSTERING -----------------------------------
    embeddings = dict()
    for rec_model in RECOGNITION_MODELS:
        print("Getting embeddings from recognition model: "+rec_model+"...")
        model = get_recognition_model(rec_model)
        embeddings[rec_model] = model.predict(data)
        
        # Get scores for each clustering model
        for clus_model in CLUSTERING_MODELS:
            model_name = ' '.join(["S", rec_model, clus_model])
            print("Running model " + model_name + "...")
            df[model_name] = cluster_data(clus_model, embeddings[rec_model], people)
    
    # CC and MVC for every combination ---------------------------------------
    for l in range(2, len(RECOGNITION_MODELS)+1):
        for subset in combinations(RECOGNITION_MODELS, l):
            # CC -------------------------------------------------------------
            # Get embedings
            for i, rec_model in enumerate(subset):
                if i == 0: embs = embeddings[rec_model]
                else: embs = np.append(embs, embeddings[rec_model], axis=1)
            print(embs.shape)

            for clus_model in CLUSTERING_MODELS:
                model_name = ' '.join(["CC", *subset, clus_model])
                print("Running model " + model_name + "...")
                df[model_name] = cluster_data(clus_model, embs, people)

            # MVC ------------------------------------------------------------
            mv_embs = []
            for rec_model in subset:
                mv_embs.append(embeddings[rec_model])

            for clus_model in MVCLUSTERING_MODELS:
                model_name = ' '.join(["MVC", *subset, "mvc_"+clus_model])
                df[model_name] = mvc_cluster_data(clus_model, mv_embs, people)

    # TEST CSV ---------------------------------------------------------------
    print("Saving test csv at: " + file_name)
    df.to_csv(file_name)

    # RESULTS CSV ------------------------------------------------------------
    cols = ["Date", "Dataset", "# People"]
    d = [datetime.now().strftime("%Y-%m-%d %H:%M"), dataset, people]

    print("Evaluating clusters...")
    for col in df:
        if col != "path" and col != "label":
            nmi = nmi_score(df["label"], df[col])
            cols.append(col)
            d.append(nmi)
            print("Model " + col + ": " + str(nmi))

    if not os.path.exists(RESULTS_PATH):
        resdf = pd.DataFrame(columns=cols)
    else: resdf = pd.read_csv(RESULTS_PATH)

    s = pd.DataFrame([d], columns=cols)
#    resdf = pd.concat([resdf, s], axis=1, verify_integrity=True)
    resdf = resdf.append(s)
    resdf.to_csv(RESULTS_PATH, index=False)

def main():
    evaluate_all("yale", -1,
    "../bin/yale_"+datetime.now().strftime("%Y%m%d_%H%M")+".csv")

if __name__ == "__main__":
    main()
