"""
run_tests.py
------------
File that includes the functions to run all or specific models with specific
datasets and save the results on a csv file.
"""
import pandas as pd
from tensorflow.keras import Model

from clustering import cluster_data 
from datasets import preprocess_dataframe, get_dataset_sample
from face_recognition import get_recognition_model

def score_cluster(df: pd.DataFrame) -> float:
    """
    Receives a pandas dataframe with the clusters of each image and the true
    labels to give a score.

    @returns: Score
    """
    return 0

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
    cs = cluster_data(cluster_model_name, embedings, len(df["label"].unique()))

    df["clusters"] = cs["clusters"]
    return df

def evaluate_model(model:Model) -> list:
    """
    Runs the model with all the datasets.

    @returns: list of floats for the scores of each dataset.
    """
    return []

def cluster_models_cc(models: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the a combination of the models with a singular view clustering.

    @returns: a pandas dataframe with the calculated clusters.
    """
    return pd.DataFrame()

def evaluate_models_cc(models: list) -> list:
    """
    Runs the model combination with all the datasets.

    @returns: list of floats for the scores of each dataset.
    """
    return []

def cluster_models_mvc(models: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the combination of the models with a multiview clustering algorithm.

    @returns: a pandas dataframe with the calculated clusters.
    """
    return pd.DataFrame()

def evaluate_models_mvc(models: list) -> list:
    """
    Runs the model combination with all the datasets.

    @returns: list of floats for the scores of each dataset.
    """
    return []

"""
TODO: JULE
def cluster_multiple_models_jule()
    ""
    Runs the combination of the models with Joint Unsupervised Learning of Deep
    Representations and Image Clusters (JULE).

    @returns: a pandas dataframe with the calculated clusters.
    ""
    return pd.DataFrame()

def evaluate_models_jule(models: list[Model]) -> list[float]:
    ""
    Runs the model combination with all the datasets.

    @returns: list of floats for the scores of each dataset.
    ""
    return []
"""

def evaluate_all(file_name: str) -> None:
    """
    Runs all the proposed models with all the databases and saves it in the
    specified file.
    """
    label_df = get_dataset_sample("lfw", -1)
    df = cluster_model("vgg16", "dbscan", label_df)
    df.to_csv(file_name)

def main():
    evaluate_all("../bin/test.csv")

if __name__ == "__main__":
    main()
