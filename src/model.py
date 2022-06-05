"""
model.py
--------
Model class to generate and save a model with pickle
"""
import numpy as np
import pandas as pd
from face_recognition import get_recognition_model
from clustering import 

DEFAULT_CONFIDENCE_DIS = 100

class FRModel:
    """
    FRModel
    -------
    Custom Face Recognition model according to this project.
    DO NOT INSTANCIATE THIS CLASS DIRECTLY. MUST USE from_folder OR create
    CLASSMETHODS!!!
    """
    @classmethod
    def from_folder(cls, folder:str) -> FRModel:
        """
        Constructor using a folder path
        The folder must contain the following structure:
        TODO
        """
        return

    @classmethod
    def create_from_data(cls, data:np.array, recs:list, pca:bool, cl:str,
    algo:str="CC", cd:float=DEFAULT_CONFIDENCE_DISTANCE, folder:str=None) -> FRModel:
        """
        Constructor to create a model from cero:
        - data: Dataframe to be used in the creation of this model
        - recognizers: list of strings used to load the recognizer models
        - pca: boolean to define if a dimention reduction is going to be used
        - cluster: cluster method name to be used
        - algo: ['CC', 'MVEC'] approach to be used.
        """
        if len(recognizers) < 2:
            raise Exception("The FRModel must receive atleast 2 different recognizers!")

        for recognizer in recognizers:
            __recognizers.append(get_recognition_model(recognizer))

        # Get embeddings
        if algo == "CC":
            sdf
        else "MVC"
        
        # TODO: Feature reduction
        
        cluster = 

        return FRModel(folder, recognizers,fd

        
    __folder = None
    __alg = "CC"
    __recognizers = []
    __pca = None
    __cluster = None
    __conf_distance = DEFAULT_CONFIDENCE_DISTANCE
    def __init__(self, folder:str, alg:str, recs:list, pca, cluster, conf_distance:float):
        this.__folder = folder
        this.__alg = alg
        this.__recognizers = recs
        this.__pca = pca
        this.__cluster = cluster
        this.__conf_distance = cd
