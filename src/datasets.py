# -*- coding: utf-8 -*-
"""
datasets.py
----------
File used for obtaining n amount of photos from the datasets obtained for this
project.
"""
import numpy as np
import pandas as pd
from PIL import Image
#import tensorflow as tf
#from tensorflow import keras


CELEBA_PATH="/home/peterglezroa/Documents/datasets/Face/CelebA"
FLICKR_PATH="/home/peterglezroa/Documents/datasets/Face/Flickr"
LFW_PATH="/home/peterglezroa/Documents/datasets/Face/Labeled Face in The Wild"
YOUTUBE_PATH="/home/peterglezroa/Documents/datasets/Face/Youtube"
YALE_PATH="/home/peterglezroa/Documents/datasets/Face/yalefaces"

def preprocess_df(df: pd.DataFrame) -> np.array:
    """
    Expects a dataframe:
        - the first column 'path': the path to the image file
        - the second column 'label': the label of said image

    @returns numpy array with the images value and the label to be processed
    through the model.
    """
    # Open image and convert it to grayscale and into a numpy array
    df["value"] = df.apply(lambda row:
        np.array(Image.open(
            f"{CELEBA_PATH}/img_align_celeba/{row['path']}"
        ).convert("L"), "uint8"),
        axis = 1
    )
    df = df.drop(columns=["path"])
    df = df.loc[:, ["value", "label"]]
    return df.to_numpy()


def obtain_celeba_images(n: int) -> np.array:
    """
    It is expected for the structure to be as following:
        <CELEBA_PATH>/
        ├─ identity_CelebA.txt
        ├─ img_align_celeba/
    * 'identity_CelebA.txt' is the downloaded identity text annotations without
    change from the dataset.
    * 'img_align_celeba' is the folder with all the downloaded images.

    @returns numpy array of n size with preprocessed images and its labels
    """
    df_labels = pd.read_csv(
        f"{CELEBA_PATH}/identity_CelebA.txt",
        names = ["path", "label"],
        sep=' '
    )
    # Select random images
    sample = df_labels.sample(n)
    # Preprocess
    return preprocess_df(sample)

def main():
    obtain_celeba_images(50)

if __name__ == "__main__":
    main()
