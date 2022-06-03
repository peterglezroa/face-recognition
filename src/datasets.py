# -*- coding: utf-8 -*-
"""
datasets.py
----------
File used for obtaining n amount of photos from the datasets obtained for this
project.

# VGGFace2
# https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw
"""
import os
import numpy as np
import pandas as pd
from numpy.random import choice
from PIL import Image

CELEBA_PATH="/home/peterglezroa/Documents/datasets/Face/CelebA"
LFW_PATH="/home/peterglezroa/Documents/datasets/Face/Labeled Face in The Wild"
MASKEDLFW_PATH="/home/peterglezroa/Documents/datasets/Face/lfw_masked/lfw_train"
YALE_PATH="/home/peterglezroa/Documents/datasets/Face/yalefaces"

AVAILABLE_DATASETS = ["celeba", "lfw", "maskedlfw", "yale"]

def preprocess_dataframe(df:pd.DataFrame, size:list=[224,224]) -> np.array:
    """
    Expects a dataframe:
        - the first column 'path': the path to the image file
        - the second column 'label': the label of said image

    @returns numpy array with the images value and the label to be processed
    through the model.
    """
    a = []

    df = df.reset_index()
    for index, row in df.iterrows():
        img = Image.open(row["path"]).resize(size)

        # Check if it is grayscale
        if len(img.size) < 3:
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            a.append(np.array(rgbimg))
        else:
            a.append(np.array(img))

    return np.array(a)

def extract_people_images(df: pd.DataFrame, n:int) -> pd.DataFrame:
    """
    From a given dataframe of the dataset, it looks for n unique people in the
    data and returns the images related to those people
    
    udf = df.groupby(by="label", as_index=False).agg({"n": pd.Series.nunique})
    udf = udf.sort_values(by=["n"])
    """
    uniq_labels = df["label"].unique()
    if len(uniq_labels) > n and n > 0: uniq_labels = choice(uniq_labels, n)
    return df[df["label"].isin(uniq_labels)]

def obtain_celeba_images(n_people:int) -> pd.DataFrame:
    """
    Unique labels: 10,177
    It is expected for the structure to be as following:
        <CELEBA_PATH>/
        ├─ identity_CelebA.txt
        ├─ img_align_celeba/
            ├─<images>
    * 'identity_CelebA.txt' is the downloaded identity text annotations without
    change from the dataset.
    * 'img_align_celeba' is the folder with all the downloaded images.

    @returns a pandas DataFrame of a n size sample with the following cols:
        - path: path to the location of the image
        - label: name of the person within the image
    """
    df = pd.read_csv(
        os.path.join(CELEBA_PATH, "identity_CelebA.txt"),
        names = ["path", "label"],
        sep=' '
    )

    # Extract according to unique number of people
    df = extract_people_images(df, n_people)

    root = os.path.join(CELEBA_PATH, "img_align_celeba/")
    df["path"] = root + df["path"]
    return df

def obtain_lfw_images(n_people:int) -> pd.DataFrame:
    """
    unique labels: 5,749
    It is expected for the structure to be as following:
    <LFW_PATH>/
    ├─<Person name/label>
        ├─<images>

    @returns a pandas DataFrame of a n size sample with the following cols:
        - path: path to the location of the image
        - label: name of the person within the image
    """
    paths = []
    labels = []
    for root, dirs, files, in os.walk(LFW_PATH):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root).replace('_', ' '))

    df = pd.DataFrame(data={"path": paths, "label": labels})
    return extract_people_images(df, n_people)

def obtain_maskedlfw_images(n_people:int) -> pd.DataFrame:
    """
    unique labels: 5,749
    It is expected for the structure to be as following:
    <MASKEDLFW_PATH>/
    ├─<Person name/label>
        ├─<images>

    @returns a pandas DataFrame of a n size sample with the following cols:
        - path: path to the location of the image
        - label: name of the person within the image
    """
    paths = []
    labels = []
    for root, dirs, files, in os.walk(MASKEDLFW_PATH):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root).replace('_', ' '))

    df = pd.DataFrame(data={"path": paths, "label": labels})
    return extract_people_images(df, n_people)

def obtain_yale_images(n_people:int) -> pd.DataFrame:
    """
    It is expected for the structure to be as following:
    <YALE_PATH>/
    ├─<images>

    @returns a pandas DataFrame of a n size sample with the following cols:
        - path: path to the location of the image
        - label: name of the person within the image
    """
    paths = []
    labels = []
    for root, dirs, files in os.walk(YALE_PATH):
        for file in files:
            label, ext = os.path.splitext(file)
            path = os.path.join(root, file)

            # If the file does not have extension (downloaded that way)
            if ext != ".gif" and ext != ".png":
                label += ext
                ext = ".gif"

            # Convert .gif to .png
            if ext == ".gif":
                img = Image.open(path)
                new_path = os.path.join(root,label+".png") 
                img.save(new_path, "png", optimize=True)
                os.remove(path)
                paths.append(new_path)
            else:
                paths.append(path)

            # Add label but remove the extra specification
            labels.append(label.split('.')[0])

    df = pd.DataFrame(data={"path": paths, "label": labels})
    return extract_people_images(df, n_people)

def get_dataset_sample(dataset:str, n:int) -> pd.DataFrame:
    if dataset == "celeba":
        return obtain_celeba_images(n)
    if dataset == "lfw":
        return obtain_lfw_images(n)
    if dataset == "maskedlfw":
        return obtain_maskedlfw_images(n)
    if dataset == "yale":
        return obtain_yale_images(n)
    raise Exception("Dataset name not found")

def main():
    df = get_dataset_sample("maskedlfw", 50)
    df.to_csv("test_dataset.csv")

if __name__ == "__main__":
    main()
