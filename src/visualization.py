"""
visualization.py
----------------
Visualization methods to view images and clusters
"""
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

def generate_image(paths: list, size:list=[56,56]) -> Image:
    dims = math.ceil(math.sqrt(len(paths)))
    dst = Image.new("RGB", (size[0], size[1]))
    img_size = (size[0]//dims, size[1]//dims)

    for i, path in enumerate(paths):
        indx = i%dims
        img = Image.open(path).resize(img_size)
        dst.paste(img, (img_size[0]*(i%dims), img_size[1]*(i//dims)))
    return dst

def view_all_clusters(df:pd.DataFrame, fname:str, colname:str, img_size:list=[2048,2048]) -> None:
    BORDER_SIZE = 20
    clusters = df[colname].unique()
    cdims = math.ceil(math.sqrt(len(clusters)))

    # Create target image
    dst = Image.new("RGB", img_size)

    # Calculate dims after considering borders
    borders = BORDER_SIZE*(cdims-1)
    cell_size = ((img_size[0]-borders)//cdims, (img_size[1]-borders)//cdims)

    # Generate image per cluster
    for i, cluster in enumerate(clusters):
        # Get paths of cluster
        paths = df[df[colname] == cluster]["path"].tolist()
        img = generate_image(paths, size=cell_size)
        
        # Calculate img location
        x = cell_size[0]*(i%cdims) + (i%cdims)*BORDER_SIZE
        y = cell_size[1]*(i//cdims) + (i//cdims)*BORDER_SIZE

        dst.paste(img, (x,y))
    dst.save(fname)

def view_all_models_from_test(fname:str) -> None:
    df = pd.read_csv(fname)
    folder = os.path.splitext(fname)[0]

    if not os.path.isdir(folder):
        os.mkdir(folder)

    for col in df:
        if col != "path" and col != "label" and col != "Unnamed: 0":
            view_all_clusters(df, os.path.join(folder, col+".png"), col)

def main():
    view_all_models_from_test("../bin/yale20220527_0501.csv")

if __name__ == "__main__":
    main()
