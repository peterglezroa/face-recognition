"""
visualization.py
----------------
Visualization methods to view images and clusters
"""
import math
from PIL import Image

def view_cluster_as_image(cluster:list, max_imgs:int=8, size:list=[100,100]) -> None:
    dims = math.ceil(math.sqrt(len(cluster)))

    dst = Image.new("RGB", (dims*size[0], dims*size[1]))

    for i, path in enumerate(cluster):
        indx = i%dims
        img = Image.open(path).resize(size)
        dst.paste(img, (size[0]*(i%dims), size[1]*math.floor(i/dims)))

    dst.show()

import pandas as pd
def main():
    df = pd.read_csv("test.csv")
    view_cluster_as_image(df[df["clusters"] == 0]["path"].tolist())

if __name__ == "__main__":
    main()
