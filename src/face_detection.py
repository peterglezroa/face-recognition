import numpy as np
import os
import pandas as pd
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image

# Init detector one time
detector = MTCNN()

def extract_faces(filename:str, size:list = [224,224]) -> np.array:
    """
    Extracts ONE face from a given image

    @returns numpy array with the ROI values.
    """
    img = pyplot.imread(filename)

    # Check if it is grayscale
    if len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)

    # Detect face
    rois = detector.detect_faces(img)
    x, y, w, h = rois[0]["box"]
    face = img[x:x+w, y:y+h]
    
    # Resize to desired size
    pil_img = Image.fromarray(face)
    pil_img = pil_img.resize(size)
    return np.array(pil_img)

def test_extract_faces(df:pd.DataFrame, dest_folder:str):
    df = df.reset_index()
    for index, row in df.iterrows():
        np_img = extract_faces(row["path"])
        pil_img = Image.fromarray(np_img)
        pil_img.save(os.path.join(dest_folder,
        f"{str(row['label']).replace(' ', '_')}{os.path.splitext(row['path'])[1]}"))
