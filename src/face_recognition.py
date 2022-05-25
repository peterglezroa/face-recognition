"""
pretrained_models.py
------------------
File used to obtain a keras model of the different pretrained feature
extractions models that will be then used for face recognition.
"""

# keras_vggface
# vgg16
from tensorflow.keras import Model
from keras_vggface.vggface import VGGFace

def get_recognition_model(model:str) -> Model:
    if model == "vgg16":
        return VGGFace(model="vgg16")
    if model == "resnet50":
        return VGGFace(model="resnet50")
    if model == "senet50":
        return VGGFace(model="senet50")

    raise Exception("Not a valid face recognition model name")

# facenet
# https://github.com/davidsandberg/facenet

# ageitgey
# https://github.com/ageitgey/face_recognition

# insightface
# https://github.com/deepinsight/insightface
# https://github.com/deepinsight/insightface/tree/master/model_zoo
