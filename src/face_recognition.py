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

RECOGNITION_MODELS = ["vgg16", "resnet50", "senet50"]

def get_recognition_model(model_name:str) -> Model:
    if model_name == "vgg16":
        model = VGGFace(model="vgg16")
        # Remove softmax layer
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        return model
    if model_name == "resnet50":
        model = VGGFace(model="resnet50")
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        return model
    if model_name == "senet50":
        model = VGGFace(model="senet50")
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        return model

    raise Exception("Not a valid face recognition model name")

# facenet
# https://github.com/davidsandberg/facenet

# ageitgey
# https://github.com/ageitgey/face_recognition

# insightface
# https://github.com/deepinsight/insightface
# https://github.com/deepinsight/insightface/tree/master/model_zoo
