from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import(
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense,
    Input
)
import numpy as np
import cv2
Class = ["with_mask", "without_mask"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def get_model():
    baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224,224,3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7,7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False
    model.load_weights("mask_detector.model")
    return model

