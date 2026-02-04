import keras
from keras.applications import EfficientNetB0
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications.efficientnet import preprocess_input
from preprocessing import data_aug

inputs = Input(shape=(224,224,3))
x = data_aug(inputs)          # random flip/rotation/contrast
x = preprocess_input(x)       # normalize to [-1,1] for EfficientNet

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base_model.trainable = False
x = base_model(x, training=False)
outputs = Dense(4, activation="softmax")(x)

model = Model(inputs, outputs)