# phase 1 - only the last dense layer learns (5-8 epochs) base_model.trainable = False
# phase 2 - unfreeze (5-10 epochs) base_model.trainable = True lower learning rate
import keras

from train import model

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

