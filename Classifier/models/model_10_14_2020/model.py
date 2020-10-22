import json
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


class Model:

    config = None
    model = None

    def __init__(self):
        self.createModel()
        with open("Classifier\models\model_10_14_2020\hyperP.json", "r") as f:
            self.config = json.load(f)
        


    def createModel(self):
        print("model is crating")
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=(28,28,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        model.summary()

        self.model = model
