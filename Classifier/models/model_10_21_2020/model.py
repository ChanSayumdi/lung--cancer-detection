import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization


class Model:

    config = None
    model = None

    def __init__(self):
        self.createModel()
        with open("Classifier\models\model_10_21_2020\hyperP.json", "r") as f:
            self.config = json.load(f)
        


    def createModel(self):
        
        cnn4 = Sequential()
        cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
        cnn4.add(BatchNormalization())

        # cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # cnn4.add(BatchNormalization())
        # cnn4.add(MaxPooling2D(pool_size=(2, 2)))
        # cnn4.add(Dropout(0.25))

        # cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        # cnn4.add(BatchNormalization())
        # cnn4.add(Dropout(0.25))

        cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(MaxPooling2D(pool_size=(2, 2)))
        cnn4.add(Dropout(0.25))

        cnn4.add(Flatten())

        cnn4.add(Dense(512, activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(Dropout(0.5))

        cnn4.add(Dense(128, activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(Dropout(0.5))

        cnn4.add(Dense(1, activation='sigmoid'))

        cnn4.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

        self.model = cnn4
