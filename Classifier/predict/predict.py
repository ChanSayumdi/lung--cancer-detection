import cv2
import tensorflow as tf

class Predict:
    CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value

    def __init__(self):
        self.predict()

    def prepare(self,filepath):
        IMG_SIZE = 28  # 50 in txt-based
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


    def predict(self):
        model = tf.keras.models.load_model("Classifier\\build\\save")
        prediction = model.predict([self.prepare('Classifier\\data\\realdata\\circle.jpg')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
        print(prediction)