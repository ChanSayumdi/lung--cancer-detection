import numpy as np
import matplotlib.pyplot as plt

class Preprocess:
    def __init__ (self):
        print("preprocessing")

    def Preprocess(self,x_train,y_train):
        # self.debugDisplay(x_train[5])
        x_train = x_train / 255.0
        x_train = x_train.reshape(x_train.shape[0],28,28,1)
        print(x_train[1].mean())
        return x_train,y_train

    def debugDisplay(self,image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()

