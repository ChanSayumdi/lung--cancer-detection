import numpy as np
import matplotlib.pyplot as plt
import cv2 

class Preprocess:
    def __init__ (self):
        print("preprocessing")

    def Preprocess(self,x_train,y_train):

        img = x_train[5]
        print(type(img))
        print (img.dtype)
        self.debugDisplay(img)

        median = cv2.medianBlur(img,3)
        canny = cv2.Canny(img,100,100)
        # x_train = x_train / 255.0
        # x_train = x_train.reshape(x_train.shape[0],28,28,1)
        # print(x_train[1].mean())
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

        plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,5),plt.imshow(median,cmap = 'gray')
        plt.title('median'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,6),plt.imshow(canny,cmap = 'gray')
        plt.title('canny'), plt.xticks([]), plt.yticks([])

        plt.show()
        return x_train,y_train

    def debugDisplay(self,image):
        cv2.imshow('sample image dicom',image)
        cv2.waitKey()

