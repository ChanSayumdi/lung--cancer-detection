import numpy as np
import matplotlib.pyplot as plt
import cv2 
from preprocess.gabor import*

class Preprocess:
    def __init__ (self):
        gabor()
        print("preprocessing")

    def Preprocess(self,x_train,y_train):

        img = x_train[4]
        print(type(img))
        print (img.dtype)
        self.debugDisplay(img)   

        median = cv2.medianBlur(img,3)
        canny = cv2.Canny(img,100,100)
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        
        size =50
        if not size%2:
            size +=1
        kernel = np.ones((size,size),np.float32)/(size*size)
        filtered= cv2.filter2D(img,-1,kernel)
        filtered = img.astype('float32') - filtered.astype('float32')
        filtered = filtered + 127*np.ones(img.shape, np.uint8)
   
        
        
        plt.subplot(3,3,1),plt.imshow(img,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,5),plt.imshow(median,cmap = 'gray')
        plt.title('median'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,6),plt.imshow(canny,cmap = 'gray')
        plt.title('canny'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3,7),plt.imshow(filtered,cmap = 'gray')
        plt.title('filtered'), plt.xticks([]), plt.yticks([])

    
        plt.show()
        return x_train,y_train

    def debugDisplay(self,image):
        cv2.imshow('sample image dicom',image)
        cv2.waitKey()

