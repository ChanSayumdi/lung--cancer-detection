import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pydicom as dicom
from skimage import exposure

class DataLoader:
    def __init__(self):
        print("dataloader")

    def loadData(self,path = "Classifier\data\Train"):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        data_path= os.path.join(fileDir, path)
        filenames=os.listdir(data_path)
        ds=dicom.dcmread(os.path.join( data_path , filenames[1]))
        dcm_sample=ds.pixel_array
        dcm_sample=exposure.equalize_adapthist(dcm_sample)
        cv2.imshow('sample image dicom',dcm_sample)
        cv2.waitkey()
        
        
        x_train =  None
        y_train =  None
        return x_train,y_train 

    def loadDataOld(self,path = "data\Train"):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        data_path= os.path.join(fileDir, path)
        filenames=os.listdir(data_path)
        labels=[(0 if re.findall(r"[\w']+", i)[0]=="circles"  else 1) for i in filenames]
        train_df = pd.DataFrame(dict({'filename' : filenames, 'class' : labels}))

        dataSet = train_df.sample(frac=1) # Shuffle data
        file_path = [os.path.join( data_path , i) for i in dataSet.filename.tolist()] 
        images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in file_path]
        x_train =  np.array(images)
        y_train =  dataSet['class'].to_numpy()
        return x_train,y_train 

    def debugDisplay(self,image):
        plt.imshow(image, cmap='gray')  # graph it
        plt.show()
