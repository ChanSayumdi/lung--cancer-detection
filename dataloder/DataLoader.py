import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import pydicom as dicom
from skimage import exposure

class DataLoader:
    def __init__(self):
        print("dataloader")

    def loadData(self,path = "data\Train"):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        data_path= os.path.join(fileDir, path)
        filenames=os.listdir(data_path)
        ds = []
        fs = []
        for i in filenames:
            try:
                file_path = os.path.join( data_path , i)
                dcm_sample = dicom.dcmread(file_path).pixel_array
                dcm_sample=exposure.equalize_adapthist(dcm_sample)
                # Convert to float to avoid overflow or underflow losses.
                image_2d = dcm_sample.astype(float)

                # Rescaling grey scale between 0-255
                image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

                # Convert to uint
                image_2d_scaled = np.uint8(image_2d_scaled)
                ds.append(image_2d_scaled)
                fs.append(i)
            except:
                print("just ignore it")

        
        # print(image_2d_scaled)
        # self.debugDisplay(image_2d_scaled)

        x_train =  ds
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
        images = [cv.imread(i, cv.IMREAD_GRAYSCALE) for i in file_path]
        x_train =  np.array(images)
        y_train =  dataSet['class'].to_numpy()
        return x_train,y_train 

    def debugDisplay(self,image):
        cv.imshow('sample image dicom',image)
        cv.waitKey()

