import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

class DataLoader:
    def __init__(self):
        print("dataloader")

    def loadData(self,path = "data\Train"):
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
