import os

from dataloder.DataLoader import DataLoader
from preprocess.preprocess import Preprocess
from models.model_10_21_2020.model import Model
from trainer.train import Train
from predict.predict import Predict
from segmentation.Segmentation import Segmentation
from dataloder.DCMLoader import DCMLoader

OUT_DIR = 'out/'

def main():
    print("main")
    data = DCMLoader()
    segment = Segmentation()
    for i in range(len(os.listdir("data/train/"))):
        images , file_names , patient = data.load("data/train/",i)
        # preprocessor = Preprocess()
        # x_train,y_train = preprocessor.Preprocess(x_train,y_train)
        # for j,image in enumerate(images):
        if len(images) > 0:
            segment.segmentation(images,file_names,patient,OUT_DIR)


    


if __name__ == "__main__":
    main()