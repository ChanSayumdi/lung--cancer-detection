
from dataloder.DataLoader import DataLoader
from preprocess.preprocess import Preprocess
from models.model_10_21_2020.model import Model
from trainer.train import Train
from predict.predict import Predict

def main():
    print("main")
    data = DataLoader()   
    x_train,y_train = data.loadData("Classifier\data\Train")
    preprocessor = Preprocess()
    x_train,y_train = preprocessor.Preprocess(x_train,y_train)
    model = Model()
    train = Train( x_train,y_train,model)
    predict = Predict()

    


if __name__ == "__main__":
    main()