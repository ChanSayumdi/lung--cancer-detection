import tensorflow as tf
import matplotlib.pyplot as plt

class Train:

    learning_rate = None
    num_epochs = None
    model_name = None

    def __init__(self, x_train,y_train,model):
        super().__init__()        
        self.learning_rate = model.config["learning_rate"]
        self.num_epochs = model.config["num_epochs"]
        self.model_name =  model.config["model_name"]
        self.train(x_train,y_train,model.model)

    def train(self,x_train,y_train,model):
        history = model.fit(x_train, y_train, batch_size=32, epochs=self.num_epochs, validation_split=0.3)
        model.save("Classifier\\build\\" + self.model_name)
        self.plotDrafs(history,self.num_epochs)

    def plotDrafs(self,history,epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()   