import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('seaborn-whitegrid')

class History:
    """
    Neural network model performance visualizer
    >>> history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
    >>> modelplot = History(history)
    >>> modelplot.accuracy(savefig=True)
    >>> modelplot.loss(savefig=True)
    """
    def __init__(self, history):
        self.history = history
        self.epoch = range(len(history.history['loss']))
        # Accuracy and Train Loss
        self.accuracy = history.history['accuracy']
        self.loss = history.history['loss']
        # Accuracy and Test Loss
        self.val_accuracy = history.history['val_accuracy']
        self.val_loss = history.history['val_loss']

    def acc_plot(self, savefig=False):
        fig, ax = plt.subplots()
        ax.plot(self.epoch, self.accuracy, marker='x', label='Training Accuracy')
        ax.plot(self.epoch, self.val_accuracy, marker='x', label='Validation Accuracy')
        ax.set_title("Train and Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        plt.xticks([*range(len(self.epoch))])
        plt.legend()
        if savefig:
            fig.savefig("accuracy.png")
        plt.show()

    def loss_plot(self, savefig=False):
        fig, ax = plt.subplots()
        ax.plot(self.epoch, self.loss, marker='x', label='Training Loss')
        ax.plot(self.epoch, self.val_loss, marker='x', label='Validation Loss')
        ax.set_title("Train and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        plt.xticks([*range(len(self.epoch))])
        plt.legend()
        if savefig:
            fig.savefig("loss.png")
        plt.show()
