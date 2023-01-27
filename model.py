import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from History import History

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
])

model.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images, train_labels, 
    validation_data=(test_images, test_labels), 
    epochs=5, verbose=2
)

train_loss, train_acc = model.evaluate(train_images, train_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)

model_plot = History(history)

model_plot.acc_plot(savefig=True)
model_plot.loss_plot(savefig=True)

with open("metrics.txt", 'w') as outfile:
        outfile.write(f"Train Loss : {train_loss:.3f}\n")
        outfile.write(f"Train Accuracy : {train_acc:.3f}\n")
        outfile.write(f"Test Loss : {test_loss:.3f}\n")
        outfile.write(f"Test Accuracy : {test_acc:.3f}\n")
        outfile.write("#")
