import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers

LOAD_MODEL = True
PREDICT = True
TRAIN_INSURANCE= False

## Parser
parser = argparse.ArgumentParser(description="Tensorflow image recognition")
parser.add_argument("input", help="Input image")
args = parser.parse_args()
print(f"Processing file: {args.input}")


## Training custom model
if TRAIN_INSURANCE:
    insurance_train = pd.read_csv("dataset/insurance.csv")
    insurance_train = insurance_train.iloc[1:]  ##Remove the header

    insurance_train.head()

    insurance_features = insurance_train.copy()
    insurance_labels = insurance_features

    insurance_features = np.array(insurance_features)

    insurance_model = tf.keras.Sequential(
        [layers.Dense(64, activation="relu"), layers.Dense(1)]
    )
    insurance_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["Age"]
    )

    model_fit = insurance_model.fit(insurance_features, insurance_labels, epochs=10)
    # Something is broken and there's no time to fix it


## Loading/Training
model = None
if LOAD_MODEL:
    model = tf.keras.models.load_model("export/model.keras")
else:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model_fit = model.fit(x_train, y_train, epochs=30)
    model.evaluate(x_test, y_test)
    model.save("export/model.keras", overwrite=True)

    ##PLOTTING
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model_fit.history["accuracy"])
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(model_fit.history["loss"])
    plt.title("Loss")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()


##PREDICTION
if PREDICT:
    img = tf.keras.utils.load_img(args.input, color_mode="grayscale", target_size=(28, 28))
    input_arr = tf.keras.utils.img_to_array(img)
    input_arr = np.array([input_arr])

    prediction = model.predict(input_arr)
    print("predicted digit: ", max(prediction[0]))
