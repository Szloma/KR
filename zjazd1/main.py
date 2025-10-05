import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers

parser = argparse.ArgumentParser(description='Tensorflow image recognition')
parser.add_argument('input', help='Input image')
args = parser.parse_args()
print(f'Processing file: {args.input}')

insurance_train = pd.read_csv(
    "dataset/insurance.csv",
    names=["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Charges"],
)
# insurance_train.head()
# ##tf.keras.loadimage
# insurance_features = insurance_train.copy()
# insurance_labels = insurance_features
#
# insurance_features = np.array(insurance_features)
#
# insurance_model = tf.keras.Sequential(
#     [layers.Dense(64, activation="relu"), layers.Dense(1)]
# )
# insurance_model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["Age"]
# )
#
# model_fit = insurance_model.fit(insurance_features, insurance_labels, epochs=10)

mnist = tf.keras.datasets.mnist
#tf.keras.loadimage(args.input)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.load_model("export/model.keras")

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model_fit = model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
model.save(
    "export/model.keras", overwrite=True
)


plt.plot(model_fit.history["accuracy"])
plt.plot(model_fit.history["loss"])
plt.title("Learning curve")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()
