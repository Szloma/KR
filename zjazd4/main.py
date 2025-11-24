import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import keras_tuner
import keras


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")

##baseline
## accuracy: 1.0000 - loss: 0.0033
## epochs: 100
## lr: 0.0001
## batch size_ 8
EPOCHS = 25
NORMALIZER = None


def call_existing_code(units, activation, dropout, lr):
    model = keras.Sequential(
        [
            NORMALIZER,
            keras.layers.Dense(units=units, activation=activation, input_shape=(13,)),
        ]
    )

    if dropout:
        model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_model(hp):
    units = hp.Int("units", min_value=16, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=0.0001, max_value=0.01, sampling="log")

    model = call_existing_code(units, activation, dropout, lr)
    return model


def load_data():
    wine_train = pd.read_csv(
        "wine/wine.data",
        names=[
            "ID",
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ],
    )
    wine_train = wine_train.sort_values(by="ID")
    one_hot = pd.get_dummies(wine_train, columns=["ID"])
    x = one_hot.drop(columns=["ID_1", "ID_2", "ID_3"])
    y = one_hot[["ID_1", "ID_2", "ID_3"]]

    return x, y


def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test)

    print("BEST MODEL:")
    print("accuracy", acc, "- loss", loss, " - epochs", EPOCHS)

    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.show()


def main():
    global NORMALIZER
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x.values, y.values, test_size=0.2, random_state=42
    )

    NORMALIZER = keras.layers.Normalization()
    NORMALIZER.adapt(x_train)

    tuner = keras_tuner.RandomSearch(
        build_model, objective="val_loss", max_trials=5, overwrite=True
    )

    # tuner.search(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))
    # best_model = tuner.get_best_models()[0]
    # best_model.save("export/WineTuned.keras")
    # #best_model.summary()
    # best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    # print("BEST HYPERPARAMETERS:")
    # for key in best_hp.values:
    #     print(key, ":", best_hp.get(key))

    best_model = tf.keras.models.load_model("export/WineTuned.keras")

    evaluate_model(best_model, x_test, y_test)
    print("BASELINE:")
    print("accuracy: 1.0000 - loss: 0.0033 - epochs: 100")


if __name__ == "__main__":
    main()
