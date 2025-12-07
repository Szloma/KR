import os
import json
import tensorflow as tf
import keras
import keras_tuner as kt
import numpy as np

from sklearn.metrics import confusion_matrix

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)


def build_dense_model(hp):
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(
                units=hp.Int("units", 64, 512, step=64),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            ),
            keras.layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)),
            keras.layers.Dense(10, activation="softmax"),
        ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float(
                "lr", min_value=0.0001, max_value=0.01, sampling="log"
            )
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(hp):
    model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            data_augmentation,
            keras.layers.Conv2D(
                filters=hp.Int("filters1", 32, 64, step=32),
                kernel_size=3,
                activation="relu",
            ),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(
                filters=hp.Int("filters2", 64, 128, step=64),
                kernel_size=3,
                activation="relu",
            ),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(
                hp.Int("dense_units", 64, 256, step=64),
                activation="relu",
            ),
            keras.layers.Dense(10, activation="softmax"),
        ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def eval_and_save(model, x_test, y_test, name):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "loss": float(loss),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
    }

    with open("export/" + name + "_metrics.json", "w") as file:
        json.dump(metrics, file)

    model.save("export/" + name + ".keras")

    print(name + "accuracy: " + str(acc) + " loss: " + str(loss))


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    dense_tuner = kt.RandomSearch(
        build_dense_model,
        objective="val_accuracy",
        max_trials=5,
        directory="tuner",
        project_name="fashion_dense",
    )

    dense_tuner.search(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=5,
    )

    best_dense = dense_tuner.get_best_models(1)[0]
    eval_and_save(best_dense, x_test, y_test, "dense")

    x_train_cnn = np.expand_dims(x_train, axis=-1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)

    cnn_tuner = kt.RandomSearch(
        build_cnn_model,
        objective="val_accuracy",
        max_trials=5,
        directory="tuner",
        project_name="fashion_cnn",
    )

    cnn_tuner.search(
        x_train_cnn,
        y_train,
        validation_split=0.2,
        epochs=5,
    )

    best_cnn = cnn_tuner.get_best_models(1)[0]
    eval_and_save(best_cnn, x_test_cnn, y_test, "cnn")


if __name__ == "__main__":
    main()
