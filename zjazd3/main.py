import argparse

import pandas as pd
import numpy as np
import plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
##--Alcohol 13.2 --Malic_acid 2.77 --Ash 2.51 --Alcalinity_of_ash 18.5 --Magnesium 101 --Total_phenols 2.05 --Flavanoids 1.77 --Nonflavanoid_phenols 0.1 --Proanthocyanins 1.0 --Color_intensity 4.0 --Hue 1.05 --OD280_OD315 3.5 --Proline 10EPOCHS
EPOCHS = 20

def load_data():
    wine_train = pd.read_csv(
        "wine/wine.data",
        names=["ID","Alcohol", "Malic acid", "Ash", "Alcalinity of ash ", "Magnesium",
               "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins","Color intensity",
               "Hue", "OD280/OD315 of diluted wines", "Proline"
               ])

    one_hot = pd.get_dummies(wine_train, columns=["ID"])

    x = one_hot.drop(columns=["ID_1", "ID_2", "ID_3"])
    y = one_hot[["ID_1", "ID_2", "ID_3"]]
    return x, y

def train_model1(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model_fit = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    model.save("export/WineModel1.keras", overwrite=True)


def train_model2(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='hard_sigmoid',name="sigmoidLayer", input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model_fit = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    model.save("export/WineModel2S.keras", overwrite=True)


def plot(model_fit1,model_fit2,):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(model_fit1.history["accuracy"], label="Model 1")
    plt.plot(model_fit2.history["accuracy"], label="Model 2")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(model_fit1.history["loss"], label="Model 1")
    plt.plot(model_fit2.history["loss"], label="Model 2")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.show()

def get_better_model(model1, model2):
    x,y =load_data()
    model1_loss = model1.evaluate(x, y)[1]
    model2_loss = model2.evaluate(x, y)[1]
    if model1_loss < model2_loss:
        print("Model 1 jest lepszy")
        return model1
    else:
        print("Model 2 jest lepszy")
        return model2

def predict_from_args(args, model):
    feature_values = [
        args.Alcohol, args.Malic_acid, args.Ash, args.Alcalinity_of_ash, args.Magnesium,
        args.Total_phenols, args.Flavanoids, args.Nonflavanoid_phenols, args.Proanthocyanins,
        args.Color_intensity, args.Hue, args.OD280_OD315, args.Proline
    ]
    input_array = np.array([feature_values])
    pred_probs = model.predict(input_array, verbose=0)
    predicted_wine = np.argmax(pred_probs, axis=1)[0]+1
    print("Przewidywana kategoria wina:",predicted_wine)

# x, y = load_data()
# train_model1(x, y)
# train_model2(x, y)
# model1 = tf.keras.models.load_model("export/WineModel1.keras")
# model2 = tf.keras.models.load_model("export/WineModel2.keras")
# plot(model1.fit(x, y,epochs=EPOCHS),model2.fit(x, y,epochs=EPOCHS))

def main():
    parser = argparse.ArgumentParser(
        description="Klasyfikacja wina na podstawie modelu przetrenowanego na danych."
    )

    parser.add_argument("--Alcohol", type=float, required=True)
    parser.add_argument("--Malic_acid", type=float, required=True)
    parser.add_argument("--Ash", type=float, required=True)
    parser.add_argument("--Alcalinity_of_ash", type=float, required=True)
    parser.add_argument("--Magnesium", type=float, required=True)
    parser.add_argument("--Total_phenols", type=float, required=True)
    parser.add_argument("--Flavanoids", type=float, required=True)
    parser.add_argument("--Nonflavanoid_phenols", type=float, required=True)
    parser.add_argument("--Proanthocyanins", type=float, required=True)
    parser.add_argument("--Color_intensity", type=float, required=True)
    parser.add_argument("--Hue", type=float, required=True)
    parser.add_argument("--OD280_OD315", type=float, required=True)
    parser.add_argument("--Proline", type=float, required=True)
    args = parser.parse_args()
    model1 = tf.keras.models.load_model("export/WineModel1.keras")
    model2 = tf.keras.models.load_model("export/WineModel2.keras")
    predict_from_args(args,get_better_model(model1,model2))

if __name__ == "__main__":
    main()
