import argparse
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageOps


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_image(path):
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).astype("float32")
    img /= 255.0

    return img


def main():
    parser = argparse.ArgumentParser(description="Mnist fashion image classifier")
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument(
        "--model", type=str, default="export/cnn.keras", help="Path to model"
    )

    args = parser.parse_args()

    model = keras.models.load_model(args.model)

    img = load_image(args.image_path)

    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]

    class_id = int(np.argmax(predictions))

    print("Prediction :" + CLASS_NAMES[class_id])
    print("Prediction probability :" + str(predictions[class_id]))


if __name__ == "__main__":
    main()
