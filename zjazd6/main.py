import tensorflow as tf
from tensorflow.keras import layers, models
import os

cat_images = tf.keras.preprocessing.image_dataset_from_directory(
    "images", label_mode=None, image_size=(128, 128), batch_size=32, shuffle=True
)
no_cats = len(os.listdir("images/data"))

dataset = cat_images.map(lambda x: x / 255.0)
dataset = dataset.map(lambda x: (x, x))

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ]
)

autoencoder = models.Sequential(
    [
        # encoder
        layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        layers.Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(), padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(), padding="same"),
        layers.MaxPooling2D(2),
        # decoder
        layers.Conv2DTranspose(
            64, 3, strides=2, activation=tf.keras.layers.LeakyReLU(), padding="same"
        ),
        layers.Conv2DTranspose(
            32, 3, strides=2, activation=tf.keras.layers.LeakyReLU(), padding="same"
        ),
        layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
    ]
)
autoencoder.compile(optimizer="adam", loss="mse")  ##binary_crossentropy

autoencoder.fit(dataset, epochs=100)
autoencoder.save("export/autoencoder.keras", overwrite=True)

sample_batch, _ = next(iter(dataset))
reconstructed = autoencoder.predict(sample_batch)

for i in range(no_cats):
    tf.keras.preprocessing.image.save_img(
        "export/images/image_" + str(i) + ".png", reconstructed[i]
    )
