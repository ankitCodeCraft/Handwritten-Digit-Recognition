
import tensorflow as tf
from tensorflow.keras import layers, models


def build_efficientnet_model():
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Resizing(32, 32)(inputs)
    x = layers.Conv2D(3, kernel_size=1, padding="same")(x)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(32, 32, 3)
    )

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_efficientnet_model()
    model.summary()
    print("\nEfficientNetB0 model created successfully.")
