import tensorflow as tf
from tensorflow.keras import layers
import params

model = tf.keras.Sequential(
    [  # This layer converts uint8 (0-255) to float32 (0-1) on the fly
        # layers.Rescaling(1.0 / 255, input_shape=(268, 66, 3)),
        layers.Rescaling(1.0 / 255, input_shape=(params.inputres[1], params.inputres[0], params.inputchannels)),
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu"),
        layers.Dropout(0.1),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
        layers.Dropout(0.1),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(50, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(1),
    ]
)
