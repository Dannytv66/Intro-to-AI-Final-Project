import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

# -----------------------------
# Load and preprocess CIFAR-10
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -----------------------------
# Build the 3-block CNN model
# -----------------------------
model = Sequential([

    # ------- Block 1 -------
    Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    # ------- Block 2 -------
    Conv2D(64, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    # ------- Block 3 -------
    Conv2D(128, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    # ------- Dense layers -------
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# -----------------------------
# Compile the model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Learning rate schedule
# -----------------------------
lr_schedule = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

# -----------------------------
# Train for 40 epochs
# -----------------------------
history = model.fit(
    x_train, y_train,
    epochs=40,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[lr_schedule]
)

# -----------------------------
# Evaluate final accuracy
# -----------------------------
loss, acc = model.evaluate(x_test, y_test)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")

# -----------------------------
# Save the trained model
# -----------------------------
model.save("cifar10_cnn_3block.h5")
print("\nModel saved as cifar10_cnn_3block.h5")