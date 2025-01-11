import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from data import X_train, y_train, X_val, y_val
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 64

### Define model itself
# make this more functional in style rather than sequential
model = keras.Sequential()
model.add(
    keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        input_shape=INPUT_SHAPE,
        activation="relu",
        padding="same",
    )
)
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        input_shape=INPUT_SHAPE,
        activation="relu",
        padding="same",
    )
)
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=INPUT_SHAPE,
        activation="relu",
        padding="same",
    )
)
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=INPUT_SHAPE,
        activation="relu",
        padding="same",
    )
)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"],
)


### Train the model
p = Path(__file__).parent.joinpath("models").joinpath("final_model.keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=p.absolute(),
        save_best_only=True,
        # save_weights_only=True,
        monitor="val_loss",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]
# Data generator - adds in horizontally flipped images examples for training
datagen = ImageDataGenerator(
    # rotation_range=45,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # zoom_range=0.2,
    horizontal_flip=True,
    # validation_split=0.2,
)
train_generator = datagen.flow(X_train, y_train, batch_size=64)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)


### Plot the historic accuracy and loss over training iterations
train_hist = history.history
fig = plt.figure(figsize=(20, 7))
plt.subplot(121)
plt.plot(train_hist["accuracy"], label="acc")
plt.plot(train_hist["val_accuracy"], label="val_acc")
plt.grid()
plt.legend()

plt.subplot(122)
plt.plot(train_hist["loss"], label="loss")
plt.plot(train_hist["val_loss"], label="val_loss")
plt.grid()
plt.legend()
