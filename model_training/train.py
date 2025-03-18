import tensorflow as tf
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

for i in range(10):
    plt.imsave(f'test_{i}_true_{y_test[i]}.png', x_test[i], cmap='gray')

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[EarlyStopping(patience=3)]
)

model.save(r"C:\Users\mriva\OneDrive\Desktop\python scripts\neural network\backend\model\digit_classifier.keras")