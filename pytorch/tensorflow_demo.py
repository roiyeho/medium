# Author: Dr. Roi Yehoshua
# March 2023

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Setting random seed
tf.random.set_seed(0)

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
validation_split = 0.1
n_epochs = 10

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the network
model = keras.models.Sequential([  
    layers.Conv2D(32, 3, input_shape=[32, 32, 3], activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Train the model
train_start_time = time.time()
model.fit(X_train, y_train, batch_size=batch_size, 
          epochs=n_epochs, validation_split=validation_split)
train_elapsed_time = time.time() - train_start_time
print(f'Training completed in {train_elapsed_time:.3f}s')

# Evaluate the model on the training and test sets
train_results = model.evaluate(X_train, y_train, verbose=0)
print(f'Accuracy on training set: {train_results[1] * 100:.3f}%')

test_results = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy on test set: {test_results[1] * 100:.3f}%')
