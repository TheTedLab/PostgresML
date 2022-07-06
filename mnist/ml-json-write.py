import json

import keras
import tensorflow as tf
from keras.layers import Dense, Flatten
from datetime import datetime

# Load and normalize MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

# Create model
model = keras.models.Sequential([
    Flatten(input_shape=x_train.shape[1:]),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Model fit with TensorBoard callback in log direction
model_logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
model_callback = tf.keras.callbacks.TensorBoard(log_dir=model_logdir, update_freq=1)
history = model.fit(x_train,
                    y_train,
                    epochs=6,
                    batch_size=128,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    callbacks=[model_callback])

# Save model config to disk (into PostgreSQL)
json_config = model.to_json()
with open('models/model_config.json', 'w') as json_file:
    json_file.write(json_config)

# Get model weights for json
model_weights = model.get_weights()

# List conversion for json dump, then need to go back to numpy.array when loading json
for i in range(len(model_weights)):
    model_weights[i] = model_weights[i].tolist()

# Save model weights to disk (into PostgreSQL)
json_weights = json.dumps(model_weights)
with open('models/model_weights.json', 'w') as json_file:
    json_file.write(json_weights)
