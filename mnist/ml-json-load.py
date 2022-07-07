import json
import psycopg2
import keras
import numpy as np
import tensorflow as tf

# Load and normalize MNIST dataset (just for testing samples)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

print(x_test)

# Load new model config from json file
new_json_file = open('models/model_config.json', 'r')
new_json_config = new_json_file.read()
new_json_file.close()

new_model = keras.models.model_from_json(new_json_config)

# Load new model weights from json file
new_model_weights_file = open('models/model_weights.json', 'r')
new_weights = json.load(new_model_weights_file)
new_model_weights_file.close()

# Numpy array conversion in order to set the weights in the new model
for i in range(len(new_weights)):
    new_weights[i] = np.array(new_weights[i])

new_model.set_weights(new_weights)

# Compile new model and get score
new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

score = new_model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Test sample
sample = x_test[3:4]
for x in sample:
    for line in x:
        for num in line:
            if num != 0:
                print('*', end=' ')
            else:
                print('.', end=' ')
        print()
predict_value = new_model.predict(sample)
digit = np.argmax(predict_value)
print('digit = ', digit)
