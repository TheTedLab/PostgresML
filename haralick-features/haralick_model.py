import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

# Каталог с данными для обучения
train_dir = 'resources/final-digitals/train_dir'
# Каталог с данными для тестирования
test_dir = 'resources/final-digitals/test_dir'
# Каталог с данными для тестирования
val_dir = 'resources/final-digitals/val_dir'
# Размеры изображения
img_width, img_height = 4, 6
# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 200
# Размер мини-выборки
batch_size = 10
# Количество изображений для обучения
nb_train_samples = 100
# Количество изображений для проверки
nb_validation_samples = 30
# Количество изображений для тестирования
nb_test_samples = 320

x_train, y_train = [], []
for image_dir in range(1, 9):
    print(image_dir)
    for image_number in range(1, 101):
        image_path = f'{train_dir}/{image_dir}/{image_number}.png'
        image = plt.imread(image_path)[:,:,:3]
        x_train.append(image)
        y_train.append(image_dir)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test, y_test = [], []
for image_dir in range(1, 9):
    print(image_dir)
    for image_number in range(101, 121):
        image_path = f'{test_dir}/{image_dir}/{image_number}.png'
        image = plt.imread(image_path)[:,:,:3]
        x_test.append(image)
        y_test.append(image_dir)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_val, y_val = [], []
for image_dir in range(1, 9):
    print(image_dir)
    for image_number in range(121, 151):
        image_path = f'{val_dir}/{image_dir}/{image_number}.png'
        image = plt.imread(image_path)[:, :, :3]
        x_val.append(image)
        y_val.append(image_dir)

x_val = np.array(x_val)
y_val = np.array(y_val)

datagen = ImageDataGenerator()

train_generator = datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size
)

test_generator = datagen.flow(
    x_test,
    y_test,
    batch_size=batch_size,
)

val_generator = datagen.flow(
    x_val,
    y_val,
    batch_size=batch_size,
)

model = Sequential()

model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    epochs=epochs,
    shuffle=True)

model.save('models/haralick_model_final.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs,
         acc, 'bo', label='Smoothed training acc')
plt.plot(epochs,
         val_acc, 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('graphs/final-haralick-model-train-and-val-acc.png', bbox_inches='tight')
plt.figure()

plt.plot(epochs,
         loss, 'bo', label='Smoothed training loss')
plt.plot(epochs,
         val_loss, 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('graphs/final-haralick-model-train-and-val-loss.png', bbox_inches='tight')
plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('graphs/final-smooth-haralick-model-train-and-val-acc.png', bbox_inches='tight')
plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('graphs/final-smooth-haralick-model-train-and-val-loss.png', bbox_inches='tight')
plt.show()

test_loss, test_acc = model.evaluate(test_generator, steps=nb_test_samples // batch_size)
print("Loss: ", test_loss, "Accuracy:", test_acc)
