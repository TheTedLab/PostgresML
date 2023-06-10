import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
plt.clf()

test_loss, test_acc = model.evaluate(test_generator, steps=nb_test_samples // batch_size)
print("Loss: ", test_loss, "Accuracy:", test_acc)

y_pred_raw = model.predict(x_test)

y_pred = np.argmax(y_pred_raw, axis=1)

cm = confusion_matrix(y_test, y_pred, labels=np.arange(1, 9, 1), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(1, 9, 1))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap='Greens', ax=ax)
ax.set_xlabel('Предсказанные метки', fontsize=16)
ax.set_ylabel('Истинные метки', fontsize=16)
ax.set_title('Матрица ошибок', fontsize=20)
ax.set_xticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
ax.set_yticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
plt.tight_layout()
plt.savefig(f'graphs/confusion-matrix-normalized.png', bbox_inches='tight')
plt.show()
plt.clf()

cm = confusion_matrix(y_test, y_pred, labels=np.arange(1, 9, 1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(1, 9, 1))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap='Greens', ax=ax)
ax.set_xlabel('Предсказанные метки', fontsize=16)
ax.set_ylabel('Истинные метки', fontsize=16)
ax.set_title('Матрица ошибок', fontsize=20)
ax.set_xticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
ax.set_yticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
plt.tight_layout()
plt.savefig(f'graphs/confusion-matrix.png', bbox_inches='tight')
plt.show()
plt.clf()
plt.close(fig)


epochs_x, acc_y, val_acc_y, loss_y, val_loss_y = [], [], [], [], []

def animate(i):
    print(i)
    if i % 6 == 0 and i < 1200:
        epochs_x.append(epochs[i // 6])
        acc_y.append(acc[i // 6])
        val_acc_y.append(val_acc[i // 6])
        loss_y.append(loss[i // 6])
        val_loss_y.append(val_loss[i // 6])

    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(epochs_x, smooth_curve(acc_y), 'bo', label='Training acc')
    plt.plot(epochs_x, smooth_curve(val_acc_y), color='orange', label='Validation acc')
    # plt.xlim([0, 200])
    # plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.title('Training and validation accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_x, smooth_curve(loss_y), 'bo', label='Training loss')
    plt.plot(epochs_x, smooth_curve(val_loss_y), color='orange', label='Validation loss')
    # plt.xlim([0, 200])
    # plt.ylim([0.0, 3.0])
    plt.legend(loc='lower left')
    plt.title('Training and validation loss')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_x, acc_y, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc_y, color='orange', label='Validation acc')
    # plt.xlim([0, 200])
    # plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.title('Training and validation accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_x, loss_y, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss_y, color='orange', label='Validation loss')
    # plt.xlim([0, 200])
    # plt.ylim([0.0, 3.0])
    plt.legend(loc='lower left')
    plt.title('Training and validation loss')
    plt.tight_layout()


fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ani = FuncAnimation(fig, animate, frames=1500, interval=1000)

ani.save('results.mp4', writer = 'ffmpeg', fps = 60)
