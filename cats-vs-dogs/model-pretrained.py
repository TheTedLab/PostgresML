import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

# Сверточная основа VGG16 - ImageNet

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.summary()

base_dir = r'/mnt/d/cats_vs_dogs_big'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Построение сети

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print('Кол-во обучаемых весов перед замораживанием сверточной основы:', len(model.trainable_weights))

conv_base.trainable = False

print('Кол-во обучаемых весов после замораживанием сверточной основы:', len(model.trainable_weights))

# Аугментация данных

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Данные проверки дополнять не нужно
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # Это целевой каталог
        train_dir,
        # Все изображения будут приведены к размеру 150x150
        target_size=(150, 150),
        batch_size=50,
        # Поскольку мы используем binary_crossentropy потерь, нам нужны binary метки
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=25,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['acc'])

# Обучение модели

history = model.fit_generator(
      train_generator,
      steps_per_epoch=125,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=125)

model.save('models/cats_and_dogs_big_pretrained.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('resources/pretrained-model/big-train-and-val-acc.png', bbox_inches='tight')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('resources/pretrained-model/big-train-and-val-loss.png', bbox_inches='tight')
plt.show()
