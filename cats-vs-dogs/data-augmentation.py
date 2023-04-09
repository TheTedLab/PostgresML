import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Аугментация данных

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

base_dir = r'/mnt/d/cats_vs_dogs_small'
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# Мы выбираем одно изображение для "дополнения"
img_path = fnames[5]

# Считываем изображение и изменяем его размер
img = image.image_utils.load_img(img_path, target_size=(150, 150))

# Преобразуем его в массив Numpy с формой (150, 150, 3)
x = image.image_utils.img_to_array(img)

# Переформируем его в (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# Приведенная ниже команда .flow() генерирует партии случайно преобразованных изображений.
# Она будет зацикливаться бесконечно, поэтому в какой-то момент нам нужно "разорвать" цикл
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.image_utils.array_to_img(batch[0]))
    plt.savefig(f'resources/data-augmentation/data-aug-{i}.png')
    i += 1
    if i % 4 == 0:
        break

plt.show()
