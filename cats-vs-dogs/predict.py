import os
import random
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

base_dir = r'/mnt/d/cats_vs_dogs_small'
test_dir = os.path.join(base_dir, 'test')
# sample_dir = r'/mnt/d/Cats_vs_Dogs/data/test'

choose = 'models'

if choose == 'models':
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')


    model_standard = models.load_model('models/cats_and_dogs_small_standard.h5')
    test_loss, test_acc = model_standard.evaluate(test_generator, steps=50)
    print(f'Standard model accuracy: {test_acc}\nStandard model loss: {test_loss}')

    model_data_aug = models.load_model('models/cats_and_dogs_small_data_aug.h5')
    test_loss, test_acc = model_data_aug.evaluate(test_generator, steps=50)
    print(f'Data augmented model accuracy: {test_acc}\nData augmented model loss: {test_loss}')

    model_pretrained = models.load_model('models/cats_and_dogs_small_pretrained.h5')
    test_loss, test_acc = model_pretrained.evaluate(test_generator, steps=50)
    print(f'Pretrained VGG16 model accuracy: {test_acc}\nPretrained VGG16 model loss: {test_loss}')

    model_fine_tunning = models.load_model('models/cats_and_dogs_small_fine_tunning.h5')
    test_loss, test_acc = model_fine_tunning.evaluate(test_generator, steps=50)
    print(f'Fine-tunned VGG16 model accuracy: {test_acc}\nFine-tunned VGG16 model loss: {test_loss}')

model_fine_tunning = models.load_model('models/cats_and_dogs_small_fine_tunning.h5')

sample_category = 'cats' if random.randint(0, 1) == 0 else 'dogs'
sample_int = random.randint(1500, 1999)
sample_str = sample_category[:-1] + '.' + str(sample_int) + '.jpg'
category_path = os.path.join(test_dir, sample_category)
img_path = os.path.join(category_path, sample_str)
img = image.image_utils.load_img(img_path, target_size=(150, 150))

plt.imshow(img)

# Тестовое изображение
x = image.image_utils.img_to_array(img)

# Переформируем его в (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

predict_value = model_fine_tunning.predict(x)
print(predict_value[0][0])
print('cat' if predict_value == 0.0 else 'dog')

plt.show()
