import random
import skimage
import numpy as np
import matplotlib.pyplot as plt
from utils import DirCategory

standard_deviation = 0.032


def make_noise(average_img, noise_range, diff, dir_name):
    for noises in range(noise_range):
        for i in range(len(average_img)):
            for j in range(len(average_img[i])):
                average_img[i][j] += random.uniform(-standard_deviation, standard_deviation)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.06, 0.04)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(average_img, aspect='auto', cmap='gray')
        img_number = noises + diff + 1
        plt.savefig(f'resources/final-digitals/{dir_name}/{img_dir}/{img_number}.png')


for img_dir in range(1, 9):
    total_img = [[0., 0., 0., 0., 0., 0.] for _ in range(4)]
    for dir_category in [
        DirCategory('test_dir', 11, 13),
        DirCategory('train_dir', 1, 11),
        DirCategory('val_dir', 13, 16)
    ]:
        for img_name in range(dir_category.low, dir_category.high):
            img_path = f'resources/preprocess-digitals/{dir_category.name}/{img_dir}/digital_image-{img_name}.png'
            img = skimage.io.imread(img_path, as_gray=True)
            print(img_path)

            row_count = 0
            for img_row in img:
                col_count = 0
                for img_col in img_row:
                    total_img[row_count][col_count] += img_col
                    col_count += 1
                row_count += 1

    average_img = np.true_divide(total_img, 15)

    make_noise(average_img, 100, 0, 'train_dir')
    make_noise(average_img, 20, 100, 'test_dir')
    make_noise(average_img, 30, 120, 'val_dir')
