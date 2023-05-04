import random
import matplotlib.pyplot as plt
from skimage import io
from utils import DirCategory

standard_deviation = 0.032


for dir_category in [
    DirCategory('test_dir', 11, 13),
    DirCategory('train_dir', 1, 11),
    DirCategory('val_dir', 13, 16)
]:
    for img_dir in range(1, 9):
        for img_name in range(dir_category.low, dir_category.high):
            for num in range(10):
                img_number = (img_name - 1) * 10 + (num + 1)
                img_path = f'resources/target-digitals/{dir_category.name}/{img_dir}/{img_number}.png'
                img = io.imread(img_path, as_gray=True)
                print(img_path)

                for noises in range(10):
                    for i in range(len(img)):
                        for j in range(len(img[i])):
                            img[i][j] += random.uniform(-standard_deviation, standard_deviation)

                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(0.06, 0.04)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(img, aspect='auto', cmap='gray')
                    img_number = (img_name - 1) * 100 + num * 10 + (noises + 1)
                    plt.savefig(f'resources/final-digitals/{dir_category.name}/{img_dir}/{img_number}.png')
