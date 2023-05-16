import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature.texture import graycomatrix, graycoprops
from utils import DirCategory


def calc_component_features(img_component):
    img_component = np.true_divide(img_component, 32)
    img_component = img_component.astype(int)
    glcm = graycomatrix(img_component, [1], [0], levels=8, symmetric=False,
                        normed=True)
    haralick_features = {
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0]
    }
    return haralick_features


def print_image(image):
    row_count = 1
    for img_row in image:
        print(f'row: {row_count}')
        col_count = 1
        for img_col in img_row:
            print(f'{col_count}: {img_col}')
            col_count += 1
        row_count += 1


for dir_category in [
    DirCategory('test_dir', 11, 13),
    DirCategory('train_dir', 1, 11),
    DirCategory('val_dir', 13, 16)
]:
    for img_dir in range(1, 9):
        img_RED_global = []
        img_GREEN_global = []
        img_BLUE_global = []
        for img_name in range(dir_category.low, dir_category.high):
            img_path = f'resources/dataset/{dir_category.name}/{img_dir}/{img_name}.bmp'
            img_new = skimage.io.imread(img_path)
            print(img_path)

            img_components = {}

            # RED component
            img_red = img_new[:, :, 0]
            img_RED_global = img_red
            img_components['R'] = calc_component_features(img_red)

            # GREEN component
            img_green = img_new[:, :, 2]
            img_GREEN_global = img_green
            img_components['G'] = calc_component_features(img_green)

            # BLUE component
            img_blue = img_new[:, :, 0]
            img_BLUE_global = img_blue
            img_components['B'] = calc_component_features(img_blue)

            # RED-GREEN component
            img_r_g = img_RED_global - img_GREEN_global
            img_components['RG'] = calc_component_features(img_r_g)

            # RED-BLUE component
            img_r_b = img_RED_global - img_BLUE_global
            img_components['RB'] = calc_component_features(img_r_b)

            # GREEN-BLUE component
            img_g_b = img_GREEN_global - img_BLUE_global
            img_components['GB'] = calc_component_features(img_g_b)

            preprocessed_image = np.zeros([4, 6])
            comp_index = 0
            for component in img_components.values():
                feature_index = 0
                for key, val in component.items():
                    preprocessed_image[feature_index][comp_index] = val
                    feature_index += 1
                comp_index += 1

            fig = plt.figure(frameon=False)
            fig.set_size_inches(0.06, 0.04)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(preprocessed_image, aspect='auto', cmap='Greys')
            plt.savefig(f'resources/preprocess-digitals/{dir_category.name}/{img_dir}/digital_image-{img_name}.png')
