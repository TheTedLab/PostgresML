import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature.texture import graycomatrix, graycoprops

dir_category = 'val_dir'
for img_dir in range(1, 9):
    for img_name in range(13, 16):
        img_path = f'resources/sources/{dir_category}/{img_dir}/{img_name}.bmp'
        img = plt.imread(img_path)
        print(img_path)

        gray_img = np.average(img, axis=2).astype(np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R = img.copy()
        G = img.copy()
        B = img.copy()
        RG = img.copy()
        RB = img.copy()
        GB = img.copy()

        R[:, :, 1] = R[:, :, 2] = 0
        G[:, :, 0] = G[:, :, 2] = 0
        B[:, :, 0] = B[:, :, 1] = 0
        RG[:, :, 2] = 0
        RB[:, :, 1] = 0
        GB[:, :, 0] = 0

        fig = plt.figure(figsize=(18, 18))
        rgbx = fig.add_subplot(241)
        rx = fig.add_subplot(242)
        gx = fig.add_subplot(243)
        bx = fig.add_subplot(244)
        grayx = fig.add_subplot(245)
        rgx = fig.add_subplot(246)
        rbx = fig.add_subplot(247)
        gbx = fig.add_subplot(248)

        titles = ['Original', 'R component', 'G component', 'B component',
                  'Gray', 'RG component', 'RB component', 'GB component']
        axes = [rgbx, rx, gx, bx, grayx, rgx, rbx, gbx]
        components = [img, R, G, B, gray_img, RG, RB, GB]
        for x in axes:
            axe_index = axes.index(x)
            x.set_title(titles[axe_index])
            if x == grayx:
                x.imshow(components[axe_index], cmap='Greys')
            else:
                x.imshow(components[axe_index])


        # plt.show()

        def calc_component_features(component):
            component_gray_img = np.average(component, axis=2).astype(np.uint8)
            # plt.imshow(component_gray_img, cmap='Greys')
            # plt.show()
            glcm = graycomatrix(component_gray_img, [1], [np.pi / 2], levels=256, normed=True)
            haralick_features = {
                'contrast': graycoprops(glcm, 'contrast') / 145.0,
                'correlation': graycoprops(glcm, 'correlation'),
                'homogeneity': graycoprops(glcm, 'homogeneity'),
                'energy': graycoprops(glcm, 'energy')
            }
            glcm = np.reshape(glcm, (256, 256, 1))
            # plt.imshow(np.log(glcm + 1e-6))
            # plt.show()
            return haralick_features


        components_features = {
            'R component': dict(), 'G component': dict(), 'B component': dict(),
            'RG component': dict(), 'RB component': dict(), 'GB component': dict()
        }

        calc_features = [R, G, B, RG, RB, GB]
        calc_titles = titles[1:4] + titles[5:8]
        for x in calc_titles:
            component_index = calc_titles.index(x)
            components_features[x] = calc_component_features(calc_features[component_index])

        preprocessed_image = np.zeros([4, 6])
        comp_index = 0
        for component in components_features.values():
            # print(list(components_features.keys())[comp_index])
            feature_index = 0
            for key, val in component.items():
                # print(key, np.squeeze(val))
                preprocessed_image[feature_index][comp_index] = np.squeeze(val)
                feature_index += 1
            comp_index += 1

        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.06, 0.04)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(preprocessed_image, aspect='auto', cmap='gray')
        plt.savefig(f'resources/outputs/{dir_category}/{img_dir}/digital_image-{img_name}.png')
        # plt.show()
