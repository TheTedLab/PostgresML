import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skimage

fig = plt.figure(frameon=False)
fig.set_size_inches(6, 4)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

def update(i):
    image_number = (i // 3) + 1
    print(f'image_number: {image_number}, i: {i}')
    image = plt.imread(f'resources/final-digitals/train_dir/4/{image_number}.png')[:, :, :3]
    ax.imshow(image, aspect='auto', cmap='gray')

ani = FuncAnimation(fig, update, frames=300, interval=10)
ani.save('noise_matrix.mp4', writer='ffmpeg', fps=60)

plt.show()
