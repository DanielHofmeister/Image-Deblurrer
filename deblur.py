import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.restoration import denoise_tv_chambolle
from skimage.util import random_noise
import skimage.io
import skimage.viewer

print("Welcome to the image deblurrer program!")
path = input("Please enter the path to an image.\n")

image = skimage.io.imread(path)
A = np.array(image)
## A = random_noise(orig, mode='gaussian', seed=1)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(image)
ax.set_axis_off()
bx = fig.add_subplot(122)
# tuple to select colors of each channel line
colors = ("r", "g", "b")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=256, range=(0, 256)
    )
    bx = plt.plot(bin_edges[0:-1], histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")
plt.show()

weight = float(input("Please enter a denoising weight.\n"))
tv_denoised = denoise_tv_chambolle(A, weight)
plt.imshow(tv_denoised)
plt.axis("off")
plt.show()