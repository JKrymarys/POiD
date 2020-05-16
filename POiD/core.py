
import cv2 
import basic_operations
import utils
import histogram
import filters
import fft

import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt


img = utils.load_image('girl.bmp')
img_grayscale = color.rgb2gray(io.imread('girl.bmp'))

hist = histogram.create_histogram(img)


def contrast_param_change(x):
    value = x/10
    utils.display_image('image',basic_operations.adjust_contrast(img, value), hist)

def brightness_param_change(x):
    value = x - 256
    utils.display_image('image',basic_operations.adjust_brightness(img, value),hist)

def negative_switch(x):
    if x == 1:
        utils.display_image('image', basic_operations.create_negative(img), hist)
    else:
        utils.display_image('image', img, hist)

def average_filter_change(lvl):
    if lvl != 0:
        utils.display_image('image', filters.apply_average_filter(img, lvl), hist)
    else:
        utils.display_image('image', img, hist)

def median_filter_change(lvl):
    print("Chosen level", lvl)
    if lvl != 0:
        utils.display_image('image', filters.apply_median_filter(img, lvl), hist)
    else:
        utils.display_image('image', img, hist)


# try:
# cv2.namedWindow('image')

#------------------------ EXERCISE 1 --------------------------------
# cv2.createTrackbar('contrast','image',10,20, contrast_param_change)
# cv2.createTrackbar('brightness','image', 256,512, brightness_param_change)
# cv2.createTrackbar('negative','image', 0,1, negative_switch)
# cv2.createTrackbar('average_filter','image', 0,3, average_filter_change)
# cv2.createTrackbar('median_filter','image', 0,3, median_filter_change)
# utils.display_image('image',img,hist)
# hist.show()

#------------------------ EXERCISE 2 --------------------------------
x = np.matrix([[1,2,1],[2,1,2],[0,1,1]])
print(img_grayscale.shape)
X, m, n = fft.fft2(img_grayscale)
fshift = fft.fftshift(X)
magnitude_spectrum = 20*np.log(np.abs(fshift))
print('\nDFT is :')
print(X)
print('\nOriginal signal is :')
print(fft.ifft2(X, m, n))

plt.subplot(121),plt.imshow(img_grayscale, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# except Exception as e:
#     print(f"Operation failed: {e}")
