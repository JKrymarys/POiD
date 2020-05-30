
import cv2 
import basic_operations
import utils
import histogram
import filters
import fft
import segmentation

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

#fft
F, m, n = fft.fft2(img_grayscale)
fshift = fft.fftshift(F)
magnitude_spectrum = 20*np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)

#inverse fft
ifshift = fft.ifftshift(fshift, m, n)
img_back = fft.ifft2(ifshift,m,n).real

plt.subplot(141),plt.imshow(img_grayscale, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('Inverse Image'), plt.xticks([]), plt.yticks([])
plt.tight_layout()

img_filtered = filters.spectrum_modification(img_grayscale)

#apply mask and inverse DFT
low_pass_mask = filters.low_pass(img_grayscale)
high_pass_mask = filters.high_pass(img_grayscale)
band_pass_mask = filters.band_pass(img_grayscale)
band_reject_mask = filters.band_reject(img_grayscale)
edge_mask = filters.edge_detection(img_grayscale)

fshift_low = fshift*low_pass_mask
ishift_low = fft.ifftshift(fshift_low, m, n)
img_back_low = fft.ifft2(ishift_low,m,n).real

fshift_high = fshift*high_pass_mask
ishift_high = fft.ifftshift(fshift_high, m, n)
img_back_high = fft.ifft2(ishift_high,m,n).real

fshift_band = fshift*band_pass_mask
ishift_band = fft.ifftshift(fshift_band, m, n)
img_back_band = fft.ifft2(ishift_band,m,n).real

fshift_reject = fshift*band_reject_mask
ishift_reject = fft.ifftshift(fshift_reject, m, n)
img_back_reject = fft.ifft2(ishift_reject,m,n).real

fshift_edge = fshift*edge_mask
ishift_edge = fft.ifftshift(fshift_edge, m, n)
img_back_edge = fft.ifft2(ishift_edge,m,n).real

# plot both images
plt.figure(figsize=(11,6))
plt.subplot(231),plt.imshow(img_grayscale, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(img_back_low, cmap = 'gray')
plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(img_back_high, cmap = 'gray')
plt.title('High Pass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img_back_band, cmap = 'gray')
plt.title('Band Pass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(img_back_reject, cmap = 'gray')
plt.title('Band Reject Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(img_back_edge, cmap = 'gray')
plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])


plt.figure(figsize=(11,6))
plt.subplot(121),plt.imshow(img_grayscale, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_filtered, cmap = 'gray')
plt.title('Filtered'), plt.xticks([]), plt.yticks([])
plt.show()

# def on_mouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print('Start Mouse Position: ' + str(x) + ', ' + str(y))
#         s_box = x, y
#         boxes.append(s_box)

# boxes = []
# filename = 'girl.bmp'
# img = cv2.imread(filename, 0)
# resized = cv2.resize(img,(256,256))
# cv2.namedWindow('input')
# cv2.setMouseCallback('input', on_mouse, 0,)
# cv2.imshow('input', resized)
# cv2.waitKey()
# print("Starting region growing based on last click")
# seed = boxes[-1]
# cv2.imshow('input', segmentation.region_growing(resized, seed))
# print("Done. Showing output now")

# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# except Exception as e:
#     print(f"Operation failed: {e}")
