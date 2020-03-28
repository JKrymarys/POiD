
import cv2 
import numpy as np
import basic_operations
import utils
import histogram
import filters
import transforms

img = utils.load_image('girl.bmp')
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

def h2_switch(x):
    if x == 1:
        utils.display_image('image', transforms.H_2(img), hist)
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

def s6_masks(mask):
    mask1 = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    mask2 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    mask3 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
    mask4 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    masks = [mask1, mask2, mask3,mask4]

    print("Chosen mask", mask)
    if mask != 0:
        utils.display_image('image', transforms.s_6(img, masks[mask]), hist)
    else:
        utils.display_image('image', img, hist)




# try:
cv2.namedWindow('image')
cv2.createTrackbar('contrast','image',10,20, contrast_param_change)
cv2.createTrackbar('brightness','image', 256,512, brightness_param_change)
cv2.createTrackbar('negative','image', 0,1, negative_switch)
cv2.createTrackbar('average_filter','image', 0,3, average_filter_change)
cv2.createTrackbar('median_filter','image', 0,3, median_filter_change)
cv2.createTrackbar('h2_filter','image', 0,1, h2_switch)
cv2.createTrackbar('s6_masks','image', 0,3, s6_masks)
utils.display_image('image',img,hist)
hist.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# except Exception as e:
#     print(f"Operation failed: {e}")
