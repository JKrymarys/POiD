
import cv2 
import basic_operations
import utils
import histogram
import filters

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
cv2.namedWindow('image')
cv2.createTrackbar('contrast','image',10,20, contrast_param_change)
cv2.createTrackbar('brightness','image', 256,512, brightness_param_change)
cv2.createTrackbar('negative','image', 0,1, negative_switch)
cv2.createTrackbar('average_filter','image', 0,3, average_filter_change)
cv2.createTrackbar('median_filter','image', 0,3, median_filter_change)
utils.display_image('image',img,hist)
hist.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# except Exception as e:
#     print(f"Operation failed: {e}")
