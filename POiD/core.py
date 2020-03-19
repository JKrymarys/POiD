
import cv2 
import basic_operations
import utils
import histogram
import filters

img = utils.load_image('cat.jpg')

def contrast_param_change(x):
    value = x/10
    utils.display_image('image',basic_operations.adjust_contrast(img, value))

def brightness_param_change(x):
    value = x - 256
    utils.display_image('image',basic_operations.adjust_brightness(img, value))

def negative_switch(x):
    if x == 1:
        utils.display_image('image', basic_operations.create_negative(img))
    else:
        utils.display_image('image', img)

def average_filter_change(x):
    if x == 1:
        utils.display_image('image', filters.average_filter(img))
    else:
        utils.display_image('image', img)

def median_filter_change(x):
    if x == 1:
        utils.display_image('image', filters.median_filter(img, 10))
    else:
        utils.display_image('image', img)


# try:
cv2.namedWindow('image')
cv2.createTrackbar('contrast','image',10,20, contrast_param_change)
cv2.createTrackbar('brightness','image', 256,512, brightness_param_change)
cv2.createTrackbar('negative','image', 0,1, negative_switch)
cv2.createTrackbar('average_filter','image', 0,1, average_filter_change)
cv2.createTrackbar('median_filter','image', 0,1, median_filter_change)
histogram.display_histograms(img) 
utils.display_image('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# except Exception as e:
#     print(f"Operation failed: {e}")
