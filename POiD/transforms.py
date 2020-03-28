import cv2
import utils
import matplotlib.pyplot as plt
from math import log1p
import numpy as np
import histogram

lut_size = 256

def H_2(img):
    new_image = img.copy()
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    gsum = 0
    gmax = 255
    gmin = 0
    alpha = 100
    numofpixels = img_width*img_height
    lut = [0 for x in range(lut_size)]
    hist = histogram.calc_histogram(img)

    for c in range(3):
        for i in range(lut_size):
            gsum = 0
            for m in range(i):
                gsum += hist[c][m]
            #lut[i] = gmin + (gmax - gmin) * (1.0 / numofpixels) * gsum
            lut[i] = gmin - (1 / alpha) * log1p(1 - (1.0 / numofpixels) * gsum)
        
    for x in range(img_width):
        for y in range(img_height):
            new_image[x][y][c] = lut[img[x][y][c]]
            
    return new_image
    
def s_6(img):   
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])