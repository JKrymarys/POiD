import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
import histogram

lut_size = 256

def look_up_table(operation):
    lut = [0 for x in range(lut_size)]

    for it in range(lut_size):
        lut[it] = operation(it)

    return lut

def H_2(img):
    new_image = img.copy()
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    gsum = 0
    gmax = 255
    gmin = 0
    numofpixels = img_width*img_height
    
    for c in range(3):
        hist = histogram.calc_histogram(img)
        for i in range(lut_size):
            gsum = 0
            for m in range(i):
                gsum += hist[m]
            lut[i] = look_up_table(gmin + (gmax - gmin) * (1.0 / numofpixels) * gsum)
        
    for x in range(img_width):
        for y in range(img_height):
            new_image[x][y][c] = lut[img[x][y][c]]
            
    return new_image
    
def s_6(img):   
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])