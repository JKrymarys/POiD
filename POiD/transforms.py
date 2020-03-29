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
    
def s_6(img, mask):   
    new_image = img.copy()
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    
    img_region = np.ones((3,3))
    for c in range(3):
        for x in range(1, img_width-1):
            for y in range(1, img_height-1):
                s = 0
                s_divider = 0
                for l in range(-1,2):
                    for k in range(-1,2):
                        s += img[x+l][y+k][c] * mask[l+1][k+1]
                        s_divider += mask[l+1][k+1]
                if s_divider == 0:
                    new_image[x][y][c] = new_image[x][y][c] * s
                    print(s_divider)
                else:
                    new_image[x][y][c] = new_image[x][y][c] * (s/s_divider)
                    print(s_divider)

    return new_image


def o_2(img): 
    new_image = img.copy()
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    
    # img_region = np.ones((3,3))
    for c in range(3):
        for x in range(1, img_width-1):
            for y in range(1, img_height-1):
               new_image[x][y][c] = calculate_value_o_2(x,y,c,img)

    return new_image


def calculate_value_o_2(x,y,c,img): 
    temp = abs(img[x][y][c] - img[x+1][y+1][c]) + abs(img[x][y+1][c] - img[x+1][y][c])
    if(temp > 256):
        return 256
    elif(temp < 0):
        return 0

    return temp

