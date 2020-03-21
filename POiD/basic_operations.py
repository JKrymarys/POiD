import numpy as np
import cv2
import utils


lut_size = 256

def look_up_table(parameter, operation):
    lut = [0 for x in range(lut_size)]

    for it in range(lut_size):
        lut[it] = operation(it, parameter);

    return lut

def basic_operations_core(parameter, img,  lut_operation ):
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    new_img = img.copy()
    lut = look_up_table(parameter, lut_operation)

    for y in range(img_height):
        for x in range(img_width):
            for c in range(3):
                new_img[x][y][c] = lut[img[x][y][c]]
    
    return new_img


def adjust_brightness_lut(pixel_value, parameter):
    if pixel_value + parameter < 0:
        return 0
    elif pixel_value + parameter > 255:
        return 255
    else:
        return pixel_value + parameter   


def adjust_contrast_lut(pixel_value, parameter):
    if (((pixel_value - 128)*parameter) + 128) > 255:
        return  255
    elif (((pixel_value - 128)*parameter) + 128) < 0:
        return 0
    else:
       return ((pixel_value - 128)*parameter) + 128


def create_negative_lut(pixel_value, parameter):
    new_value = 255 - pixel_value;
    if new_value < 0:
        return 0
    else :
        return new_value

    
def adjust_brightness(img, parameter): 
    return basic_operations_core(parameter, img, adjust_brightness_lut)

def adjust_contrast(img, parameter):
    return basic_operations_core(parameter, img, adjust_contrast_lut)

def create_negative(img):
    return basic_operations_core(0, img, create_negative_lut)
