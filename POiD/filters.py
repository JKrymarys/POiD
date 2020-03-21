import numpy as np
import cv2
import utils

def average_filter(img):
    new_image = img.copy()
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    chanels = [b,g,r]
    chanel_no = 0
    for c in chanels:
        print(c)
        n = 0
        average_sum = 0
        for i in range(0, len(c)):
            for j in range(0, len(c[i])):
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        if (len(c) > (i + k) >= 0) and (len(c[i]) > (j + l) >= 0):
                            average_sum += c[i+k][j+l]
                            n += 1
                print(chanel_no)
                new_image[i][j][chanel_no] = (int(round(average_sum/n)))
                average_sum = 0
                n = 0
        chanel_no += 1
    return new_image


def calculate_median(image, x, y, c, lvl):
    temp_arr = []
    median_range = range(-lvl, lvl)

    if x-lvl > 0 and x+lvl < 255 and y-lvl > 0 and y+lvl < 255:
        for i in median_range:
            for j in median_range:
                temp_arr.append(image[x+i][y+j][c])


        temp_arr.sort()
        arr_length = len(temp_arr)
        return temp_arr[ arr_length//2 ]
    else:
        return image[x][y][c]


def median_filter(img, lvl=4):
    new_image = img.copy()
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)

    for y in range(img_height):
        for x in range(img_width):
            for c in range(3):
                new_image[x][y][c] = calculate_median(img, x, y, c, lvl)
    
    return new_image

