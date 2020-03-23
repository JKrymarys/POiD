import numpy as np
import cv2
import utils

def filter(img, lvl, filter_type):
    new_image = img.copy()
    b = new_image[:,:,0]
    g = new_image[:,:,1]
    r = new_image[:,:,2]

    chanels = [b,g,r]
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)

    for c in chanels:
        for x in range(img_height):
            for y in range(img_width):
                c[x][y] = filter_type(x, y, c, lvl, img_height, img_width)
    
    new_image = np.dstack((b, g, r))
 
    return new_image

def apply_average_filter(img, lvl):
    return filter(img, lvl, average)

def apply_median_filter(img, lvl):
    return filter(img, lvl, median)

def average(x, y, c, lvl, img_height, img_width):
    new_pixel = c[x][y]
    median_range = range(-lvl, lvl+1)
    average_sum = 0
    n = 0

    if x-lvl > 0 and x+lvl < img_height and y-lvl > 0 and y+lvl < img_width:
        for i in median_range:
            for j in median_range:
                average_sum += c[x+i][y+j]
                n += 1
        new_pixel = (int(round(average_sum/n)))
        average_sum = 0
    
    return new_pixel

def median(x, y, c, lvl, img_height, img_width):
    temp_arr = []
    median_range = range(-lvl, lvl+1)
    print(median_range)
    if x-lvl > 0 and x+lvl < img_height and y-lvl > 0 and y+lvl < img_width:
        for i in median_range:
            print(i)
            for j in median_range:
                print(j)
                temp_arr.append(c[x+i][y+j])
        print(temp_arr)
        temp_arr.sort()
        print(temp_arr)
        arr_length = len(temp_arr)
        new_pixel = temp_arr[ arr_length//2 ]
        print(new_pixel)
    else:
        new_pixel = c[x][y]

    return new_pixel
