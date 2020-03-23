import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
size = 256

def calc_histogram(img):
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)

    histogram_values = [
        [0 for x in range(size)],
        [0 for x in range(size)],
        [0 for x in range(size)] 
    ]
    
    for y in range(img_height):
        for x in range(img_width):
            for c in range(3): #channels
                histogram_values[c].append(img[x][y][c])

    return histogram_values


def create_histogram(img): 
    r = img[:,:,2]
    g = img[:,:,1]
    b = img[:,:,0]

    histogram_values = calc_histogram(img)

    arr_b = np.array(histogram_values[0])
    arr_g = np.array(histogram_values[1])
    arr_r = np.array(histogram_values[2])

    plt.hist(arr_r,256,[0,256], facecolor='r')
    plt.hist(arr_g,256,[0,256], facecolor='g')
    plt.hist(arr_b,256,[0,256], facecolor='b')

    plt.title('Histogram (RGB)')
    plt.grid(True)
    
    return plt

def update_histogram(img, plt):
    r = img[:,:,2]
    g = img[:,:,1]
    b = img[:,:,0]

    histogram_values = calc_histogram(img)

    arr_b = np.array(histogram_values[0])
    arr_g = np.array(histogram_values[1])
    arr_r = np.array(histogram_values[2])

    plt.cla()
    plt.hist(arr_r,256,[0,256], facecolor='r')
    plt.hist(arr_g,256,[0,256], facecolor='g')
    plt.hist(arr_b,256,[0,256], facecolor='b')
    plt.draw()
   

