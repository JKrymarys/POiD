import numpy as np
import cv2
import utils
import fft
from math import pi

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

def low_pass(img):
    # Circular HPF mask, center circle is 0, remaining all ones
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    r = 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    return mask

def high_pass(img):
    # Circular HPF mask, center circle is 0, remaining all ones
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols), np.uint8)
    r = 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    return mask

def band_pass(img):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    r_out = 80
    r_in = 10
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    return mask

def band_reject(img):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols), np.uint8)
    r_out = 50
    r_in = 20
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 0
    return mask

def edge_detection(img):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    r_out = 150
    r_in = 15
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    return mask

def spectrum_modification(img):
    N, M = img.shape
    F, m, n = fft.fft2(img)
    fshift = fft.fftshift(F)
    combined = [[0 for i in range(N-1)] for j in range(M-1)]
    phase_spectrum = np.angle(fshift)
    print(phase_spectrum.shape)
    k = 1
    l = 1
    for n in range(0, N-1):
        for m in range(0, M-1):
            combined[n][m] = np.multiply((phase_spectrum[n][m]), np.exp(1j*(((-n*k*2*pi)/N)+((-m*l*2*pi)/M)+(k+l)*pi)))
            print(combined[n][m])
    imgCombined = np.real(fft.ifft2(combined, m, n))
    print (imgCombined)
    return imgCombined
