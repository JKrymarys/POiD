import numpy as np
import cv2
import utils


def increase_image_brightness(img,brightness_change):
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    new_img = img.copy()

    for y in range(img_height):
        for x in range(img_width):
            for c in range(3):
                if img[y][x][c] + brightness_change > 255 :
                    new_img[y][x][c] = 255
                else:
                    new_img[y][x][c] = img[y][x][c] + brightness_change

    return new_img

def decrease_image_brightness(img,brightness_change):
    img_height = utils.get_image_height(img)
    img_width = utils.get_image_width(img)
    new_img = img.copy()

    for y in range(img_height):
        for x in range(img_width):
            for c in range(3):
                if img[y][x][c] - brightness_change < 0 :
                    new_img[y][x][c] = 0
                else:
                    new_img[y][x][c] = img[y][x][c] - brightness_change

    return new_img


