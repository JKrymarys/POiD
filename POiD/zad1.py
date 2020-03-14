import numpy as np
import cv2

def load_image(file_path):
    # Load an color image in grayscale
    img = cv2.imread(file_path,0)
    return img

def display_image(image_url):
    cv2.imshow("test",image_url)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_height(img):
    return img.shape[0]

def get_image_width(img):
    return img.shape[1]


# def change_image_brightness():
#     img = load_image('cat.jpg')
#     new_image = []
#     it = 0
#     # print(img)
#     # for pixel_value in img:
#         # new_image.add(pixel_value + 10)
    
#     # displsay_image(new_image)





try:
    img = load_image('cat.jpg')
    print(get_image_height(img))
    print(get_image_width(img))

    display_image(img)

except Exception as e:
    print(f"Upsik: {e}")

#display_image(img)