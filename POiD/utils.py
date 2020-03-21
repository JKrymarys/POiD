import cv2


def load_image(file_path):
    img = cv2.imread(file_path)
    return img

def display_image(name,image_url):
    cv2.imshow(name,image_url)

def get_image_height(img):
    return img.shape[0]

def get_image_swidth(img):
    return img.shape[1]

