import cv2
import histogram


def load_image(file_path):
    img = cv2.imread(file_path)
    return img

def display_image(name,img, hist):
    cv2.imshow(name,img)
    histogram.update_histogram(img, hist)
    # histogram.show()


def display_histogram(img):
    histogram.display_histograms(img) 

def get_image_height(img):
    return img.shape[0]

def get_image_width(img):
    return img.shape[1]

