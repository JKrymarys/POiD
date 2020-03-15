
import cv2 
import basic_operations
import utils

# try:
img = utils.load_image('cat.jpg')
utils.display_image('original',img)

utils.display_image('brighter', basic_operations.adjust_brightness(img, 50))
utils.display_image('darker', basic_operations.adjust_brightness(img, -50))

utils.display_image('brighter2', basic_operations.adjust_brightness(img, 100))
utils.display_image('darke2', basic_operations.adjust_brightness(img, -100))

utils.display_image('contrast0', basic_operations.adjust_contrast(img, 0.5))
utils.display_image('contrastbase', basic_operations.adjust_contrast(img, 1))
utils.display_image('contrast', basic_operations.adjust_contrast(img, 1.5))

utils.display_image('negative', basic_operations.create_negative(img))


cv2.waitKey(0)
cv2.destroyAllWindows()

# except Exception as e:
#     print(f"Operation failed: {e}")
