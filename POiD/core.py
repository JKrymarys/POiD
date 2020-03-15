
import cv2 
import basic_operations
import utils

try:
    img = utils.load_image('cat.jpg')
    utils.display_image('original',img)

    utils.display_image('brighter', basic_operations.increase_image_brightness(img, 50))
    utils.display_image('darker', basic_operations.decrease_image_brightness(img, 50))

    utils.display_image('brighter2', basic_operations.increase_image_brightness(img, 100))
    utils.display_image('darke2', basic_operations.decrease_image_brightness(img, 100))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Operation failed: {e}")
