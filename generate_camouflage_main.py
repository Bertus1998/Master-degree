from Camouflage1 import generate_camouflage as fcm_generate_texture
# from Camouflage2 import generate_camouflage as, determinate_color
from Camouflage2 import determinate_color
from Camouflage3 import generate_camouflage
from ImageManager import load_images_jpg
from Utils import extract_enclose_objects
import cv2 as cv

if __name__ == '__main__':
    # save_image("testName", fcm_generate_texture(load_images_jpg('./test_images/snow', 1)), 'snow')
    # save_image(("xd.jpg", fcm_generate_texture(load_images('./test_images')))
    # determinate_color(load_images_jpg('./test_images/snow', 2))
    generate_camouflage(load_images_jpg('./test_images/snow', 2))
    # images = load_images_jpg('./test_images/snow', 2)
    # extract_enclose_objects(images[0])
    #src = cv.imread(r"C:\Users\Admin\PycharmProjects\FCM\FCM\test_images\snow\4.jpg", cv.IMREAD_GRAYSCALE)
    #cv.imshow("orig", src)
    #extract_enclose_objects(src)
