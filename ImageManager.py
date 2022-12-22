import glob
import os

import cv2
import imageio
import matplotlib.image as img
import numpy as np


class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return os.path.splitext(os.path.basename(self.__name))[0]


def load_images_jpg(img_dir, number):
    if number == 1:  # for coumflage number 1
        return [MyImage(file) for file in glob.glob(img_dir + '/*.*')]
    elif number == 2:  # for coumflage number 2
        return [img.imread(file) for file in glob.glob(img_dir + '/*.*')]


def load_images_png(img_dir):
    images = []
    for image_path in glob.glob(img_dir + "\\*.png"):
        images.append(imageio.imread(image_path))
    return images


def load_images_png_second_issue(img_dir, numbers):
    return


def save_image(name, image, place):
    # os.chdir('C:\\Users\\Admin\\PycharmProjects\\FCM\\FCM')
    os.chdir('C:\\Users\\Admin\\PycharmProjects\\FCM\\FCM\\outputImages\\' + place)
    cv2.imwrite(name + ".jpg", image)
    print("Image was saved")


def resize_image(image, dim, masks=False):
    if masks:
        return cv2.threshold(cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                             , 0.5, 1, cv2.THRESH_BINARY)[1]
    else:
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def extract_colors(image, number):
    data = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, dominant_colors = cv2.kmeans(data, number, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    print(dominant_colors)
    return dominant_colors


def show_image(image):
    cv2.imshow("images", image)
    cv2.waitKey()


def show_colors(colors):
    bars = []
    rgb_values = []
    for index, row in enumerate(colors):
        bar, rgb = create_bar(200, 200, row)
        bars.append(bar)
        rgb_values.append(rgb)
    cv2.imshow("Colors", np.hstack(bars))
    cv2.waitKey()


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)


def normalize(number):
    if number <= 0:
        return 1 + number
    elif number >= 1:
        return 1 - number
    else:
        return number
