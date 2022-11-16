import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ImageManager import resize_image, extract_colors

NUMBER_OF_FINAL_COLORS = 3
IMAGE_SIZE = 800
SIZE_OF_SCRAMBILNG = int(IMAGE_SIZE / 10)


def generate_camouflage(images):
    result_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=float)
    resized_images = []
    for image in images:
        resized_images.append(resize_image(image, (IMAGE_SIZE, IMAGE_SIZE)))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            for image in resized_images:
                result_image[i][j][0] += image[i][j][0]
                result_image[i][j][1] += image[i][j][1]
                result_image[i][j][2] += image[i][j][2]
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            result_image[i][j][:] = result_image[i][j][:] / len(resized_images)

    scrambled_image = scrambling_image(result_image.astype(int))
    Z = scrambled_image.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(scrambled_image.shape)
    kernel = np.ones((5, 5), np.uint8)

    # Using cv2.erode() method
    after_eroded = cv2.erode(res2, kernel)
    # plt.imshow(after_eroded)
    # plt.show()
    closing = cv2.morphologyEx(after_eroded, cv2.MORPH_CLOSE, kernel, iterations=4)
    blur = cv2.blur(closing, (7, 7))
    plt.imshow(blur)
    plt.show()


def scrambling_image(image):
    scrambled_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=float)
    for i in range(0, IMAGE_SIZE, SIZE_OF_SCRAMBILNG):
        for j in range(0, IMAGE_SIZE, SIZE_OF_SCRAMBILNG):
            x = random.randint(0, int(IMAGE_SIZE/SIZE_OF_SCRAMBILNG)-1)
            y = random.randint(0, int(IMAGE_SIZE/SIZE_OF_SCRAMBILNG)-1)
            crop_img2 = image[x*SIZE_OF_SCRAMBILNG:x*SIZE_OF_SCRAMBILNG + SIZE_OF_SCRAMBILNG, y*SIZE_OF_SCRAMBILNG:y* SIZE_OF_SCRAMBILNG + SIZE_OF_SCRAMBILNG]
            scrambled_image[j:j + SIZE_OF_SCRAMBILNG, i:i + SIZE_OF_SCRAMBILNG] = crop_img2
    return scrambled_image
