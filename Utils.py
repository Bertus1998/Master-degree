from __future__ import print_function

import cmath
import random

import cv2 as cv
import numpy as np
from skimage import morphology as morph
from skimage.measure import regionprops

# from Camouflage2 import PATCH_SIZE
THRESH_MINI = 220  # pixel value above which you want MAX VALUE
THRESH_MAX_VAL = 255  # pixel value above our THRESH_MIN will be converted to this value
MIN_SIZE = 2000


def check_similarity_of_patches(vector1, vector2):
    result = 0
    for i in range(len(vector1)):
        result += (vector1[i] - vector2[i]) ** 2
    return cmath.sqrt(result / 5)


def converts_sRGBColor_to_RGB_Array(n):
    return n.rgb_r, n.rgb_g, n.rgb_b


def count_distance(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def count_covered_area(patch):
    white_pixels = 0
    dim = patch.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if patch[i, j] == 255:
                white_pixels += 1
    return white_pixels / (patch.shape[0] * patch.shape[1])


def count_all_filled_pixels(patch):
    white_pixels = 0
    dim = patch.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if patch[i, j] == 255:
                white_pixels += 1
    return white_pixels


#
def count_circumference(img):
    regions = regionprops(img.astype(int))
    perimeter = regions[0].perimeter
    return perimeter


def count_circularity(patch):
    # https://stackoverflow.com/questions/29814229/how-to-calculate-the-value-of-the-roundness-shape-feature-of-an-image-contour
    regions = regionprops(patch.astype(int))
    perimeter = regions[0].perimeter
    area = count_all_filled_pixels(patch) - perimeter
    return 4 * cmath.pi * area / (perimeter ** 2)


def normalize_gauss(value, mean, max):
    return ((value - mean) / max + 1) / 2


def gauss_normalize_parameters(parameter_lists):
    return np.mean(parameter_lists), np.amax(parameter_lists)


# https://link.springer.com/content/pdf/10.1007/s001380050101.pdf
def count_rectangularity(patch):
    x1, y1, w, h = cv.boundingRect(patch)
    return count_all_filled_pixels(patch) / (w * h)


def count_aspect_ration(patch):
    x1, y1, w, h = cv.boundingRect(patch)
    return w / h


def extract_crops(threshold, given_img):
    # Standard Canny Edge Detection Implementation
    canny_output = cv.Canny(given_img, threshold, threshold * 2)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    return given_img[int(boundRect[i][1]):int(boundRect[i][1] + boundRect[i][3]),
           int(boundRect[i][0]):int(boundRect[i][0] + boundRect[i][2])]


def generate_random_background(colors, size):
    background = np.zeros((size, size, 3), dtype=int)
    max_color = len(colors)
    size_of_rectangle = int(size / 40)
    for i in range(0, size, size_of_rectangle):
        for j in range(0, size, size_of_rectangle):
            temp_rand = random.randint(0, max_color - 1)
            for x in range(0, size_of_rectangle):
                for k in range(0, size_of_rectangle):
                    background[i + x][j + k][0] = colors[temp_rand][0]
                    background[i + x][j + k][1] = colors[temp_rand][1]
                    background[i + x][j + k][2] = colors[temp_rand][2]
    return background


def extract_enclose_objects(img):
    # plt.imshow(img)
    # plt.show()
    kernel = np.ones((3, 3), np.uint8)
    img_int = cv.dilate(cv.erode(img, kernel, iterations=1), kernel, iterations=1)

    labeled = morph.label(img_int, connectivity=2)
    crops = []
    for i in np.unique(labeled)[1:]:
        im_obj = np.zeros(img_int.shape)
        im_obj[labeled == i] = 255
        crops.append(extract_crops(100, np.uint8(im_obj)))

    return crops

    # src_gray = cv.blur(img, (3, 3))
    # th, im_th = cv.threshold(src_gray, THRESH_MINI, THRESH_MAX_VAL, cv.THRESH_BINARY_INV)
    #
    # # Noise and Character Removal
    # im_th = noise_removal(im_th)
    #
    # # filling
    # im_floodfill, im_floodfill_inv = filling(im_th)

    # # cv.imshow("Floodfilled Image", im_floodfill)
    # # cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    #
    # # Creating Bounding boxes
    # Stop images from disappearing
