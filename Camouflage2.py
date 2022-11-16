import copy
import multiprocessing
import random
import time
from multiprocessing import Process

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from patchify import patchify

from ImageManager import resize_image, extract_colors
from Multithreading import ThreadWithReturnValue
from Utils import converts_sRGBColor_to_RGB_Array, extract_enclose_objects, count_covered_area, \
    count_circumference, count_circularity, count_rectangularity, count_aspect_ration, gauss_normalize_parameters, \
    normalize_gauss, generate_random_background

NUMBER_OF_SELECTED_PATCHES = 10
CAMOUFLAGE_SIZE = 400
IMAGE_SIZE = 400
MIN_SIZE_PATCH = IMAGE_SIZE / 20
MAX_SIZE_PATCH = IMAGE_SIZE / 8
NUMBER_OF_PATCHES = CAMOUFLAGE_SIZE / 2
PATCH_SIZE = 100
NUMBER_OF_EXTRACTED_COLOR_PER_IMAGE = 3
NUMBER_OF_FINAL_COLORS = 4
MAX_NUMBER_OF_PATCHES = 8
IMSHOW = False


def color_difference(color1_lab, color2):
    color2_rgb = sRGBColor(color2[0] / 255, color2[1] / 255, color2[2] / 255)

    color2_lab = convert_color(color2_rgb, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e


def get_proportion(image, color, x, mang):
    color1_rgb = sRGBColor(color[0] / 255, color[1] / 255, color[2] / 255)
    color1_lab = convert_color(color1_rgb, LabColor)
    pixel_with_similar_color = 0
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            res = color_difference(color1_lab, image[i][j])
            if res < 10:
                pixel_with_similar_color += 1
    result = pixel_with_similar_color / (IMAGE_SIZE * IMAGE_SIZE)
    mang[x] = result


def color_selection(colors, images, number_of_colors=2):
    resized_images = []
    chosen_colors = {}
    colors_number = len(colors)
    number_of_images = len(images)
    start = time.time()
    threads = []
    for i in range(number_of_images):
        threads.append(ThreadWithReturnValue(target=resize_image, args=(images[i], [IMAGE_SIZE, IMAGE_SIZE])))
    for i in range(number_of_images):
        threads[i].start()
    for i in range(number_of_images):
        resized_images.append(threads[i].join())
    threads.clear()

    procs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    proportion_results = []

    for j in range(colors_number):
        for i in range(number_of_images):
            proportion_results.append(0.0)
            proc = Process(target=get_proportion,
                           args=(copy.deepcopy(resized_images[i]), copy.deepcopy(colors[j]), j * number_of_images + i,
                                 return_dict))
            procs.append(proc)
            proc.start()

    for i in range(len(procs)):
        procs[i].join()
        proportion_results[int(i / colors_number)] += return_dict.values()[i]
    for j in range(colors_number):
        chosen_colors[sRGBColor(colors[j][0], colors[j][1], colors[j][2])] = proportion_results[j]
    for i in range(len(chosen_colors) - number_of_colors):
        chosen_colors.pop(min(chosen_colors, key=chosen_colors.get))
    finish = time.time()
    print("Color selection took :")
    print(finish - start)
    return chosen_colors


# def find_the_most_validate_patches:


def determinate_color(images):
    HIGH = 0
    LOW = 1
    for i in range(len(images)):
        for j in range(len(images)):
            if i != j:
                image1 = images[i]
                image2 = images[j]
                dim = (IMAGE_SIZE, IMAGE_SIZE)
                resized1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)
                hist1 = cv2.calcHist([resized1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                resized2 = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)
                hist2 = cv2.calcHist([resized2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                value = np.sum(np.minimum(hist1, hist2)) / (np.minimum(np.sum(hist1), np.sum(hist2)))
                if value > HIGH:
                    HIGH = value
                if value < LOW:
                    LOW = value
    colors = []
    for i in images:
        img = resize_image(i, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_image = img.reshape((-1, 3))
        colors.append(extract_colors(reshaped_image, NUMBER_OF_EXTRACTED_COLOR_PER_IMAGE))
    colors = np.asarray(colors).reshape(len(colors) * len(colors[0]), 3)
    selected_colors = color_selection(colors, images, NUMBER_OF_FINAL_COLORS)
    data = np.arange(len(selected_colors)).reshape(1, len(selected_colors))
    cmap = plt.cm.gray
    norm = plt.Normalize(data.min(), data.max())
    rgba = cmap(norm(data))
    selected_colors = list(map(converts_sRGBColor_to_RGB_Array, list(selected_colors.keys())))
    for i in range(len(selected_colors)):
        rgba[0, i, 0] = selected_colors[i][0]
        rgba[0, i, 1] = selected_colors[i][1]
        rgba[0, i, 2] = selected_colors[i][2]
    start = time.time()
    layers = []
    # to trza zoptymalizować
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(len(images)):
        proc = Process(target=converted_into_layers,
                       args=(copy.deepcopy(images[i]), copy.deepcopy(selected_colors),
                             return_dict, i))
        processes.append(proc)
        proc.start()

    for process in processes:
        process.join()
    for i in range(len(images)):
        layers.append(return_dict[i])
    finish = time.time()
    print("Time of generating layers")
    print(finish - start)
    start = time.time()
    patches = []
    for layers_per_image in layers:
        for layer in layers_per_image:
            tem_patches = extract_enclose_objects(layer)
            for patch in tem_patches:
                if MIN_SIZE_PATCH < patch.shape[0] < MAX_SIZE_PATCH \
                        and MAX_SIZE_PATCH > patch.shape[1] > MIN_SIZE_PATCH:
                    patches.append(patch)

    finish = time.time()
    vectors_first_argument = []
    vectors_two_argument = []
    vectors_three_argument = []
    vectors_four_argument = []
    vectors_five_argument = []
    vectors = []
    for patch in patches:
        vectors_first_argument.append(count_covered_area(patch))
        vectors_two_argument.append(count_circumference(patch))
        vectors_three_argument.append(count_circularity(patch))
        vectors_four_argument.append(count_rectangularity(patch))
        vectors_five_argument.append(count_aspect_ration(patch))

    mean1, max1 = gauss_normalize_parameters(vectors_first_argument)
    mean2, max2 = gauss_normalize_parameters(vectors_two_argument)
    mean3, max3 = gauss_normalize_parameters(vectors_three_argument)
    mean4, max4 = gauss_normalize_parameters(vectors_four_argument)
    mean5, max5 = gauss_normalize_parameters(vectors_five_argument)

    for i in range(len(vectors_first_argument)):
        vectors.append(
            [normalize_gauss(vectors_first_argument[i], mean1, max1),
             normalize_gauss(vectors_two_argument[i], mean2, max2),
             normalize_gauss(vectors_three_argument[i], mean3, max3),
             normalize_gauss(vectors_four_argument[i], mean4, max4),
             normalize_gauss(vectors_five_argument[i], mean5, max5)]
        )
    selected_patches = []
    # for i in range(NUMBER_OF_SELECTED_PATCHES):
    #     selected_patches.append()

    camouflage = generate_camouflage(selected_colors, CAMOUFLAGE_SIZE, patches)
    plt.imshow(camouflage)
    plt.show()
    print("Time of extract patches")
    print(finish - start)
    # patches = extract_patches(images, selected_colors)
    # for patches_per_image in patches:
    #    for patch in patches_per_image:
    #        x = patch.shape
    #        for i in range(x[0]):
    #            for j in range(x[1]):
    #                extract_enclose_objects(patch[i][j])
    if IMSHOW:
        plt.imshow([np.array(selected_colors).astype(int)])
        plt.show()
    # print(patches)
    # print(selected_colors)
    # print(HIGH)
    # print(LOW)


def extract_patches(images, selected_colors):
    patches = []
    for i in images:
        patches.append(
            extract_patches_for_single_set_of_layers(
                converted_into_layers(
                    i, selected_colors
                )
            )
        )
    selected_patches = []
    for i in range(len(patches)):
        if check_if_patch_validation_true(patches[i]):
            selected_patches.append(patches[i])
    return patches


def check_if_patch_validation_true(patch):
    return True


def extract_patches_for_single_set_of_layers(layers):
    patches = []
    for i in range(3):
        temp = patchify(np.array(layers[i]), (IMAGE_SIZE / 4, IMAGE_SIZE / 4), PATCH_SIZE)
        patches.append(temp)
    # show_patches(patches)
    return patches


def show_patches(patches):
    for i in range(len(patches)):
        for j in range(len(patches[0])):
            for k in range(len(patches[0][0])):
                plt.imshow(np.array(patches[i][j][k]).astype(int), cmap='gray')
                plt.show()


def set_patches(random_background, patches, colors):
    for patch in patches:
        x = random.randint(0, len(random_background) - 1)
        y = random.randint(0, len(random_background[0]) - 1)
        random_color = colors[random.randint(0, len(colors) - 1)]
        for i in range(len(patch)):
            for j in range(len(patch[0])):
                if patch[i][j] == 255:
                    if (x + i) < CAMOUFLAGE_SIZE and y + j < CAMOUFLAGE_SIZE:
                        random_background[x + i][y + j][0] = random_color[0]
                        random_background[x + i][y + j][1] = random_color[1]
                        random_background[x + i][y + j][2] = random_color[2]
    return random_background


def generate_camouflage(selected_colors, size, patches):
    preprocess_patches = []
    kernel_array = (7, 7)
    for i in range(0, int(NUMBER_OF_PATCHES)):
        random_index_patch = random.randint(0, len(patches) - 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_array)
        closing = cv2.morphologyEx(patches[random_index_patch], cv2.MORPH_CLOSE, kernel)
        preprocess_patches.append(closing)
    random_background = generate_random_background(selected_colors, size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_array)

    return cv2.morphologyEx(set_patches(random_background, preprocess_patches, selected_colors).astype('uint8'),
                            cv2.MORPH_CLOSE, kernel)


def layer_draw(n, color):
    layer = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    layer.fill(255)
    color1_rgb = sRGBColor(color[0] / 255, color[1] / 255, color[2] / 255)
    color1_lab = convert_color(color1_rgb, LabColor)

    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            result_of_difference = color_difference(color1_lab, n[i][j])
            if result_of_difference > 10:
                layer[i][j] = 0
            elif (i <= 3 or i >= IMAGE_SIZE - 3) or (j <= 3 or j >= IMAGE_SIZE - 3):
                layer[i][j] = 0
            # plt.imshow(layer)
            # plt.show()

    return layer


def drawing_layers(n, colors):
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            min_difference = 10000
            selected_color = None
            for color in colors:
                color1_rgb = sRGBColor(color[0] / 255, color[1] / 255, color[2] / 255)
                color1_lab = convert_color(color1_rgb, LabColor)

                difference = color_difference(color1_lab, n[i][j])
                if min_difference > difference:
                    min_difference = difference
                    selected_color = color
            n[i][j] = selected_color
    layers = []
    for color in colors:
        temp = layer_draw(n, color)
        # plt.imshow(temp)
        # plt.show()
        layers.append(temp)
    return layers


def converted_into_layers(image, colors, return_direct, i):
    image = resize_image(image, [IMAGE_SIZE, IMAGE_SIZE])
    reshaped_image = image.reshape((-1, 3))
    float32 = np.float32(reshaped_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    ret, label, center = cv2.kmeans(float32, NUMBER_OF_EXTRACTED_COLOR_PER_IMAGE, None, criteria, 3,
                                    cv2.KMEANS_RANDOM_CENTERS)
    result = np.uint8(center)[label.flatten()].reshape(image.shape)
    from_color_answer = []
    if IMSHOW:
        plt.imshow(result)

    temp_layer = drawing_layers(result, colors)

    if IMSHOW:
        for i in range(len(from_color_answer)):
            cv2.imshow("image", from_color_answer[i])
            cv2.waitKey()
    # for i in temp_layer:
    #     plt.imshow(i)
    #     plt.show()
    return_direct[i] = temp_layer
    # map(converts_sRGBColor_to_RGB_Array, list(selected_colors.keys()))
    # cv2.imshow('res2', result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9700840
    # three-channel
    # histogram of two background images respectively
