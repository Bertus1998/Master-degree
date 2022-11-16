import collections

import cv2 as cv
import numpy as np

from ImageManager import resize_image, show_image
from Utils import count_distance


def generate_camouflage(images, number_of_color_per_image=4):
    image_colors_map = {}
    images_after_segmentation = []
    x_and_y_size = 300
    for image in images:
        # show_image(image.img)
        dim = (x_and_y_size, x_and_y_size)
        resized = resize_image(image.img, dim)
        blur = cv.medianBlur(resized, 7)
        reshaped_image = blur.reshape((-1, 3))
        float32 = np.float32(reshaped_image)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        ret, label, center = cv.kmeans(float32, number_of_color_per_image, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # show image after clustering
        output_image = np.uint8(center)[label.flatten()].reshape(resized.shape)
        # show_image(output_image)
        # center  = kolory
        # label wszystkie miejsca gdzie dany kolor ma być
        # list_of_images.append(output_image)
        image_colors_map[str(image)] = center
        images_after_segmentation.append((ret, label, center))
        # show_colors(center)
    layers = create_layers(images_after_segmentation, x_and_y_size, number_of_color_per_image)
    final_colors = calculate_final_colors(image_colors_map)
    final_layer = compute_final_layers(layers, x_and_y_size)
    generated_camouflage = np.uint8(final_colors)[final_layer.flatten().astype(int)].reshape(
        (x_and_y_size, x_and_y_size, 3))
    # show_image(generated_camouflage)
    # cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    return cv.resize(generated_camouflage, dsize=(1200, 1200), interpolation=cv.INTER_CUBIC)


def generate_all_possible_combination_of_layers(numbers_of_layers):
    numbers_of_layers = np.arange(numbers_of_layers)
    possible_combination = []
    length = len(numbers_of_layers)
    for i in range(0, length):
        for j in range(0, length):
            for k in range(0, length):
                for l in range(0, length):
                    if i != j & j != k & k != i & l != j & l != k & l != i:
                        possible_combination.append(
                            [numbers_of_layers[i], numbers_of_layers[j], numbers_of_layers[k], numbers_of_layers[l]])
    return possible_combination


def count_biggest_of_every_value(array):
    return collections.Counter(array).most_common()[0][1]


def find_number_of_colors(array):
    return collections.Counter(array)


def compute_final_layers(layers, size):
    elements_combinations = generate_all_possible_combination_of_layers(len(layers))
    all_possible_combinations_layers = np.zeros((len(elements_combinations), size, size))

    # laters to trzy warstwy  każda 600 na 600 i mają wartości od 0 d 2
    # i kombinacja, x element kombinacji, j i k rozmiar obrazu
    for i in range(len(elements_combinations)):
        for x in range(len(elements_combinations[0])):
            for j in range(size):
                for k in range(size):
                    if layers[elements_combinations[i][x]][j][k] == 1:  # jeśli dla danej
                        all_possible_combinations_layers[i][j][k] = elements_combinations[i][x]  # Chyba git

    max_value_of_elements = size * size
    choosen = None
    for i in range(len(all_possible_combinations_layers)):
        temp = count_biggest_of_every_value(all_possible_combinations_layers[0].flatten())
        if temp < max_value_of_elements:
            max_value_of_elements = temp
            choosen = i
    return all_possible_combinations_layers[choosen]


def create_layers(datas, size, number_of_colors):
    layers = np.zeros((number_of_colors, size, size))

    for color_number in range(0, number_of_colors):
        for data_number in range(0, len(datas)):
            for i in range(0, size):
                for j in range(0, size):
                    if datas[data_number][1][j + size * i] == color_number:
                        layers[color_number][i][j] = 1
        # show_image(layers[color_number])
    return layers

    # data (ret, label, center)
    # def replace_colors(datas, colors, resized):
    #     result = []
    #     for data in datas:
    #         result.append(np.uint8(colors)[data[1].flatten()].reshape(resized.shape))
    #


def calculate_final_colors(image_colors_map):
    final_colors = list(image_colors_map.items())[0][1]
    for i in (image_colors_map.values()):
        final_colors = find_pair(final_colors, i)
    # show_colors(final_colors)
    return final_colors


def find_pair(colors1, colors2):
    temp_colors = colors2
    results_colors = []
    for color1 in colors1:
        found_color = find_most_similar_color(color1, temp_colors)
        np.delete(temp_colors, np.where(temp_colors == found_color), None)
        results_colors.append((color1 + found_color) / 2)

    return results_colors


def find_most_similar_color(color, colors):
    the_most_similar_color = None
    distance = 1000
    for i in colors:
        temp_counted_distance = count_distance(i, color)
        if temp_counted_distance < distance:
            distance = temp_counted_distance
            the_most_similar_color = i
    return the_most_similar_color

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=612152
