import random

import numpy as np
import tensorflow as tf
from PIL import ImageOps
from matplotlib import pyplot as plt

SIZE_OF_IMAGE = 192


def normalize(input_image, input_mask):
    input_image = input_image / 255.0
    # input_mask = tf.cast(input_mask, tf.uint8)
    # input_mask -= 1
    return input_image, input_mask


#
# def normalize(input_image):
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     return input_image


def resize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (SIZE_OF_IMAGE, SIZE_OF_IMAGE), method="nearest")
    input_mask = tf.image.resize(input_mask.reshape(SIZE_OF_IMAGE, SIZE_OF_IMAGE, 1), (SIZE_OF_IMAGE, SIZE_OF_IMAGE),
                                 method="nearest")
    return input_image, input_mask


def load_networks():
    loaded_classification_model = tf.keras.models.load_model("classification_stream.h5")
    loaded_segmentation_model = tf.keras.models.load_model("segmentation_stream_200_no_preprocess_epoch.h5")

    loaded_classification_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics="accuracy")
    loaded_segmentation_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics="accuracy")
    return loaded_segmentation_model, loaded_classification_model


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


#
# def augment(input_image):
#     if tf.random.uniform(()) > 0.5:
#         # Random flipping of the image and mask
#         input_image = tf.image.flip_left_right(input_image)
#     return input_image


def preprocess_data(input_image, input_mask):
    # input_image, input_mask = resize(input_image, input_mask)
    # input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


# def preprocess_data(input_image):
#     input_image = augment(input_image)
#     input_image = normalize(input_image)
#     return input_image
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0].numpy()


def process_to_mask(val_preds):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return mask


# def normalize(input_image, input_mask):
#     # Normalize the pixel range values between [0:1]
#     img = tf.cast(input_image, dtype=tf.float32) / 255.0
#     input_mask -= 1
#     return img, input_mask
#

@tf.function
def load_train_ds(image, msk):
    img = tf.image.resize(image,
                          size=(SIZE_OF_IMAGE, SIZE_OF_IMAGE))
    mask = tf.image.resize(msk,
                           size=(SIZE_OF_IMAGE, SIZE_OF_IMAGE))

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    img, mask = normalize(img, mask)
    return img, mask


@tf.function
def load_test_ds(image, msk):
    img = tf.image.resize(image,
                          size=(SIZE_OF_IMAGE, SIZE_OF_IMAGE))
    mask = tf.image.resize(msk,
                           size=(SIZE_OF_IMAGE, SIZE_OF_IMAGE))

    img, mask = normalize(img, mask)
    return img, mask
