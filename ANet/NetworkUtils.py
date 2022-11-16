import tensorflow as tf
from matplotlib import pyplot as plt

from ANet.Segmentation_Stream import SIZE_OF_IMAGE


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255
    return input_image, input_mask


def resize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (SIZE_OF_IMAGE, SIZE_OF_IMAGE), method="nearest")
    input_mask = tf.image.resize(input_mask.reshape(SIZE_OF_IMAGE, SIZE_OF_IMAGE, 1), (SIZE_OF_IMAGE, SIZE_OF_IMAGE),
                                 method="nearest")
    return input_image, input_mask


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def preprocess_data(input_image, input_mask):
    # input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


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
