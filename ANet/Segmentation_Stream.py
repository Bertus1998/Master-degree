import tensorflow as tf
import numpy as np
# 1. resize images
from ImageManager import load_images_jpg, load_images_png, resize_image

train_images1 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\Images\Train'
test_images1 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\Images\Test'
train_images2 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Non-Camouflage\Images\Train'
test_images2 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Non-Camouflage\Images\Test'
masks1 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\GT'
masks2 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Non-Camouflage\GT\GT_Objectness'
SIZE_OF_IMAGE = 192

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(path):
    numbers = []
    images_to_train = load_images_jpg(path, 1)

    def dupa(image):
        return resize_image(image.img, (SIZE_OF_IMAGE, SIZE_OF_IMAGE))

    for image in images_to_train:
        numbers.append(int(str(image)[-5:]))
    return np.array(list(map(dupa, images_to_train))), numbers


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = tf.keras.layers.concatenate([x, conv_features])
    # dropout
    x = tf.keras.layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def unet_model():
    inputs = tf.keras.layers.Input(shape=(SIZE_OF_IMAGE, SIZE_OF_IMAGE, 3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="softmax")(u9)
    # unet model with Keras Functional API
    return tf.keras.Model(inputs, outputs, name="U-Net")


def load_masks(path, numbers):
    def dupa(image):
        return resize_image(image, (SIZE_OF_IMAGE, SIZE_OF_IMAGE)).reshape(SIZE_OF_IMAGE, SIZE_OF_IMAGE, 1)

    return np.array(list(map(dupa, load_images_png(path, numbers))))

# def FCN_model(len_classes=5, dropout_rate=0.2):


# Input layer
# input = tf.placeholder(tf.float32, [None, None, None, 3]) # size of image is not set, if error you have to resize image
# output = tf.placeholder(tf.float32, [None, None, None, 3])

# A convolution block


# Stack of convolution blocks

# def train(image)
