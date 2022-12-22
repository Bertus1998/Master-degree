import numpy as np
import tensorflow as tf

from ANet.NetworkUtils import preprocess_data
from ANet.Segmentation_stream import load_image, load_masks, SIZE_OF_IMAGE


def learn_classification_stream():

    def classification_stream():
        inputs = tf.keras.Input(shape=(SIZE_OF_IMAGE, SIZE_OF_IMAGE, 3))
        layer1 = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer="he_normal")(inputs)
        layer2 = tf.keras.layers.Dropout(rate=0.5)(layer1)
        layer3 = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer="he_normal")(layer2)
        layer4 = tf.keras.layers.Dropout(rate=0.5)(layer3)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(layer4)
        return tf.keras.Model(inputs, outputs, name="Classification_Stream")

    train_images1 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\Images\Train'
    test_images1 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\Images\Test'
    train_images2 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Non-Camouflage\Images\Train'
    test_images2 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Non-Camouflage\Images\Test'
    masks1 = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\GT'
    no_camouflage_mask = r"C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Non-Camouflage\GT\GT_Camouflage"
    NUM_EPOCHS = 50

    train_images1 = load_image(path=train_images1)
    test_images1 = load_image(path=test_images1)
    train_images2 = load_image(path=train_images2)
    test_images2 = load_image(path=test_images2)
    train_masks1 = np.asarray(load_masks(path=masks1 + r"\Train_mask"))
    tests_masks1 = np.asarray(load_masks(path=masks1 + r"\Test_mask"))
    train_masks_no_camouflage = np.asarray(load_masks(path=no_camouflage_mask + r"\Train"))
    tests_masks_no_camouflage = np.asarray(load_masks(path=no_camouflage_mask + r"\Test"))

    train_images = np.concatenate((train_images1, train_images2))
    train_masks = np.concatenate((train_masks1, train_masks_no_camouflage))

    test_images = np.concatenate((test_images1, test_images2))
    tests_mask = np.concatenate((tests_masks1, tests_masks_no_camouflage))

    preprocessed_train_image, preprocessed_train_masks = preprocess_data(train_images,
                                                                         train_masks)

    preprocessed_tests_image, preprocessed_tests_masks = preprocess_data(test_images, tests_mask)

    BATCH_SIZE = 1
    STEPS_PER_EPOCH = len(train_masks) // BATCH_SIZE
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = len(test_images) // BATCH_SIZE

    ###########learn_classification_stream###################
    classification_stream = classification_stream()
    classification_stream.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                           amsgrad=False),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics="accuracy")

    classification_stream.fit(train_images, train_masks,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=(test_images, tests_mask),
                              shuffle=True)
    classification_stream.save("classification_stream.h5")
