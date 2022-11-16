import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

from ANet.NetworkUtils import preprocess_data, display, create_mask
from ANet.Segmentation_Stream import load_image, test_images1, masks1, load_masks, train_images1, unet_model, \
    train_images2, test_images2, masks2

tf.get_logger().setLevel('WARNING')
tf.config.list_physical_devices('GPU')

# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
if __name__ == '__main__':
    # Received a label value of 255 which is outside the valid range of [0, 3).
    np_config.enable_numpy_behavior()
    NUM_EPOCHS = 25

    train_images1, train_numbers1 = load_image(path=train_images1)
    test_images1, tests_numbers1 = load_image(path=test_images1)
    train_images2, train_numbers2 = load_image(path=train_images2)
    test_images2, tests_numbers2 = load_image(path=test_images2)
    train_masks1 = load_masks(path=masks1, numbers=train_numbers1)
    tests_masks1 = load_masks(path=masks1, numbers=tests_numbers1)
    train_masks2 = load_masks(path=masks2, numbers=train_numbers2)
    tests_masks2 = load_masks(path=masks2, numbers=tests_numbers2)

    preprocessed_train_image, preprocessed_train_masks = preprocess_data(np.concatenate((train_images1, train_images2)),
                                                                         np.concatenate((train_masks1, train_masks2)))
    preprocessed_tests_image, preprocessed_tests_masks = preprocess_data(np.concatenate((test_images1, test_images2)),
                                                                         np.concatenate((tests_masks1, tests_masks2)))

    train_dataset = tf.data.Dataset.from_tensor_slices((preprocessed_train_image, preprocessed_train_masks))
    test_dataset = tf.data.Dataset.from_tensor_slices((preprocessed_tests_image, preprocessed_tests_masks))

    TRAIN_LENGTH = len(train_images1)
    BATCH_SIZE = 16
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH
    TEST_LENGTH = len(tests_masks1)
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE
    train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    validation_batches = test_dataset.batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)
    ############learn_classification_stream###################
    # classification_stream = classification_stream()
    # classification_stream.compile(optimizer=tf.keras.optimizers.Adam(),
    #                               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #                               metrics="accuracy")
    # classification_stream.fit(x=train_batches,
    #                           epochs=NUM_EPOCHS,
    #                           steps_per_epoch=STEPS_PER_EPOCH,
    #                           validation_steps=VALIDATION_STEPS,
    #                           validation_data=test_batches)
    # classification_stream.save("classification_stream_v1.h5")
    #############learn_segmentation_stream####################

    unet_model = unet_model()

    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss="sparse_categorical_crossentropy",
                       metrics="accuracy")

    unet_model.fit(x=train_batches,
                   epochs=NUM_EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH,
                   validation_steps=VALIDATION_STEPS,
                   validation_data=validation_batches)
    unet_model.save("test_neural_networkV9.h5")
    ###############load_segmentation_stream############
    ###################TEST################
    # loaded_model = tf.keras.models.load_model("test_neural_networkV8.h5")
    # loaded_model.compile(loss='binary_crossentropy',
    #                      optimizer='rmsprop',
    #                      metrics=['accuracy'])
    # path_of_image = 'C:\\Users\\Admin\\PycharmProjects\\FCM\\FCM\\ANet\\test_data\\CAMO-COCO-V.1.0\\CAMO-COCO-V.1.0-CVIU2019\\Camouflage\\Images\\Test\\camourflage_00002.jpg'
    # path_of_mask = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\test_data\CAMO-COCO-V.1.0\CAMO-COCO-V.1.0-CVIU2019\Camouflage\GT\camourflage_00002.png'
    # img = cv2.imread(path_of_image)
    # mask = cv2.imread(path_of_mask)
    # image = cv2.resize(img, (192, 192))
    # image2 = np.reshape(image, [1, 192, 192, 3])
    # classes = loaded_model.predict(image2)

    display([image, cv2.resize(mask, (192, 192)), create_mask(classes)])
    # cv2.imshow("images", classes.reshape(64, 64, 3))
    # cv2.waitKey()
    #
    # output_image = cv2.resize(classes.reshape(64, 64, 3), (width, height))
    # Verti = np.concatenate((img, output_image), axis=0)
    #
    # cv2.imshow("images", output_image)
    # cv2.waitKey()

    # cv2.imshow("images", cv2.resize(classes.reshape(64, 64, 3), (width, height), interpolation=cv2.INTER_AREA))
    # cv2.waitKey()
