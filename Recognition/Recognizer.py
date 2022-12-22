import numpy as np

from ANet.NetworkUtils import load_networks, display, process_to_mask
from ANet.Segmentation_stream import SIZE_OF_IMAGE, load_masks
from ImageManager import load_images_jpg, resize_image

if __name__ == '__main__':
    segmentation_load, classification_load = load_networks()
    path_of_test_image = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\network_tests\Image'
    path_of_test_masks = r'C:\Users\Admin\PycharmProjects\FCM\FCM\ANet\network_tests\Mask'
    images = load_images_jpg(path_of_test_image, 2)
    masks = load_masks(path=path_of_test_masks)

    # image = resize_image(images[0], (SIZE_OF_IMAGE, SIZE_OF_IMAGE))
    # result_of_segmentation = segmentation_load.predict(image)

    # result_of_classification = classification_load.predict(image)

    for i in range(len(path_of_test_masks)):
        resized_image = resize_image(images[i], (SIZE_OF_IMAGE, SIZE_OF_IMAGE))
        reshaped_image = np.reshape(resized_image, [1, SIZE_OF_IMAGE, SIZE_OF_IMAGE, 3])
        display([reshaped_image[0], masks[i], segmentation_load.predict(reshaped_image)[0]
                 ])
    # combine(result_of_classification, result_of_segmentation)
