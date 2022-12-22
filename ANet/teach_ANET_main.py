import random

import numpy as np
import tensorflow as tf

from ANet.Classification_Stream import learn_classification_stream
from ANet.Segmentation_stream import learn_segmentation_model

tf.get_logger().setLevel('WARNING')
tf.config.list_physical_devices('GPU')

# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
if __name__ == '__main__':
    learn_segmentation_model()
    # learn_classification_stream()
