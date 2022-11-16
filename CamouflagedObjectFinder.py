# from matplotlib import pyplot as plt
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import Dropout
# from tensorflow.python.keras.layers import Flatten
# from tensorflow.python.keras.layers import MaxPool2D
# from tensorflow.python.keras.layers import ReLU
# from tensorflow.python.keras.layers import Softmax
# from tensorflow.python.keras.models import Sequential
# import tensorflow_datasets as tfds
#
# # https://github.com/tensorflow/examples
# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
#
# TRAIN_LENGTH = info.splits['train'].num_examples
# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#
#
# def classification_stream(image, dim):
#     # defining model
#     X_train = 100
#     Y_train = 100
#     model = Sequential()
#     model.add(tf.keras.Input(shape=(16,)))
#     model.add(Dense(input_dim=4096, activation="relu"))
#     model.add(ReLU(input_dim=2048, activation="dropout"))
#     model.add(Dropout(input_dim=2048, activation="dense"))
#     model.add(Dense(input_dim=2048, activation="relu"))
#     model.add(ReLU(input_dim=2048, activation="dropout"))
#     model.add(Dense(input_dim=2, activation="dense"))
#     model.add(Softmax(input_dim=2, activation="dense"))
#
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # fitting the model
#     model.fit(X_train, Y_train, epochs=10)
#
#
# # https://www.analyticsvidhya.com/blog/2021/08/beginners-guide-to-convolutional-neural-network-with-implementation-in-python/
#
# # def image_segmentation():
# # Segmentacja https://www.tensorflow.org/tutorials/images/segmentation
# def normalize(input_image, input_mask):
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     input_mask -= 1
#     return input_image, input_mask
#
#
# class Augment(tf.keras.layers.Layer):
#     def __init__(self, seed=42):
#         super().__init__()
#         # both use the same seed, so they'll make the same random changes.
#         self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
#         self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
#
#     def call(self, inputs, labels):
#         inputs = self.augment_inputs(inputs)
#         labels = self.augment_labels(labels)
#         return inputs, labels
#
#
# def display(display_list):
#     plt.figure(figsize=(15, 15))
#
#     title = ['Input Image', 'True Mask', 'Predicted Mask']
#
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
#         plt.axis('off')
#     plt.show()
#
#
# train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
#
# train_batches = (
#     train_images
#     .cache()
#     .shuffle(BUFFER_SIZE)
#     .batch(BATCH_SIZE)
#     .repeat()
#     .map(Augment())
#     .prefetch(buffer_size=tf.data.AUTOTUNE))
#
# test_batches = test_images.batch(BATCH_SIZE)
#
# for images, masks in train_batches.take(2):
#   sample_image, sample_mask = images[0], masks[0]
#   display([sample_image, sample_mask])