# document: https://www.tensorflow.org/beta/tutorials/load_data/images

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
print("version: ", tf.__version__)

img_raw = tf.io.read_file('images/' + filepath)
print(img_raw.dtype)

img = tf.image.decode_jpeg(img_raw, channels = 3)
print(img.dtype)
plt.imshow(img)

img = tf.image.convert_image_dtype(img, tf.float32)
print(img.dtype)
print(img.shape)
img = tf.image.resize(img, [218, 218])

shape = tf.cast(tf.shape(img)[:-1], tf.float32)

############# With Datasent
AUTOTUNE = tf.data.experimental.AUTOTUNE
filepaths = os.listdir('images/')
im_size = 218

def load_and_preprocessing(filepath):
    img_raw = tf.io.read_file('images/' + filepath)
    img = tf.image.decode_jpeg(img_raw, channels = 3)
    img = tf.image.resize(img, [218, 218])
    img /= 255.0
    return img 

# Create a dataset
path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
image_ds = path_ds.map(load_and_preprocessing, num_parallel_calls = AUTOTUNE)


#####################################################################
############# Pretrained Models
import tensorflow as tf
from tf.keras.applications import MobileNetV2

im_size = 218
model = MobileNetV2(input_shape = (im_size, im_size, 3), include_top = False)
model.trainable = False

for layer in model.layers:
    print(layer.name)

# Get the input information
help(tf.keras.applications.mobilenet_v2.preprocess_input)

