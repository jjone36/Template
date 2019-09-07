import numpy as np
import tensorflow as tf
from tensorflow import keras
print("GPU availability: ", tf.test.is_gpu_available())

# tensors like numpy! 
print(tf.add([1, 2], [3, 4]))
print(tf.square(3))
print(tf.matmul([[2]], [[2, 3]]))

a = tf.constant(3, shape = (2, 2))
b = tf.ones_like(a)
b.shape
c = tf.fill([2, 2], 3)
c.numpy()
tf.reshape(c, [4, 1])

# basic operation 
tf.multiply(a, b)   # element-wise multiplication
tf.matmul(a, b)

a = tf.constant([[1, 2, 3], 
                [2, 3, 4]])
a.numpy()
print(tf.reduce_sum(a))
print(tf.reduce_sum(a, 1))

# gradients
x = tf.Variable(1, trainable = True, dtype = tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x, xde)

g = tape.gradient(y, x)
g.numpy()

#################################################
# Load the dataset
def load_dataset(batch_size):
    (tr_x, tr_y), (te_x, te_y) = tf.keras.datasets.mnist.load_data()
    print("The dims of the train set: ", tr_x.shape)
    print("The dims of the test set: ", te_x.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((tr_x, tr_y))
    train_ds = train_ds.shuffle(100).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((te_x, te_y)).batch(batch_size)
    return train_ds, test_ds


def load_and_preprocessing(filepath, im_size):
    img_raw = tf.io.read_file('images/' + filepath)
    img = tf.image.decode_jpeg(img_raw, channels = 3)
    img = tf.image.resize(img, [im_size, im_size])
    img /= 255.0
    return img 

path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
image_ds = path_ds.map(load_and_preprocessing, num_parallel_calls = AUTOTUNE)



# Pretrained Models
import tensorflow as tf
from tf.keras.applications import MobileNetV2

model = MobileNetV2(input_shape = (im_size, im_size, 3), include_top = False)
model.trainable = False

for layer in model.layers:
    print(layer.name)

# Get the input information
help(tf.keras.applications.mobilenet_v2.preprocess_input)
