# References: https://www.tensorflow.org/beta/tutorials/quickstart/advanced

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def load_dataset(batch_size):
    
    (tr_x, tr_y), (te_x, te_y) = tf.keras.datasets.mnist.load_data()
    
    # tr_x, te_x = tr_x/255.0, te_x/255.0
    tr_x = tf.cast(tr_x, tf.float64)
    te_x = tf.cast(te_x, tf.float64)

    tr_x = tr_x[..., tf.newaxis]
    te_x = te_x[..., tf.newaxis]
    
#     tr_y = tf.one_hot(tr_y, 10)
#     te_y = tf.one_hot(te_y, 10)
    
    print("The dims of the train set: ", tr_x.shape)
    print("The dims of the test set: ", te_x.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((tr_x, tr_y))
    train_ds = train_ds.shuffle(100).batch(batch_size)
    
    test_ds = tf.data.Dataset.from_tensor_slices((te_x, te_y)).batch(batch_size)

    return train_ds, test_ds

##########
def load_and_preprocessing(filepath, im_size):
    img_raw = tf.io.read_file('images/' + filepath)
    img = tf.image.decode_jpeg(img_raw, channels = 3)
    img = tf.image.resize(img, [im_size, im_size])
    img /= 255.0
    return img 
##########

# Training loss 
def compute_loss(labels, preds):
    return tf.keras.losses.SparseCategoricalCrossentropy(labels, preds)
 

class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.conv1 = Conv2D(32, 3, activation='relu')
      self.flatten = Flatten()
      self.d1 = Dense(128, activation='relu')
      self.d2 = Dense(10, activation='softmax')

    def call(self, x):
      x = self.conv1(x)
      x = self.flatten(x)
      x = self.d1(x)
      return self.d2(x)

    
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model.call(images)
#         loss = compute_loss(labels, preds)
        loss = loss_object(labels, preds)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, preds)


@tf.function
def test_step(images, labels):
    preds = model.call(images)
#     loss = compute_loss(labels, preds)
    loss = loss_object(labels, preds)

    test_loss(loss)
    test_accuracy(labels, preds)



if __name__ == "__main__":
    
    epochs = 5
    batch_size = 32

    # Import and parse the dataset
    print("======= Loading dataset..")
    train_ds, test_ds = load_dataset(batch_size)

    im_size = 28
    n_classes = 10
    
    # Modeling & Optimizer
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate = .01)

    # loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')

    print("======= Training..")
    for epoch in range(epochs):
        print("=== Epoch: {}th..".format(epoch+1))
        for tr_x, tr_y in train_ds:
            train_step(tr_x, tr_y)

        for te_x, te_y in test_ds:
            test_step(te_x, te_y)

        template = '==> Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(train_loss.result(), train_accuracy.result()*100,
                              test_loss.result(), test_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    # Plot the loss 
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()