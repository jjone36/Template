#################################################
############# Tensorflow Basic
import tensorflow as tf

a = tf.constant(2)      # create tensors
b = tf.constant(10)
c = tf.multiply(a, b)   # write operations between the tenseors

sess = tf.Session()     # create a Session
print(sess.run(c))      # run the session and initialize the variables


y_hat = tf.constant(36, name = 'y_hat')
y = tf.constant(39, name = 'y')
loss = tf.Variable((y - y_hat)**2, name = 'loss')

init = tf.global_variables_initializer()

with tf.Session() as sessi:
    sess.run(init)
    print(sess.run(loss))

# placeholders whose values you will specify only later
x = tf.placeholder(tf.int64, name = 'x')         # create placeholders
sigmoid = tf.sigmoid()                           # specify the computation graph
print(sess.run(sigmoid, feed_dict = {x : 3}))    # create and run the session using feed dictionary
sess.close()

#####################################################
############# Keras Basic
from keras.layers import Input, Dense
input_tensor = Input(shape = (1,))
output_tensor = Dense(1)(input_tensor)

from keras.models import Model
model = Model(inputs = input_tensor, outputs = output_tensor)
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
model.summary()
model.get_weights()

# Plot the model
plot_model(model, to_file = 'model.png')
import matplotlib.pyplot as plt
img = plt.imread('model.png')
plt.imshow(img)
plt.show()

# Fit the model
model.fit(X_train, y_train, epochs = 10, batch_size = 64, validation_split = .2, verbose = True)
model.evaluate(x = X_dev, y = y_dev)

# Prediction
model.predict(X_test)

########################## More layers
# Embedding layer
from keras.layers import Embedding, Flatten
embed_layer = Embedding(input_dim = m, input_length = 1, output_dim = 1, name = '')
embed_tensor = embed_layer(input_tensor)
flatten_tensor = Flatten()(embed_tensor)
model = Model(inputs = input_tensor, outputs = flatten_tensor)

# Merging layers
in_tensor_1 = Input(shape = (1,))
in_tensor_2 = Input(shape = (1,))
out_tensor = Add()([in_tensor_1, in_tensor_2])   # Subtract(), Concatenate()
model = Model(inputs = [in_tensor_1, in_tensor_2], outputs = out_tensor)

#####################################################
############# Artificial Neural Network #############
import keras
from keras.models import Sequential
from Keras.layers import Dense

# initialize the model
model = Sequential()

# create hidden layers
model.add(Dense(input_dim = 11, output_dim = 6, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))  # softmax
# compile: stochastic gradient descent
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# -> 'mean_squared_error' / 'binary_crossentropy'

# early_stopper
from keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(patience = 5)

model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, callbacks = [early_stopper])   # validation_split
y_pred = model.predict(X_test)

# plot the error
history = training.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.show()


# Fit the model using the checkpoint
model.fit(X_train, y_train, epochs = 10, validation_split = .2, callbacks = callbacks_list)

model.load_weights('weights.hdf5')
model.predict_classes(X_test)

# model save
model.save('model_file.h5')
from keras.model import load_model
my_model = load_model('model_file.h5')

########################## Regularization
# Dropout
from keras.layers import Dropout
model.add(Dropout(.25))

# Batch normalization
from keras.layers import BatchNormalization
model.add(BatchNormalization())

########################## grid search
from keras.optimizers import SGD
lr_to_test = [.000001, .01, 1]
for lr in lr_to_test:
    my_optimizer = SGD(lr = lr)
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    model.fit(predictors, target)


##############################################################
############# Convolutional Neural Networks #############
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization

# importing datasets
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

im_height
im_weight
im_size = (im_height, im_width)
batch_size

# initialize the model
model = Sequential()
# convolutional layer
model.add(Conv2D(32, kernel_size = 3, inpurt_shape = (im_width, im_height, 3), activation = 'relu', padding = 'same', strides = 5))
          # Convolution2D(10, 3, 3)
# Normalization
model.add(BatchNormalization())

# adding new convolutional layer
# model.add(Convolution2D(64, 3, 3), activation = 'relu')

# pooling layer
model.add(MaxPool2D(2))
# into one big array
model.add(Flatten())

# full connection layer
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 5, activation = 'softmax'))

# building and optimizatiion
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

# Fitting
model.fit(X_train, y_train, epochs = 10, batch_size = 64, validation_split = .2, verbose = True)
model.evaluate(x = X_dev, y = y_dev)

########################## image data preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)
train_generator = train_gen.flow_from_directory(dir,
                                                target_size=(64, 64),
                                                batch_size=32,
                                                models_mode='binary')

val_gen = ImageDataGenerator(rescale = 1./255)
val_generator = val_gen.flow_from_dataframe(val_df, dir,
                                            x_col='filename',
                                            target_size= im_size,
                                            class_mode='binary',
                                            batch_size= batch_size)

history = model.fit_generator(train_generator,
                    epochs = 50,
                    validation_data = val_generator,
                    validation_steps = val_df.shape[0] // batch_size,
                    steps_per_epoch= train_df.shape[0] // batch_size,
                    callbacks = callbacks)

# model_saving
model.save('model_file.h5')

# prediction
# Augmentation with test set
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(test_df, dir,
                                              x_col='filename',
                                              y_col='category',
                                              target_size = im_size,
                                              batch_size = batch_size)

predict = model.predict_generator(test_generator, steps = n_samples // batch_size))
test_df['prob'] = predict
test_df['category'] = np.where(test_df['prob'] > .5, 1, 0)

########################## callbacks
# Learning rate reduction
from keras.callbacks import ReduceLROnPlateau
learning_rate_reducer = ReduceLROnPlateau(monitor='val_acc',
                                          patience=2,
                                          verbose=1,
                                          factor=0.5,
                                          min_lr=0.00001)

# Monitoring validation loss
from keras.callbacks import ModelCheckpoint
checker = ModelCheckpoint('weights.hdf5', monitor = 'val_loss', save_best_only = True)

callbacks = [early_stopper, learning_rate_reducer, checker]


########################## Get the weights
conv_1 = model.layers[0]
weights_1 = conv_1.get_weights()
len(weights_1)

kernels_1 = weights_1[0]
kernels_1.shape   # -> (f, f, Nc', Nc)

kernels_1 = kernels_1[:, :, 0, 0]
plt.imshow(kernels_1)
plt.show()

out = convolution(X_test[1, :, :, 0], kernels_1)
plt.imshow(out)
plt.show()
