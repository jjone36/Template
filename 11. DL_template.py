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
model.predict()

############# More layers
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
from keras.callbacks import EarlyStopping

# initialize the ANN
clas = Sequential()

# create hidden layers
clas.add(Dense(input_dim = 11, output_dim = 6, init = 'uniform', activation = 'relu'))
clas.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
clas.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))  # softmax
# compile: stochastic gradient descent
clas.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# -> 'mean_squared_error' / 'categorical_crossentropy'

early_stopper = EarlyStopping(patience = 2)

clas.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, callbacks = [early_stopper])   # validation_split
y_pred = clas.predict(X_test)


model.save('model_file.h5')
from keras.model import load_model
my_model = load_model('model_file.h5')


from keras.optimizers import SGD
lr_to_test = [.000001, .01, 1]
for lr in lr_to_test:
    my_optimizer = SGD(lr = lr)
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    model.fit(predictors, target)

############# Convolutional Neural Networks #############
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

clas = Sequential()

# convolutional layer
clas.add(Convolution2D(32, 3, 3), input_shape = (64, 64, 3), activation = 'relu')
# pooling layer
clas.add(MaxPooling2D(pool_size = (2, 2)))

# adding new convolutional layer
# clas.add(Convolution2D(64, 3, 3), activation = 'relu')
# clas.add(MaxPooling2D(pool_size = (2, 2)))

# flattening
clas.add(Flatten())
# full connection
clas.add(Dense(output_dim = 128, activation = 'relu'))
clas.add(Dense(output_dim = 5, activation = 'softmax'))
clas.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# image data preprocessing
'https://keras.io/preprocessing/image/'
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_frgooom_directory('data/validation',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
clas.fit_generator(train_generator,
                   steps_per_epoch=2000,
                   epochs=50,
                   validation_data = validation_generator,
                   validation_steps=800)


#################################################
