#################################################
############# Artificial Neural Network #############
import keras
from keras.models import Sequential
from Keras.layers import Dense

# initialize the ANN
clas = Sequential()

# create hidden layers
clas.add(Dense(input_dim = 11, output_dim = 6, init = 'uniform', activation = 'relu'))
clas.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
clas.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))   # softmax
# compile: stochastic gradient descent
clas.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

clas.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred = clas.predict(X_test)


############# Convolutional Neural Networks #############
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

clas = Sequential()

# convolutional layer (^?)
clas.add(Convolution2D(32, 3, 3), input_shape = (64, 64, 3), activation = 'relu')
# pooling layer
clas.add(MaxPooling2D(pool_size = (2, 2)))

# adding new convolutional layer
# clas.add(Convolution2D(64, 3, 3), activation = 'relu')
# clas.add(MaxPooling2D(pool_size = (2, 2)))

# flattening
clas.add(Flatten())
# full connection (^?)
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
validation_generator = test_datagen.flow_from_directory('data/validation',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
clas.fit_generator(train_generator,
                   steps_per_epoch=2000,
                   epochs=50,
                   validation_data = validation_generator,
                   validation_steps=800)


#################################################
#############
