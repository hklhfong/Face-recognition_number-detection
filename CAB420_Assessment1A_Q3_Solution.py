import os
import datetime
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorboard import notebook
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.keras.backend.clear_session()
from scipy.io import loadmat

# #Load training set 
test = loadmat(r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q3\q3_test')
test_X = test["test_X"]
test_Y = test["test_Y"]
train = loadmat(r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q3\q3_train')
train_X = train["train_X"]
train_Y = train["train_Y"]


print('input shape before = {}'.format(train_X.shape))
# also plotting it to make sure we reshaped everything correctly
new_train_X = np.swapaxes(train_X, 3, 0)
new_test_X = np.swapaxes(test_X, 3, 0)

print('input first swap = {}'.format(new_train_X.shape))
new_train_X = np.swapaxes(new_train_X, 3, 1)
new_test_X = np.swapaxes(new_test_X, 3, 1)
print('input first swap = {}'.format(new_train_X.shape))
new_train_X = np.swapaxes(new_train_X, 3, 2)
new_test_X = np.swapaxes(new_test_X, 3, 2)
print('input final = {}'.format(new_train_X.shape))
# also plotting it to make sure we reshaped everything correctly
fig = plt.figure(figsize=[10, 10])
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    # If these were full colour RGB images, this would not be needed
    ax.imshow(train_X[:,:,:, i])
plt.show()

testingvalue = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
print(np.shape(train_X))
print(train_X[:,:,:, i])
print(np.shape(train_Y))


reshaped_new_train_X = new_train_X.reshape(1000, 3072).astype('float32') / 255
reshaped_new_test_X = new_test_X.reshape(10000, 3072).astype('float32') / 255

# create an input, we need to specify the shape of the input, in this case it's a vectorised images with a 784 in length
inputs = keras.Input(shape=(3072,), name='reshaped_new_train_X')
# first layer, a dense layer with 64 units, and a relu activation. This layer recieves the 'inputs' layer as it's input
x = layers.Dense(64, activation='relu')(inputs)
# second layer, another dense layer, this layer recieves the output of the previous layer, 'x', as it's input
x = layers.Dense(64, activation='relu')(x)
# output layer, length 10 units. This layer recieves the output of the previous layer, 'x', as it's input
outputs = layers.Dense(11)(x)

# create the model, the model is a collection of inputs and outputs, in our case there is one of each
model = keras.Model(inputs=inputs, outputs=outputs, name='train_model')
# print a summary of the model
model.summary()


model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = model.fit(reshaped_new_train_X, train_Y,
                    batch_size=128,
                    validation_split=0.2)



def eval_model(model, test_X, test_Y):
    test_scores = model.evaluate(test_X, test_Y, verbose=1)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(test_X);
    indexes = tf.argmax(pred, axis=1)

    cm = confusion_matrix(test_Y, indexes)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(1,11))
    c.plot(ax = ax)

eval_model(model, reshaped_new_test_X, test_Y)




#Second Model (optional)


# our input now has a different shape, 28x28x1, as we have 28x28 single channel images
inputs = keras.Input(shape=(32, 32, 3, ), name='new_train_X')
# rather than use a fully connected layer, we'll use 2D convolutional layers, 8 filters, 3x3 size kernels
x = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu')(inputs)
# 2x2 max pooling, this will downsample the image by a factor of two
x = layers.MaxPool2D(pool_size=(2, 2))(x)
# more convolution, 16 filters, followed by max poool
x = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
# final convolution, 32 filters
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
# a flatten layer. Matlab does a flatten automatically, here we need to explicitly do this. Basically we're telling
# keras to make the current network state into a 1D shape so we can pass it into a fully connected layer
x = layers.Flatten()(x)
# a single fully connected layer, 64 inputs
x = layers.Dense(64, activation='relu')(x)
# and now our output, same as last time
outputs = layers.Dense(11)(x)

# build the model, and print the summary
model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='train_mnist_cnn_model')
model_cnn.summary()


model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
history = model_cnn.fit(new_train_X, train_Y,
                        batch_size=64,
                        epochs=20,
                        validation_split=0.2)

eval_model(model_cnn, new_test_X, test_Y)


#Third model (Main Model)



# our model, input again, still in an image shape
inputs = keras.Input(shape=(32, 32, 3, ), name='new_train_X')
# run pairs of conv layers, all 3s3 kernels
x = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
x = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
# batch normalisation, before the non-linearity
x = layers.BatchNormalization()(x)

# max pooling, 2x2, which will downsample the image
x = layers.MaxPool2D(pool_size=(2, 2))(x)
# rinse and repeat with 2D convs, batch norm, dropout and max pool
x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.1)(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
# final conv2d, batch norm and spatial dropout
x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)

x = layers.MaxPool2D(pool_size=(2, 2))(x)


x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.1)(x)

# flatten layer
x = layers.Flatten()(x)
# we'll use a couple of dense layers here, mainly so that we can show what another dropout layer looks like 
# in the middle
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation='relu')(x)
# the output
outputs = layers.Dense(11)(x)

# build the model, and print a summary
model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='train_mnist_cnn_model')
model_cnn.summary()



model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
              metrics=['accuracy'])
history = model_cnn.fit(new_train_X, train_Y,
                        batch_size=64,
                        epochs=20,
                        validation_split=0.2)

eval_model(model_cnn, new_test_X, test_Y)





# now with Augementation

datagen = ImageDataGenerator(
                            # rotate between -5, +5 degrees
                            rotation_range=5,
                            # horiziontal shift by +/- 5% of the image width
                            width_shift_range=0.05,
                            # vertical shift by +/- 5% of the image width
                            height_shift_range=0.05,
                            # range for zooming
                            zoom_range=0.1,
                            # allow horizontal flips of data
                            horizontal_flip=True,
                            # what value to place in new pixels, given the nature of our data (clothes on a black backround)
                            # we'll set this to a constant value of 0
                            fill_mode='constant', cval=0)

def CreateModel():
    # our model, input in an image shape
    inputs = keras.Input(shape=(32, 32, 3, ), name='img')

    # 7x7 conv
    x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=None)(inputs)
    # batch normalisation, before the non-linearity
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # 5x5 conv
    x = layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation=None)(x)
    # batch normalisation, before the non-linearity
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # 5x5 conv
    x = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=None)(x)
    # batch normalisation, before the non-linearity
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # flatten layer
    x = layers.Flatten()(x)

    # dense layer, 256 neurons
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)

    # the output, one neuron for the cost, relu activation becuase the cost must be positive
    outputs = layers.Dense(11, activation='softmax')(x)

    # build the model, and print a summary
    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='kmnist_cnn_model')
    return model_cnn 

# def eval(model_cnn, history, test, test_y):
#     fig = plt.figure(figsize=[20, 6])
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(history.history['loss'], label="Training Loss")
#     ax.plot(history.history['val_loss'], label="Validation Loss")
#     ax.legend()
    
#     fig = plt.figure(figsize=[20, 6])
#     ax = fig.add_subplot(1, 1, 1)
#     w = 0.4
#     pos = np.arange(0, np.shape(test_y)[0], 1)
#     ax.bar(pos-w, test_y[:,0], label="Actual", width=w)
#     pred = model_cnn.predict(test)
#     ax.bar(pos, pred[:,0], label="Predicted", width=w)
#     ax.legend()


# batch = datagen.flow(new_train_X/255, train_Y, batch_size=100)
# fig = plt.figure(figsize=[20, 25])
# for i,img in enumerate(batch[0][0]):
#     ax = fig.add_subplot(10, 10, i + 1)
#     ax.imshow(img[:,:,:])


# model_cnn = CreateModel()
# model_cnn.summary()
# model_cnn.compile(loss='mean_squared_error',
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])
# history = model_cnn.fit(datagen.flow(new_train_X, train_Y, batch_size=16),
#                     steps_per_epoch=450 // 16,
#                     epochs=100,
#                     verbose=False)

# eval_model(model_cnn, new_test_X, test_Y)