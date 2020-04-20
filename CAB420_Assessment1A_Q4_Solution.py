import os
import glob
import cv2

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
tf.keras.backend.clear_session()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

path = r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q4\UTKFace\*'
files = glob.glob(path)

data = []
for f in files:
    d = {}
    head, tail = os.path.split(f)
    parts = tail.split('_')
    if (len(parts) == 4):
        d['age'] = int(parts[0])
        d['gender'] = int(parts[1])
        d['race'] = int(parts[2])
        d['image'] = cv2.imread(f)
        data.append(d)
    else:
        print('Could not load: ' + f + '! Incorrectly formatted filename')

temp_X = np.array([d['image'] for d in data[:]])
print('The shape of temp_X is :',np.shape(temp_X))
X = np.zeros((23705,32,32,3))    
for i in range(23705):
    try:
        X[i] = cv2.resize(temp_X[i], (32, 32)) 
    except IndexError as e:
        print('Invalid frame!')
        continue
    
Y = np.array([d['age'] for d in data[:23705]])
X = X.astype('float32') / 255

fig = plt.figure(figsize=[10, 10])
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    ax.imshow(X[i,:,:,:])

print('The shape of pictures are :',np.shape(X))
print('The age set shape is :',np.shape(Y))
        
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15, random_state=7)    
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.15, random_state=1)

"""# Part 1
Train a model from scratch, using no data augmentation, on the provided abridged
SVHN training set.
"""

def build_model(num_classes):
    # inputs = keras.Input(shape=(200,200,3, ), name='train_X')
    # x = layers.Conv2D(64, (5,5), activation='relu')(inputs)
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Conv2D(128, (5,5), activation='relu')(x)
    # x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Flatten()(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dense(64, activation='relu')(x)

    # # the output
    # outputs = layers.Dense(117)(x)

    inputs = keras.Input(shape=(32, 32, 3, ), name='img')
    x = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(inputs)
# batch normalisation, before the non-linearity
    x = layers.BatchNormalization()(x)

# max pooling, 2x2, which will downsample the image
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
# rinse and repeat with 2D convs, batch norm,and max pool
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool2D(pool_size=(2, 2))(x)
# final conv2d, batch norm 
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

# flatten layer
    x = layers.Flatten()(x)
# we'll use a couple of dense layers here, mainly so that we can show what another dropout layer looks like 
# in the middle
    x = layers.Dense(256, activation='relu')(x)

    x = layers.Dense(128, activation='relu')(x)
# the output
    outputs = layers.Dense(117, activation=None)(x)

    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='kmnist_cnn_model')
    
    return model_cnn

model_cnn = build_model(117)
model_cnn.summary()

model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = model_cnn.fit(train_X, train_Y,
                        epochs=10,
                        validation_data=(val_X, val_Y),verbose=False)

def eval_model(model, history, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)

    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(117))
    c.plot(ax = ax)


    ax = fig.add_subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5,1])
    plt.legend(loc='lower right')

eval_model(model_cnn, history, test_X, test_Y)

def eval_model_no_val(model, history, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)

    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(117))
    c.plot(ax = ax)


    ax = fig.add_subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5,1])
    plt.legend(loc='lower right')

    return test_scores[1]

eval_model(model_cnn, history, test_X, test_Y)

"""# Part 2
Train a model from cross-fold evaluation protocol based on the race annotation.
"""
temp_X_0 = []
temp_X_1 = []
temp_X_2 = []
temp_X_3 = []
temp_X_4 = []
temp_Y_0 = []
temp_Y_1 = []
temp_Y_2 = []
temp_Y_3 = []
temp_Y_4 = []
for d in data[:23705]:
  try:
    if d['race'] == 0:
      temp_X_0.append(cv2.resize(d['image'], (32, 32)))
      temp_Y_0.append(d['age'])
    if d['race'] == 1:
      temp_X_1.append(cv2.resize(d['image'], (32, 32)))
      temp_Y_1.append(d['age'])
    if d['race'] == 2:
      temp_X_2.append(cv2.resize(d['image'], (32, 32)))
      temp_Y_2.append(d['age'])
    if d['race'] == 3:
      temp_X_3.append(cv2.resize(d['image'], (32, 32)))
      temp_Y_3.append(d['age'])
    if d['race'] == 4:
      temp_X_4.append(cv2.resize(d['image'], (32, 32)))
      temp_Y_4.append(d['age'])
  except cv2.error as e:
      print('Invalid frame!')
  cv2.waitKey()
print(np.array(temp_X_0).shape)
print(np.array(temp_X_1).shape)
print(np.array(temp_X_2).shape)
print(np.array(temp_X_3).shape)
print(np.array(temp_X_4).shape)

temp_Y_0 = np.array(temp_Y_0)
temp_Y_1 = np.array(temp_Y_1)
temp_Y_2 = np.array(temp_Y_2)
temp_Y_3 = np.array(temp_Y_3)
temp_Y_4 = np.array(temp_Y_4)


temp_X_0 = np.array(temp_X_0).astype('float32') / 255
temp_X_1 = np.array(temp_X_1).astype('float32') / 255
temp_X_2 = np.array(temp_X_2).astype('float32') / 255
temp_X_3 = np.array(temp_X_3).astype('float32') / 255
temp_X_4 = np.array(temp_X_4).astype('float32') / 255

cross_set_0_train_x = np.concatenate((temp_X_1,temp_X_2,temp_X_3,temp_X_4),axis=0)
cross_set_0_train_y = np.concatenate((temp_Y_1,temp_Y_2,temp_Y_3,temp_Y_4),axis=0)

cross_set_1_train_x = np.concatenate((temp_X_0,temp_X_2,temp_X_3,temp_X_4),axis=0)
cross_set_1_train_y = np.concatenate((temp_Y_0,temp_Y_2,temp_Y_3,temp_Y_4),axis=0)

cross_set_2_train_x = np.concatenate((temp_X_1,temp_X_0,temp_X_3,temp_X_4),axis=0)
cross_set_2_train_y = np.concatenate((temp_Y_1,temp_Y_0,temp_Y_3,temp_Y_4),axis=0)

cross_set_3_train_x = np.concatenate((temp_X_1,temp_X_2,temp_X_0,temp_X_4),axis=0)
cross_set_3_train_y = np.concatenate((temp_Y_1,temp_Y_2,temp_Y_0,temp_Y_4),axis=0)

cross_set_4_train_x = np.concatenate((temp_X_1,temp_X_2,temp_X_3,temp_X_0),axis=0)
cross_set_4_train_y = np.concatenate((temp_Y_1,temp_Y_2,temp_Y_3,temp_Y_0),axis=0)

print(cross_set_0_train_x.shape)
print(cross_set_1_train_x.shape)
print(cross_set_2_train_x.shape)
print(cross_set_3_train_x.shape)
print(cross_set_4_train_x.shape)

history_0 = model_cnn.fit(cross_set_0_train_x, cross_set_0_train_y,
                        epochs=10, 
                        verbose=False)
print('history_0 done')
history_1 = model_cnn.fit(cross_set_1_train_x, cross_set_1_train_y,
                        epochs=10,
                        verbose=False)
print('history_1 done')
history_2 = model_cnn.fit(cross_set_2_train_x, cross_set_2_train_y,
                        epochs=10,
                        verbose=False)
print('history_2 done')
history_3 = model_cnn.fit(cross_set_3_train_x, cross_set_3_train_y,
                        epochs=10,
                        verbose=False)
print('history_3 done')
history_4 = model_cnn.fit(cross_set_4_train_x, cross_set_4_train_y,
                        epochs=10,
                        verbose=False)
print('history_4 done')

scores = []
scores.append(eval_model_no_val(model_cnn, history_0, temp_X_0, temp_Y_0))
scores.append(eval_model_no_val(model_cnn, history_1, temp_X_1, temp_Y_1))
scores.append(eval_model_no_val(model_cnn, history_2, temp_X_2, temp_Y_2))
scores.append(eval_model_no_val(model_cnn, history_3, temp_X_3, temp_Y_3))
scores.append(eval_model_no_val(model_cnn, history_4, temp_X_4, temp_Y_4))


print('Scores from each Iteration: ', scores)
print('Average K-Fold Score :' , np.mean(scores))
