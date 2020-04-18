import os
import glob
import cv2

import datetime
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorboard import notebook
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import interactive
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
tf.keras.backend.clear_session()
from scipy.io import loadmat
from skimage.transform import downscale_local_mean
from math import trunc
from statsmodels import api as sm
from scipy import stats

path = r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q4\UTKFace\*.jpg'
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
X = np.zeros((1586,200,200,3))    
for i in range(1586):
    X[i] = temp_X[i]
    
Y = np.array([d['age'] for d in data[:]])

print('The shape of each picture is :',np.shape(X))

X = X.astype('float32') / 255

fig = plt.figure(figsize=[10, 10])
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    ax.imshow(X[i,:,:,:])
        
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=7)    
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.3, random_state=1)

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 5]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_OPTIMIZER, HP_KERNEL_SIZE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
    inputs = keras.Input(shape=(200,200,3, ), name='train_X')
    x = layers.Conv2D(hparams[HP_NUM_UNITS], (5,5), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(hparams[HP_NUM_UNITS]*2, (5,5), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(hparams[HP_NUM_UNITS]*2, activation='relu')(x)
    x = layers.Dense(hparams[HP_NUM_UNITS], activation='relu')(x)

    # the output
    outputs = layers.Dense(117)(x)

    # build the model, and print a summary
    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='cnn_model')
    model_cnn.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    model_cnn.fit(train_X, train_Y, epochs=20, ) # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model_cnn.evaluate(test_X, test_Y)
    return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for kernel_size in (HP_KERNEL_SIZE.domain.values):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_KERNEL_SIZE: kernel_size,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1

"""# Part 1
Train a model from scratch, using no data augmentation, on the provided abridged
SVHN training set.
"""

def build_model():
    # our model, input in an image shape
    inputs = keras.Input(shape=(200,200,3, ), name='train_X')
    x = layers.Conv2D(64, (5,5), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (5,5), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # the output
    outputs = layers.Dense(117)(x)

    # build the model, and print a summary
    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='cnn_model')
    
    return model_cnn

model_cnn = build_model()
model_cnn.summary()

model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model_cnn.fit(train_X, train_Y,
                        epochs=20,
                        validation_data=(test_X, test_Y), verbose=False)

def eval_model(model, history, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)
    gt_idx = tf.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)


    ax = fig.add_subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5,1])
    plt.legend(loc='lower right')

eval_model(model_cnn, history, test_X, test_Y)
    
    
    