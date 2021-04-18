import keras
import talos
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score
import numpy as np

import sys
sys.path.append('..')
from readImages import read_images, splitIDsTrainTest, read_imagesWOsplit

pathList = ['/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/ReadTextPDImages/',
            '/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/ReadTextHCImages/',
            ]

x_data, y_data = read_imagesWOsplit(pathList)

p = {'lr':[0.001, 0.01, 0.05, 0.1],
      'batch_size': [4,8,16,32,64],
      'epochs': [20, 30, 40, 60, 80]}

def cnn_model(x_train, y_train, x_val, y_val, params):
  model = Sequential()

  model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                  activation ='relu', input_shape = (128,130,3)))
  model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                  activation ='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))


  model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                  activation ='relu'))
  model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                  activation ='relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.25))


  model.add(Flatten())
  model.add(Dense(256, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(1))

  optimizer = RMSprop(lr=params['lr'], rho=0.9, epsilon=1e-08, decay=0.0)
  model.compile(optimizer = optimizer, loss=keras.losses.BinaryCrossentropy(
      from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

  out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_data=(x_val, y_val),
                    verbose=1)

  return out, model

scan_object = talos.Scan(x_data, y_data, model=cnn_model, val_split=0.2,params=p, experiment_name='iris', fraction_limit=.1)