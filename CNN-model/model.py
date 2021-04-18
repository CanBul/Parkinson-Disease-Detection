import keras
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import numpy as np

import sys
sys.path.append('..')
from readImages import read_images, splitIDsTrainTest

pathList = ['/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/ReadTextPDImages/',
            '/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/ReadTextHCImages/',
            #'/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/SpontaneousDialoguePDImages/',
            #'/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/SpontaneousDialogueHCImages/'
            ]

n_split=4
train_folds, test_folds = splitIDsTrainTest(pathList, n_split=n_split)

accuracies = []
for fold in range(n_split):
  X_train, X_test, y_train, y_test = read_images(
      pathList, train_folds[fold], test_folds[fold])

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

  optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  model.compile(optimizer = optimizer, loss=keras.losses.BinaryCrossentropy(
      from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

  early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                            patience = 5, mode = 'min', verbose = 1,
                            restore_best_weights = True)
  reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                patience = 4, min_delta = 0.001, 
                                mode = 'min', verbose = 1)
  callbacks = [early_stop, reduce_lr]

  print("---------Training for fold "+str(fold+1)+"------------")
  model.fit(x=X_train, y=y_train, batch_size=32, epochs=60, verbose=1)#, callbacks=callbacks)
  y_pred = (model.predict(X_test) > 0.5).astype("int32")
  #accuracy
  acc=accuracy_score(y_test, y_pred)
  print("---------Fold "+str(fold+1)+" Accuracy "+str(acc), " ------------\n")
  accuracies.append(acc)

print("Overall Accuracy: ", np.mean(accuracies))