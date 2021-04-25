import keras
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.models import load_model
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from readImages import read_images, splitIDsTrainTest

pathList = ['/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/ReadTextPDImages/',
            '/content/Parkinson-Disease-Detection/data/KGL/data3sec/BWimages/ReadTextHCImages/',
            '/content/Parkinson-Disease-Detection/data/IT/data3sec/BWimages/PDImages/',
            '/content/Parkinson-Disease-Detection/data/IT/data3sec/BWimages/HCImages/',
            ]

n_split=4
KC_train_folds, KC_test_folds = splitIDsTrainTest(pathList[:2], n_split=n_split)
IT_train_folds, IT_test_folds = splitIDsTrainTest(pathList[2:], n_split=n_split)

accuracies = []
for fold in range(n_split):
  fold=1
  KC_X_train, X_test, KC_y_train, y_test, fnames = read_images(
      pathList[:2], KC_train_folds[fold], KC_test_folds[fold])

  IT_X_train, IT_X_test, IT_y_train, IT_y_test,_ = read_images(
      pathList[2:], IT_train_folds[fold], IT_test_folds[fold])

  X_train = np.concatenate((IT_X_train, IT_X_test, KC_X_train), axis=0)#np.stack((IT_X_train, IT_X_test, KC_X_train), axis=0)
  y_train = np.hstack((IT_y_train, IT_y_test, KC_y_train)).ravel()

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
  mc = ModelCheckpoint('best_model.h5', monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
  callbacks = [mc]

  print("---------Training for fold "+str(fold+1)+"------------")
  model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), batch_size=32, epochs=60, verbose=1, callbacks=callbacks)
  # load the saved model
  saved_model = load_model('best_model.h5')
  # evaluate the model
  _, acc = saved_model.evaluate(X_test, y_test, verbose=0)
  y_pred = (saved_model.predict(X_test) > 0.5).astype("int32")
  foldInfo=pd.DataFrame()
  foldInfo['ids'] = fnames
  foldInfo['ids2'] = fnames
  foldInfo['Label'] = y_test
  foldInfo['preds'] =  y_pred
  foldInfoUnq=foldInfo.groupby(['ids']).agg({'Label': 'first','ids2':'count', 'preds': 'sum'}).reset_index()
  foldInfoUnq.to_excel('fold'+str(fold)+'.xlsx', index=False)
  
  print("---------Fold "+str(fold+1)+" Accuracy "+str(acc), " ------------\n")
  accuracies.append(acc)

print("Overall Accuracy: ", np.mean(accuracies))