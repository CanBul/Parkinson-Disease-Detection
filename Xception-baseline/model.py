from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import numpy as np

import sys
sys.path.append('..')
from tqdm.keras import TqdmCallback
from readImages import read_images, splitIDsTrainTest

pathList = ['/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/ReadTextPDimages/',
            '/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/ReadTextHCimages/',
            #'/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/SpontaneousDialoguePDimages/',
            #'/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/SpontaneousDialogueHCimages/'
            ]

n_split=6
train_folds, test_folds = splitIDsTrainTest(pathList, n_split=n_split)

accuracies = []
for fold in range(n_split):
  X_train, X_test, y_train, y_test = read_images(
      pathList, train_folds[fold], test_folds[fold])

  base_model = keras.applications.Xception(
      weights='imagenet',  # Load weights pre-trained on ImageNet.
      input_shape=(221, 223, 3),
      include_top=False)

  base_model.trainable = False
  inputs = keras.Input(shape=(221, 223, 3))

  x = base_model(inputs, training=False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  outputs = keras.layers.Dense(1)(x)
  model = keras.Model(inputs, outputs)

  early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                            patience = 5, mode = 'min', verbose = 1,
                            restore_best_weights = True)
  reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                patience = 4, min_delta = 0.001, 
                                mode = 'min', verbose = 1)
  callbacks = [early_stop, reduce_lr]

  model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=keras.losses.BinaryCrossentropy(
      from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

  print("---------Training for fold "+str(fold+1)+"------------")
  model.fit(x=X_train, y=y_train, batch_size=32, epochs=40, verbose=1)#, callbacks=callbacks)
  y_pred = np.argmax(model.predict(X_test), axis=1)

  #accuracy
  acc=accuracy_score(y_test, y_pred)
  print("---------Fold "+str(fold+1)+" Accuracy "+str(acc), " ------------\n")
  accuracies.append(acc)

print("Overall Accuracy: ", np.mean(accuracies))
