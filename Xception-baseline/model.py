from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import numpy as np

import sys
sys.path.append('..')
from tqdm.keras import TqdmCallback
from readImages import read_images, splitIDsTrainTest

pathList = ['/content/Parkinson-Disease-Detection/data/KGL/data3sec/ReadTextPDImages/',
            '/content/Parkinson-Disease-Detection/data/KGL/data3sec/ReadTextHCImages/',
            #'/content/Parkinson-Disease-Detection/data/KGL/data3sec/SpontaneousDialoguePDImages/',
            #'/content/Parkinson-Disease-Detection/data/KGL/data3sec/SpontaneousDialogueHCImages/'
            ]

n_split=6
train_folds, test_folds = splitIDsTrainTest(pathList, n_split=n_split)

accuracies = []
for fold in range(n_split):
  X_train, X_test, y_train, y_test = read_images(
      pathList, train_folds[fold], test_folds[fold])

  base_model = keras.applications.Xception(
      weights='imagenet',  # Load weights pre-trained on ImageNet.
      input_shape=(128, 130, 3),
      include_top=False)

  base_model.trainable = False
  inputs = keras.Input(shape=(128, 130, 3))

  x = base_model(inputs, training=False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  outputs = keras.layers.Dense(1)(x)
  model = keras.Model(inputs, outputs)

  model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(
      from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])
  print("---------Fold "+str(fold+1)+"------------")
  model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=60, verbose=1)
  y_pred = np.argmax(model.predict(X_test), axis=1)

  #accuracy
  acc=accuracy_score(y_test, y_pred)
  print("---------Fold "+str(fold+1)+" Accuracy "+str(acc), " ------------\n")
  accuracies.append(acc)

print("Overall Accuracy: ", np.mean(accuracies))
