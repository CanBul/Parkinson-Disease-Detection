import cv2
import numpy as np
import os
import math
from sklearn.model_selection import StratifiedKFold


def read_imagesWOsplit(pathList):

    x_data = []
    y_data = []

    for i in range(len(pathList)):
        imgPath = pathList[i]

        files = [f for f in os.listdir(imgPath) if f.endswith('.png')]

        for f in files:

            img = cv2.imread(imgPath + f)  # read image

            img = img.astype('float32')  # change its type
            img /= 255  # normalize
            #img = (img - img.mean(axis=(0,1,2), keepdims=True)) /img.std(axis=(0,1,2), keepdims=True)
            x_data.append(img)

            # Make sure your directory name has PD in it for PD patients!!
            if 'PD' in imgPath:
                y_data.append(1)
            else:
                y_data.append(0)

    x_data, y_data = np.array(x_data), np.array(y_data)

    return x_data, y_data


def read_images(pathList, train_files, test_files):

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(pathList)):
        imgPath = pathList[i]

        files = [f for f in os.listdir(imgPath) if f.endswith('.png')]

        for f in files:
            img = cv2.imread(imgPath + f)  # read image
            img = img.astype('float32')  # change its type
            img /= 255  # normalize

            if f.split('_')[1] in train_files:
                x_train.append(img)

                # Make sure your directory name has PD in it for PD patients!!
                if 'PD' in imgPath:
                    y_train.append(1)
                else:
                    y_train.append(0)

            else:
                x_test.append(img)

                # Make sure your directory name has PD in it for PD patients!!
                if 'PD' in imgPath:
                    y_test.append(1)
                else:
                    y_test.append(0)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, x_test, y_train, y_test


def splitIDsTrainTest(pathList, n_split=6):

  hc_ids = []
  pd_ids = []

  for i in range(len(pathList)):
      imgPath = pathList[i]

      if 'HC' in imgPath:
          hc_ids.extend(list(np.unique(
              [file.split('_')[1] for file in os.listdir(imgPath) if file.endswith('.png')])))
      else:
          pd_ids.extend(list(np.unique(
              [file.split('_')[1] for file in os.listdir(imgPath) if file.endswith('.png')])))

  hc_ids = np.unique(hc_ids).tolist()
  pd_ids = np.unique(pd_ids).tolist()

  X = hc_ids + pd_ids
  y = [0]*len(hc_ids) + [1]*len(pd_ids)
  kf=StratifiedKFold(n_splits=n_split)
  train_folds=[]
  test_folds = []
  for train_index, test_index in kf.split(X,y):
    train_folds.append(np.array(X)[train_index])
    test_folds.append(np.array(X)[test_index])

  return train_folds, test_folds
"""
    train_files = pd_ids[:int(math.ceil((1 - splitSize) * len(pd_ids)))].tolist()
    train_files.extend(hc_ids[:int(math.ceil((1 - splitSize) * len(hc_ids)))])
    test_files = pd_ids[-int(math.floor(splitSize * len(pd_ids))):].tolist()
    test_files.extend(hc_ids[-int(math.floor(splitSize * len(hc_ids))):])"""
