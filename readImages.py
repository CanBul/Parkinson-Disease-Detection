import cv2
import numpy as np
import os


def read_images(pathList):

    x_data = []
    y_data = []

    for i in range(len(pathList)):
        imgPath = pathList[i]

        files = [f for f in os.listdir(imgPath) if f.endswith('.png')]

        for f in files:

            img = cv2.imread(imgPath+f)  # read image
            img = img.astype('float32')  # change its type
            img /= 255  # normalize
            x_data.append(img)

            # Make sure your directory name has PD in it for PD patients!!
            if 'PD' in imgPath:
                y_data.append(1)
            else:
                y_data.append(0)

    x_data, y_data = np.array(x_data), np.array(y_data)

    return x_data, y_data
