from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


import sys
sys.path.append('..')
from tqdm.keras import TqdmCallback
from readImages import read_images, splitIDsTrainTest

pathList = ['/content/Parkinson-Disease-Detection/data/KGL/ReadTextPDImages/',
            '/content/Parkinson-Disease-Detection/data/KGL/ReadTextHCImages/']
train_files, test_files = splitIDsTrainTest(pathList, splitSize=0.2)
X_train, X_test, y_train, y_test = read_images(
    pathList, train_files, test_files)


# Split data into training and test sets

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

model.fit(x=X_train, y=y_train, epochs=2, validation_data=(
    X_test, y_test), verbose=1)
