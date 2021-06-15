
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow import keras
from augmentation import Augmentation
import os

def create_batch(directories, test=False):

    x_data= []
    y_data = []

    for directory in directories:
        files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        for f in files:
            aug = Augmentation(directory+f)
            augmentedSample = aug.process(test=test)

            x_data.append(augmentedSample)

            if 'pd' in f:
                y_data.append(1)
            else:
                y_data.append(0)
    
    return np.array(x_data), np.array(y_data)

def create_model():

    model = Sequential()

    model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                    activation ='relu', input_shape = (128,130,1)))
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
    model.add(Dense(1, activation = "sigmoid"))

    return model



paths = ['./data/fold0/', './data/fold1/', './data/fold2/','./data/fold3/']

accuracy_report = {}

for each in range(len(paths)):
    test_path = [paths[each]]
    train_path = [path for path in paths if path != test_path[0]]

    x_test, y_test = create_batch(test_path, test=True)
    x_test = x_test.reshape(-1,128,130,1)

    

    model = create_model()

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optimizer , loss='binary_crossentropy', metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='max')
    mcp_save = ModelCheckpoint('test_fold_model' + str(each)+'.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=0.00001)


    batch_size =32
    epochs = 60

    testName = test_path[0][-6:-1]
    accuracy_report[testName] = 0
    

    for epoch in range(epochs):
        x_train, y_train = create_batch(train_path)
        x_train = x_train.reshape(-1,128,130,1)

        
        print('Training has started...')
        history = model.fit(x_train,y_train, batch_size=batch_size, verbose=2,
                                epochs = 1, validation_data = (x_test,y_test))#, steps_per_epoch=x_train.shape[0] // batch_size, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])


        if history.history['val_accuracy'][0] > accuracy_report[testName]:            
            accuracy_report[testName] = history.history['val_accuracy'][0]

print(accuracy_report)    
