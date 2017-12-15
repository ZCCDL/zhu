
##!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:56:01 2017

@author: user
"""

from keras.layers import Conv2D, Input
from keras.models import Model
from keras import layers
from keras.layers import BatchNormalization,ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD,Adadelta 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras.backend as K
import keras
#from keras.layers.normalization import BatchNormalization as BN
from keras.utils import plot_model



input = Input(shape=(640, 360, 3))

x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(input)
x = BatchNormalization(name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)


tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)

zuhe = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
zuhe = Flatten()(zuhe)  # this converts our 3D feature maps to 1D feature vectors
A = BatchNormalization(name='bn_conv2')(zuhe)

zuhe=Dropout(0.5)(x)
x1 = Dense(1024)(zuhe)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)

x2 = Dense(128)(x1)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)

x2=Dropout(0.5)(x2)
predictions = Dense(30, activation='softmax')(x2)


# This creates a model that includes
# the Input layer and three Dense layers
  # Create model.
model = Model(inputs=input,outputs=predictions)
#model=make_parallel.make_parallel(model,2)
adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-09)     
#model.compile(loss='mse', optimizer=adadelta,metrics=["accuracy"])  
model.compile(loss="categorical_crossentropy", optimizer=adadelta,metrics=["accuracy"])  
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255,
                                   fill_mode='constant',
                                  cval=0.
                                  )

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'new data/train/',  # this is the target directory
        target_size=(640, 360),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
print( train_generator)
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'new data/vali/',
        target_size=(640, 360),
        batch_size=32,
        class_mode='categorical')


#print validation_generator
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="center right")
        plt.show()

history = LossHistory()

#model.load_weights('fiv_try.h5')
result=model.fit_generator(
        train_generator,
        samples_per_epoch=800,
        nb_epoch=200,
        validation_data=validation_generator,
        nb_val_samples=200,
        callbacks=[history])

score = model.evaluate_generator(validation_generator,400)
print('Test score:', score[0])
print('Test accuracy:', score[1])

test_generator = test_datagen.flow_from_directory(
        'a/test_A/',
        target_size=(640, 360),
        batch_size=32,
        class_mode='categorical',
        shuffle=None)



pre=model.predict_generator(test_generator,100)
print (pre)


model.summary()

history.loss_plot('epoch')

model.save_weights('resnet3.h5')
plot_model(model, to_file='resnet3.png')
