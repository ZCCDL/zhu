#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:24:33 2017

@author: user
"""
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.layers import BatchNormalization,ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD,Adadelta 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import keras,make_parallel
#from keras.layers.normalization import BatchNormalization as BN
from keras.utils import plot_model
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(30, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model_p=make_parallel.make_parallel(model,2)
adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-09)     
#model.compile(loss='mse', optimizer=adadelta,metrics=["accuracy"])  
model_p.compile(loss="categorical_crossentropy", optimizer=adadelta,metrics=["accuracy"])  
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

train_generator = train_datagen.flow_from_directory(
        'new data/train/',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=30,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
print( train_generator)
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'new data/vali/',
        target_size=(224, 224),
        batch_size=30,
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
result=model_p.fit_generator(
        train_generator,
        samples_per_epoch=800,
        nb_epoch=500,
        validation_data=validation_generator,
        nb_val_samples=500,
        callbacks=[history])

score = model_p.evaluate_generator(validation_generator,400)
print('Test score:', score[0])
print('Test accuracy:', score[1])

test_generator = test_datagen.flow_from_directory(
        'test_A/',
        target_size=(224, 224),
        batch_size=30,
        class_mode='categorical',
        shuffle=None)



pre=model_p.predict_generator(test_generator,100)
print (pre)


model_p.summary()

history.loss_plot('epoch')

model_p.save_weights('resnet2.h5')
plot_model(model_p, to_file='resnet2.png')
