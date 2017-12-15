#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:19:13 2017

@author: user
"""

import pandas as pd
from keras.preprocessing.image import *
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import csv
 



#df = pd.read_csv("Pig_Identification_Upload_Sample_for_Test_A.csv",header=None)
pre=np.load('pre.npy')

        
        

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'aa/test_A',
        target_size=(480, 270),
        batch_size=30,
        class_mode='categorical',
        shuffle=None)
#indexl=np.argmax(pre, axis=1)+1

#y_pred = pre.clip(min=0.00000500, max=0.99500000)
tag=[]
for i, fname in enumerate(test_generator.filenames):
        print i
        idp = int(fname.split('/')[1].split('.')[0])        
        for k in range(30):
            tag.append([idp,k+1,str("%.8f" % pre[i][k])])            
with open('Pig_Identification_Upload_Sample_for_Test_A.csv', 'wb') as csvfile:
    writer=csv.writer(csvfile)
    for x in tag:
        writer.writerow(x)


