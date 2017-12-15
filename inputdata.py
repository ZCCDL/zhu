#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:34:56 2017

@author: user
"""

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

path='/home/user/zhu/train/'

#将所有的图片resize成100*100
w=299
h=299
c=3




cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
print cate[0].split('/')

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpeg'):
            print('reading the images:%s'%(im))
            label=int(folder.split('/')[5])-1
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(label)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)


data,label=read_img(path)
