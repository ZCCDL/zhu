#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:34:43 2017

@author: user
"""

import pandas as np
df = pd.read_csv("Pig_Identification_Upload_Sample_for_Test_A.csv",header=None)


df.to_csv('predjj.csv', index=None)