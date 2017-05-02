#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:30:19 2017

@author: dhartig
"""

import pandas as pd, numpy as np, csv

with open("/opt/school/stat672/subway/boston_stations.csv") as csvin:
    features = pd.read_csv(csvin)
    
with open("/opt/school/stat672/subway/chicago_stations.csv") as csvin:
    test = pd.read_csv(csvin)

labels = {}  
with open('/opt/school/stat672/subway/boston_subway_ridership.csv') as csvin:
    csvrdr = csv.reader(csvin, delimiter=";")
    for row in csvrdr:
        labels[row[0]] = row[1]
    
# Remove stations that don't exist in our dataasets
# Assembly
features = features[features.name != "'Assembly'"]

    
features['pop'] = features['area'] * features['popdensity']
features['households'] = features['area'] * features['housedensity']
features['totalpay'] = features['area'] * features['paydensity']
features['employment'] = features['area'] * features['empdensity']  

X = features.as_matrix(columns=['pop','households','totalpay','employment'])
Xtest = features.as_matrix(columns=['pop','households','totalpay','employment'])

y = np.array([labels[x] for x in features['name']])

print(X)
print(X.shape)

print(y)
print(y.shape)

coeff, resid, rank, s = np.linalg.lstsq(X, y)
print(coeff)
print()
print(np.dot(Xtest, coeff))


    