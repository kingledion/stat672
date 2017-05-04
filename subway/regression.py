#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:30:19 2017

@author: dhartig
"""

import pandas as pd, numpy as np, sklearn.svm as svm, warnings

def std_features(X, mean, std):
    X = X - mean
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            X = X/std
        except Warning as warn:
            X = np.nan_to_num(X)
    return(X)

# Read in data; merge into two datasets
with open("/opt/school/stat672/subway/boston_stations.csv") as csvin:
    sdata = pd.read_csv(csvin)
with open('/opt/school/stat672/subway/boston_subway_ridership.csv') as csvin:
    rdata = pd.read_csv(csvin, delimiter = ';', names = ['name', 'riders'])
data1 = pd.merge(sdata, rdata, how = 'inner', on='name')
    
with open("/opt/school/stat672/subway/chicago_stations.csv") as csvin:
    sdata = pd.read_csv(csvin)  
with open('/opt/school/stat672/subway/chicago_subway_ridership.csv') as csvin:
    rdata = pd.read_csv(csvin,  delimiter = ';', names = ['name', 'riders'])  
data2 = pd.merge(sdata, rdata, how = 'inner', on='name')
    
# For all datasets, calculate total values from density values
for d in [data1, data2]:  
    d['pop'] = d['area'] * d['popdensity']
    d['households'] = d['area'] * d['housedensity']
    d['totalpay'] = d['area'] * d['paydensity']
    d['employment'] = d['area'] * d['empdensity']  

# Select columns for 
X1 = data1.as_matrix(columns=['pop','households','totalpay','employment'])
X2 = data2.as_matrix(columns=['pop','households','totalpay','employment'])

y1 = np.ravel(data1.as_matrix(columns=['riders']))
y2 = np.ravel(data2.as_matrix(columns=['riders']))

print('Linear Least Squares Regression')
# Create model for Boston
coeff, resid, rank, s = np.linalg.lstsq(X1, y1)
# Get score for Boston
sstot = sum((y1 - np.ones(y1.shape)*np.mean(y1))**2)
print(1-resid[0]/sstot)
# Get score for Chicago
predicted = np.dot(X2, coeff)
sstot = sum((y2 - np.ones(y2.shape)*np.mean(y2))**2)
ssres = sum((y2 - predicted)**2)
print(1-ssres/sstot)
print()

# Create model for Chicago
coeff, resid, rank, s = np.linalg.lstsq(X2, y2)
# Get score for Chicago
sstot = sum((y2 - np.ones(y2.shape)*np.mean(y2))**2)
print(1-resid[0]/sstot)
# Get score for Boston
predicted = np.dot(X1, coeff)
sstot = sum((y1 - np.ones(y1.shape)*np.mean(y1))**2)
ssres = sum((y1 - predicted)**2)
print(1-ssres/sstot)


print('Radial Basis Function SVR (C = 10**5)')
# Create model for Boston
# Standardize X values
mn, st = np.mean(X1, axis=0), np.std(X1, axis=0)
X1std = std_features(X1, mn, st)
X2std = std_features(X2, mn, st)

model = svm.SVR(kernel='rbf', C = 10**5)
model.fit(X1std, y1)
# Get score for Boston
print(model.score(X1std, y1))
# Get score for Chicago
print(model.score(X2std, y2))
print()

# Create model for Boston
# Standardize X values
mn, st = np.mean(X2, axis=0), np.std(X2, axis=0)
X1std = std_features(X1, mn, st)
X2std = std_features(X2, mn, st)

model = svm.SVR(kernel='rbf', C = 10**5)
model.fit(X2std, y2)
# Get score for Boston
print(model.score(X2std, y2))
# Get score for Chicago
print(model.score(X1std, y1))
print()






#for c in [30000,40000,50000,60000,70000,80000,90000,100000, 110000, 120000, 130000, 140000, 150000]:
#    print(c)
#    model = svm.SVR(kernel='rbf', C = c)
#    model.fit(Xsvm, y)  
#    print(model.score(Xsvm, y)) 
#    print(model.score(Xtestsvm, ytest))