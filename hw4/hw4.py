# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm

path = "." # set path equal to location of csv file. "." represents same directory

# read in data
Xtrain = np.loadtxt(open(path + "/spam_Xtrain.csv", "r"), delimiter = ",") 
ytrain = np.loadtxt(open(path + "/spam_Ytrain.csv", "r"), delimiter = ",")

Xtest = np.loadtxt(open(path + "/spam_Xtest.csv", "r"), delimiter = ",") 
ytest = np.loadtxt(open(path + "/spam_Ytest.csv", "r"), delimiter = ",")

features = []
with open(path + "/featurenames", "r") as fin:
    for row in fin:
        if len(row.strip()) > 0:
            features.append(row.strip())
        
# scale Xtest and Xtrain
Xtest = Xtest / np.sqrt(np.sum(Xtest**2))
Xtrain = Xtrain/ np.sqrt(np.sum(Xtrain**2))

cost = [10**((x-2)/2) for x in range(12)]
#cost = [10**4.5]

for c in cost:
    print('C =', c)
    model = svm.SVC(kernel='linear', C=c)
    model.fit(Xtrain, ytrain)
    print("support vectors:", len(model.support_))
    print("training error:", "{0:.3f}".format(model.score(Xtrain, ytrain)))
    print("test error:", "{0:.3f}".format(model.score(Xtest, ytest)))
    truepos = sum([1 if i == 1 and j == 1 else 0 for i, j in zip(ytest, model.predict(Xtest))])
    falsepos = sum([1 if i == -1 and j == 1 else 0 for i, j in zip(ytest, model.predict(Xtest))])
    print("True positive rate", "{0:.3f}".format(truepos/len(ytest)))
    print("False postive rate", "{0:.3f}".format(falsepos/len(ytest)))
    wstar = model.coef_
    print("w*:", wstar)    
    print()

# part d
ind = np.argpartition(wstar[0], -5)[-5:]
print("Most spammy features:", [features[i] for i in ind])
ind = np.argpartition(wstar[0], 5)[:5]
print()

print("Least spammy features:", [features[i] for i in ind]) 
print()

# part f
for c in [10**4.5, 10**7]:
    print('C =', c)
    model = svm.SVC(kernel='linear', C=c, class_weight={1: 1, -1: 10})
    model.fit(Xtrain, ytrain)
    print("support vectors:", len(model.support_))
    print("training error:", "{0:.3f}".format(model.score(Xtrain, ytrain)))
    print("test error:", "{0:.3f}".format(model.score(Xtest, ytest)))
    truepos = sum([1 if i == 1 and j == 1 else 0 for i, j in zip(ytest, model.predict(Xtest))])
    falsepos = sum([1 if i == -1 and j == 1 else 0 for i, j in zip(ytest, model.predict(Xtest))])
    print("True positive rate", "{0:.3f}".format(truepos/len(ytest)))
    print("False postive rate", "{0:.3f}".format(falsepos/len(ytest)))   
    print()
