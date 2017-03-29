# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

path = "." # set path equal to location of csv file. "." represents same directory

# part a
X = np.loadtxt(open(path + "/hw3_X.csv", "r"), delimiter = ",") 
y = np.loadtxt(open(path + "/hw3_y.csv", "r"), delimiter = ",")

# part b
X = np.concatenate((X, *[(X.T[col] * X.T[col:]).T for col in range(X.shape[1])]), axis = 1)

n, D = X.shape

print(X.shape)
print(y.shape, "\n")

# part c
test = X[3::4].copy() # must make deep copy; otherwise slicing notation returns view
                        # object that will alter every fourth column of original matrix X
                        # when we center and scale

training = np.delete(X, slice(3, None, 4), axis = 0)

y_test = y[3::4].copy()
y_train = np.delete(y, slice(3, None, 4), axis = 0)

print(test.shape)
print(training.shape, "\n")

n_train = training.shape[0]
n_test = test.shape[0]

# part d

# center y
y_test = y_test - np.mean(y_test)
y_train = y_train - np.mean(y_train)

# standardize test and training
test = test - np.mean(test)
training = training - np.mean(training)

print("Test error:", np.sum(test)) # does not equal zero, seems likely to be floating point error
print("Training error:", np.sum(training), "\n") # hopefully outside scope of this class to fix this

test = test / np.sqrt(np.sum(test**2))
training = training/ np.sqrt(np.sum(training**2))

# part e
def ridge_reg():

    U, s, V = np.linalg.svd(training, full_matrices = False)
    lam = [(x-26)/2 for x in range(53)]  
    Uy = np.dot(U.T, y_train)
    
    errors = []
    
    for l in lam:
        slam = np.diag(s/(s**2+2**l))
        wridge = np.dot(V.T, np.dot(slam, Uy))
        errors.append((1/n_test)*np.linalg.norm(y_test - np.dot(test, wridge))**2)
        print(l, errors[-1])
    
    mindex = errors.index(min(errors))
    print(lam[mindex], errors[mindex])
    
    return lam, errors
    
   
ridge_lam, ridge_errors = ridge_reg()

# part f
def lasso_reg():
    
    lam = [(x-24)/4 for x in range(49)]
    errors = []   
    
    lam_factor = np.sqrt(np.log(D)/n_train)
    print(lam_factor, D, n_train)
    
    for l in lam:
        lasso = linear_model.Lasso(alpha=lam_factor*2**l)
        lasso.fit(training, y_train)
        errors.append((1/n_test)*np.linalg.norm(y_test - lasso.predict(test))**2)
        print(l, errors[-1])
        
    mindex = errors.index(min(errors))
    print(lam[mindex], errors[mindex])

    return lam, errors
  
lasso_lam, lasso_errors = lasso_reg()  

fig = plt.figure()
plt.plot(ridge_lam, ridge_errors, '-rx', lasso_lam, lasso_errors, '-bx') # training set in red, independent set in blue
plt.xlabel("lambda (2^x)")
plt.ylabel("test error")
fig.suptitle("Red = ridge, blue = lasso")
plt.show() 

    









