# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:11:42 2017

@author: dhartig
"""

import numpy as np, warnings, itertools as it
import sklearn.linear_model as sklinear



def main():
       
    part1()
    part2()
    part2D()

# Part 1
def part1():
    
    print("PART 1\n")    
    
    X = np.random.normal(size=(100,10000))
    y = np.array([1, -1]*50)
    
    # print a sample of two rows, 5 columns, so we can see that the second row is negated
    # after being multiplied by -1 in y
    print(X[:2, :5], "\n")
    
    # multiply each row of X by the corresponding entry of y
    cjmatrix = X*y[:, np.newaxis]
    # observe tha the second row is negated compared to above
    print(cjmatrix[:2, :5], "\n")
    
    # take the abs mean along the 0th axis, to get cj; find the top five features
    cj = np.abs(np.mean(cjmatrix, axis=0))
    maxindices = np.argpartition(cj, -5)[-5:]
    print(maxindices)
    
    # select the columns corresponding to the five maxima from the X matrix
    X_S = X[:, maxindices] 
    # observe that we have 5 features for each of 100 rows
    print(X_S.shape)
    
    cverror = cross_validate(X_S, y, 10)
    print("Cross Validation Error:", cverror)
    
def part2():
    
    print("\nPART 2A-B\n")    
    
    # Read in data
    path = "."
    
    Xtrain = np.loadtxt(open(path + "/Xtrain.csv", "r"), delimiter = ",") 
    ytrain = np.loadtxt(open(path + "/Ytrain.csv", "r"), delimiter = ",")
    
    Xtest = np.loadtxt(open(path + "/Xtest.csv", "r"), delimiter = ",") 
    ytest = np.loadtxt(open(path + "/Ytest.csv", "r"), delimiter = ",")
    
    # number of features
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
    
    # indices for 5 fold split. In a list of 4-tuples with the training indices,
    # test indices, training y, and test y. Precomputing so as to not duplicate
    # within loops
    folds = np.array_split(range(n), 5)
    folds = [(list(set(range(n)) - set(f)), f) for f in folds]
    folds = [(f[0], f[1], ytrain[f[0]], ytrain[f[1]]) for f in folds]
       
    # list of all feature subsets and list of errors to be indexed identically
    pset = powerset(d)
    pset_err = []
    
    # iterate through all feature subsets, and store error from five-fold cross
    # validation
    count = 0
    for ind in pset:
        print("Subset {0}:{1}".format(count, len(pset)), end='\r')
        count += 1
        
        errors = []
        Xi = Xtrain[:, ind]
        for itrn, itst, ytrn, ytst in folds:
          
            X = Xi[itrn, :]
            
            # fit model
            weight = np.dot(X.T,X)
            weight = weight if np.isscalar(weight) else np.linalg.inv(weight)
            weight = np.dot(np.dot(weight, X.T), ytrn)
            
            # calculate errors for test set
            est = np.sign(np.dot(Xi[itst, :], weight))
            err = sum(np.abs(ytst - est))/2/len(ytst)
           
            errors.append(err)
          
        pset_err.append(np.mean(errors))
            
    # select the lowest error feature indices to generate final classficer
    minind = np.argmin(pset_err)
    print("\n\nIndices:", pset[minind])
    print("Cross Validation Error:", pset_err[minind])
    
    # select final classifier
    Xi = Xtrain[:, pset[minind]]
    weight = np.dot(Xi.T,Xi)
    weight = weight if np.isscalar(weight) else np.linalg.inv(weight)
    weight = np.dot(np.dot(weight, Xi.T), ytrain)   

    # Evaluate the test set
    est = np.sign(np.dot(Xtest[:, pset[minind]], weight))
    err = sum(np.abs(ytest - est))/2/len(ytest)  
    
    print("Test set Error:", err)

    # part b
    print("\nAlignment of features with ytrain")
    for i in range(d):
        test = sum(np.abs(ytrain - np.sign(Xtrain[:, i])))/2/len(ytrain)        
        print(i+1, test)
    
def part2D():
    
    print("\nPART 2D\n")
    
    # Read in data, easier to just copy and paste here
    path = "."
    
    Xtrain = np.loadtxt(open(path + "/Xtrain.csv", "r"), delimiter = ",") 
    ytrain = np.loadtxt(open(path + "/Ytrain.csv", "r"), delimiter = ",")
    
    # number of features
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]
          
    # list of all feature subsets and list of errors to be indexed identically
    pset = powerset(d)
    
    for i in range(20):

        # reshuffle ytrain before setting the folds   
        np.random.shuffle(ytrain)     
        
        folds = np.array_split(range(n), 5)
        folds = [(list(set(range(n)) - set(f)), f) for f in folds]
        folds = [(f[0], f[1], ytrain[f[0]], ytrain[f[1]]) for f in folds]
        
        # iterate through all feature subsets, and store error from five-fold cross
        # validation
        pset_err = []
        count = 0
        for ind in pset:
            print("Subset {0}:{1}".format(count, len(pset)), end='\r')
            count += 1
            
            errors = []
            Xi = Xtrain[:, ind]
            for itrn, itst, ytrn, ytst in folds:
              
                X = Xi[itrn, :]
                
                # fit model
                weight = np.dot(X.T,X)
                weight = weight if np.isscalar(weight) else np.linalg.inv(weight)
                weight = np.dot(np.dot(weight, X.T), ytrn)
                
                # calculate errors for test set
                est = np.sign(np.dot(Xi[itst, :], weight))
                err = sum(np.abs(ytst - est))/2/len(ytst)
                
                errors.append(err)
              
            pset_err.append(np.mean(errors))
            
        minind = np.argmin(pset_err)
        print("\x1b[2K\rSubset {0} complete".format(i+1))
        print("Indices:", pset[minind])
        print("Cross Validation Error:", pset_err[minind], "\n")        




    # For a given integer, return the powerset of all subsets of range(num) with
    # len() > 0. This is a powerset of all possible combinations indices of 
    # the features (columns) of an m x n  matrix with n = num.
def powerset(num):
    chn = it.chain.from_iterable(it.combinations(range(num), r+1) for r in range(num))
    # chn is an iterator over tuples, so we hvae to turn it into a list before returning
    # if we want ot use it more than once
    return [x for x in chn]



    # NOT USED AS DISCUSSED IN EMAIL, SAVING HERE FOR MY FUTURE USE

    # Some of the features may have variance zero. These features should have a 
    # standardized representation of zero. If we detect such a warning, we use
    # try_nan_to_num to replaces the 'nan' from the divide by zero std operation
    # with a zero. We also convert the warning to be raised instead of printed
    # so we can catch and ignore it instead of dirtying up the screen.
def std_features(X, mean, std):
    X = X - mean
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            X = X/std
        except Warning as warn:
            X = np.nan_to_num(X)
    return(X)


     
    # perform cross validation. Takes as arguments X, a 2-d array or matrix  
    # (there is no error checking to enforce this input); y a 1-d array of 
    # classifications for X; num_folds, an integer which is the 
    # number of way to partition the data into test sets
def cross_validate(X, y, num_folds):
    
    n = X.shape[0]    
    
    # Split the rows of X_S into even folds. make a list of the indices in each fold
    folds = np.array_split(range(n), num_folds)

    # For each fold, select the training set from X_S and y with indices indicated by 
    # train_i. Fit the training set, then test the test set, with indices indicated
    # by each f in folds. Store the errors in err
    errs = []
    for f in folds:
        train_i = list(set(range(n)) - set(f))

        # C is left as default 1.0; fit intercept is default true
        logmodel = sklinear.LogisticRegression()
        logmodel.fit(X[train_i, :], y[train_i])
        error = 1-logmodel.score(X[f, :], y[f])
        #print('Test Error:', error)
        errs.append(error)
    
    print()
    return np.average(errs)

    
main()