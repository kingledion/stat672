# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#from datetime import datetime

def main():
    prob1c()
    prob1d()
    prob1e()

def prob1c():
    x, y = gen_sample(1)
    print(x, y)
        
def prob1d():
    for d in [1, 2, 5]:
        m = logit_model(*gen_sample(10, d))
        # nothing really to output here
        
def prob1e():
    data = {d: [] for d in [1, 2, 5]}
    for d in [1, 2, 5]:
        for n in [10, 20, 50, 100, 200, 500, 1000]:
            scores = [get_scores(n, d) for _ in range(50)]
            data[d].append((n, *[np.mean(x) for x in zip(*scores)]))
            

    for d in [1, 2, 5]:
        n, erremp, errgen = zip(*data[d])
        fig = plt.figure()
        plt.plot(n, erremp, '-rx', n, errgen, '-bx') # training set in red, independent set in blue
        plt.xlabel("n = number of samples")
        plt.ylabel("% correct")
        plt.axis((0, 1000, .5, 1))
        fig.suptitle("d = {0} dimensions".format(d))
        plt.show()    

# use the logit_model function to generate a training set and get error scores, 
    #then get error scores on a different independent sample
# argument n and d are as in gen_sample
# returns error scores for the training set and independent set, respectively
def get_scores(n, d):

    # run logit_model until a successful sample is obtained. See not in that logit_model()
    m = None
    while not m:
        training = gen_sample(n, d)
        m = logit_model(*training)
    
    erremp = m.score(*training)
    errgen = m.score(*gen_sample(1000, d))
    
    return erremp, errgen
    
    
        
            
# Generate a sample x and y
# argument n is the number of samples, d is the number of dimension for exponentialion of x
# output x is an [n x d] matrix where column 0 is ones, column 1 is random numbers and columns 2 to d are x[1]**d
    # if d is omitted, then x is an n length 1-dim array. 
# output y is an n length 1-dim array containing y values generated from comparing a randomly generated number
    # to either 0.9 or 0.2 depending on the value of x       
def gen_sample(n, d = 0):

    x = np.random.random(n)
    y = np.random.random(n)
    y = np.select([(x<0.2)+(x>0.8), True], [y<0.9, y<0.2])
    
    if d > 0:
        x = np.stack([x**d for d in range(d+1)], axis=1)
    
    return x, y

# Run fit logit model on test set x and responses y
def logit_model(x, y):
    model = LogisticRegression(fit_intercept = False, C=1e5)
    
    # if the model does not get two classes; i.e. the output has only 1s or only 0s
    # Then no valid model can be made and the logistic regression fails on a
    # ValueError. In that case we will return None to indicate to the calling 
    # function that failure happened.
    try:
        model.fit(x, y)
    except ValueError as e:
        return None
    return model
                             
    
main()
