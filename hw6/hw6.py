# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:59:54 2017

@author: dhartig
"""

import numpy as np
from statistics import mean

# check either greater than or less than equality
# Calculate an array that evaluates to 1 when the feature value is compared to
# t and zero otherwise. Score this against the labels if this score is higher
# than the last score, 
def classify(feature, labels, weights, gt, t):
    eval_t = (feature < t)
    if gt: # evaluate less than. if you want to get geq, invert
        eval_t = np.invert(eval_t)
    
    errors = (eval_t != labels)
    return sum(errors * weights)/sum(weights), errors
        
# PART B
# features must be an nxd matrix, labels a len n array
# weights a len n array. No error checking.
def boost(features, labels, weights):
    
    
    # calculate the best prediction set (c, t) for each j in d

    # since c_1 should not equal c_2 (or we will get no
    # information), we will use 
    # c = 1 to represent c_1 = 1; c_2 = -1 and 
    # c = 0 to represent c_1 = -1; c_2 = 1

    n = features.shape[0]
    d = features.shape[1]
    
    #initialize return values
    min_score = 1000000 # very high number, so = classify(feature, labels, weights, gt, upper)any score is lower
    c = 1
    j_star = 0
    t_star = 0
    
    
    for j in range(d):
        
        # select the jth feature
        feature = features[:, j]
        slist = sorted(feature)
        
        for gt in range(2): # iterates through 0 and 1; for < and >
        
            # Mergesort inspired search for the best possible stumps
            # Sort all he values of the feature j, O(nlog(n)); 
            # in each iteration split the features in half and choose a half
            # based on which half's midpoint returns a lower score, O(log(n))
            # Each classification as we iterate takes time O(n) for a total
            # runtime of O(nlog(n))

            min_i = 0; max_i = len(slist)
            last = 0; current = int(mean((min_i, max_i)))
            
            while last != current and max_i - min_i > 1:

                upper = int(mean((current, max_i)))
                lower = int(mean((current, min_i)))

                uscore, umcl = classify(feature, labels, weights, gt, slist[upper])
                lscore, lmcl = classify(feature, labels, weights, gt, slist[lower])
                
                if uscore < lscore:
                    score, mcl = uscore, umcl; t = slist[upper]
                    last = current; min_i = current; current = upper
                else:
                    score, mcl = lscore, lmcl; t = slist[lower]
                    last = current; max_i = current; current = lower
                if score < min_score:
                    min_score = score
                    j_star = j
                    c = gt
                    t_star = t
                    misclass = mcl # misclass is a vector of length n that is
                                    # 1 if Y_i != g(X_i)m and 0 otherwise 

                
                
        # ALTERNATIVE METHOD BASED ON SIMPLY DIVIDING THE FEATURE RANGE
        # INTO NO MORE THAN LOG(N) SUBSETS
        # AFTER A SMALL AMOUNT OF TESTING, THIS SEEMS TO BE FASTER, AND
        # POSSIBLY MORE ACCURATE. INCLUDED JUST IN CASE IT IS.
        #min_t, max_t = min(features[:, j]), max(features[:, j])
        #steps = max(min(n, 10), int(np.log(n))) # truncate to get an int
        #stepsize = (max_t - min_t)/steps
                       
        #for i in range(steps+1):
        #    t = min_t + stepsize*i
        # 
        #    # test c = 1 condition
        #    for gt in range(2): # iterates through 0 and 1; for < and >
        #        score, mcl = classify(feature, labels, weights, gt, t)
        #        #print(score, j, gt, t)
        #        if score < min_score:
        #            min_score = score
        #            j_star = j
        #            c = gt
        #            t_star = t
        #            misclass = mcl # misclass is a vector of length n that is
                                    # 1 if Y_i != g(X_i)m and 0 otherwise
                    
    return j_star, c, t_star, misclass

# PART C 
# SINCE WE WANT TO KNOW THE ERRORS BETWEEN EACH ITERATION, I CHANGED THIS 
# METHOD TO PERFORM THE COMPARISON IN EACH LOOP WITH A TUPLE OF PASSED DATA 
# Takes features, labels, weights, and a number of iterations 'M'. The optional
# variable tests contains Xtest and ytest; if it is passed, this method will
# print an error score on the test set for each loop. Returns
# a set of decision stumps as tuples (a_m, j_star, c, t_star) 

def adapt(features, labels, weights, M, tests = None):

    stumps = []
    for i in range(M):
        j_star, c, t_star, misclass = boost(features, labels, weights)
        if sum(misclass) == 0: 
                            # implies vector of all 0's, no error
                            # not only will this break the following calculation
                            # but it also indicates that our 'weak' classifier
                            # is actually a perfect classifier. In that case 
                            # return the perfect classifier
            return [(1, j_star, c, t_star)]
            
        e_m = sum(misclass*weights)/sum(weights)
        a_m = np.log((1-e_m)/e_m)
        
        #print("\n", misclass*weights, e_m, a_m, "\n")
        

        #print(weights, a_m, misclass)
        #print(a_m * misclass)
        #print(weights * np.exp(a_m * misclass))

        
        weights = weights * np.exp(a_m * misclass)
        weights = weights / sum(weights)

                
        stumps.append((a_m, j_star, c, t_star))
        
        if tests:
            Xtest, ytest = tests
            #print(">>>", stumps[-1])
            print(i, compare(Xtest, ytest, stumps))
            #input()
    return stumps

  
    
# PART D
def predict(features, stumps):
    
    predict = np.zeros(features.shape[0]) # initialize a zeros array the same size as 
                                # labels to use for predictions
    for a_m, j_star, c, t_star in stumps:
        feature = features[:, j_star]
        
        # c_vector will be (1 if c else -1) if a feature >= t_star, and
        # be (-1 if c else 1) otherwise
        c_vector = (feature >= t_star).astype(int)
        c_vector[c_vector == 0]  = -1 # where the boolean was false, put a -1
        c_vector = c_vector if c else np.invert(c_vector)
                     
        predict += a_m * c_vector       
        
    return (predict > 0).astype(int)
    
# extend predict to compare to known labels of test set and return error score
def compare(feature, labels, stumps):
    predictions = predict(feature, stumps)
    return np.sum(predictions == labels)/len(labels)

# build the test sets according to the rules for part e
# takes n, d, dimensions of the feature matrix as input
# returns feature matrix X and labels Y as output
def buildptesets(n, d):
    
    X = np.random.rand(n,d)
    comp = (X[:, :5] > np.array([[0.4,0.2,0.7,0.8,0.3]])).astype(int)
    comp[comp == 0] = -1 # change 0/false to -1
    y = np.sign(np.sum(comp, axis=1))
    # make y a boole    comp = comp * np.sign(np.random.rand(5) - 0.5)*0.4+0.5 an array 
    y = (y > 0)
    
    return X, y
    
# build the test sets according to the rules for part f
def buildptfsets(n, d):
    
    X = np.random.rand(n,d)
    xi = (np.random.rand(5) < 0.9)
    xi[xi == 0] = -1
    comp = (X[:, :5] > np.array([[0.4,0.2,0.7,0.8,0.3]])).astype(int)
    comp[comp == 0] = -1 # change 0/false to -1
    y = np.sign(np.sum(comp, axis=1))
    # make y a boolean array 
    y = (y > 0)
    
    return X, y
        
    
def part_e():
    print("Part e.")
    Xtrain, ytrain = buildptesets(1000, 100)
    Xtest, ytest = buildptesets(10000, 100)
    
    weights = np.ones(Xtrain.shape[0]) * 1/Xtrain.shape[0]
    
    adapt(Xtrain, ytrain, weights, 100, (Xtest, ytest))

def part_f():
    print("\n", "Part f.") 
    Xtrain, ytrain = buildptfsets(1000, 100)
    Xtest, ytest = buildptfsets(10000, 100)
    
    weights = np.ones(Xtrain.shape[0]) * 1/Xtrain.shape[0]
    
    adapt(Xtrain, ytrain, weights, 100, (Xtest, ytest))

        
       
part_e()
part_f()


# Test code while building
#features = np.array([[4474,7033,10827],[577,1328,2994],[1192,1983,3268],[599,1185,2154],[1052,1622,2672],[1131,1752,2685],[786,1288,2272],[160,630,1565],[311,708,1537],[172,641,1637],[612,1075,1820],[496,963,1523],[398,698,1370],[366,633,1377]])               
#labels = np.array([1,0,1,1,1,1,1,0,0,0,0,0,0,0])                             
#weights = np.ones(features.shape[0])/features.shape[0]

#ret = adapt(features, labels, weights, 4)
#for stump in ret:
#    print(stump)
#print(predict(features, ret))
   
                
                
                
                
                
                
    
    