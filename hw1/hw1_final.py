# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:36:34 2017

@author: dhartig
"""

# tested in python 3.5.2


import numpy.random as random, numpy.linalg as linalg, numpy as np
from datetime import datetime
from math import sqrt

def main():
    prob_1() # comment out to run one problem or the other
    #prob_2a()

# generate random vectors from unit cube and determine if the are within the unit sphere
# argument d is number of dimensions, n is the number of random vectors to generate 
# return a list of tuples with random vectors within the unit sphere
def cartVectorGen(d, n): 
   
    vectors = []

    while len(vectors) < n:
    
        v = [random.random()*2-1 for i in range(d)]
        vectors = vectors + [tuple(v)] if linalg.norm(v) <= 1 else vectors
        
    return vectors

# generate random vectors from sphere
# argument d is number of dimensions, n is the number of random vectors to generate, 
# return a list of tuples with random vectors within the unit sphere
def sphereVectorGen(d, n):
    
    Theta = np.random.randn(n, d)
    R = random.rand(n)**(1.0/d)
    vectors = Theta * np.reshape(R / np.linalg.norm(Theta, axis=1), (n, 1))
    
    #vectors = []
    
    #while len(vectors) < n:
        
     #   r = random.random()
     #   z = [random.normal() for i in range(d)]
     #   theta = [r * x / linalg.norm(z) for x in z]
        
     #   vectors.append(tuple(theta))
        
    return vectors
        

def prob_1():
    print("Problem 1")
    startTime = datetime.now()
    cartVectorGen(10, 100)
    print("Part a - runtime:", datetime.now() - startTime)
    # Average result ~.73 seconds on my computer
    
    startTime = datetime.now()
    sphereVectorGen(10, 100)
    print("Part b - runtime:", datetime.now() - startTime)
    # Average result ~.012 seconds on my computer
    
    print()
    

def errorProb(d):
    count = 0
    for i in range(1000):
        x_plus = random.normal(5/sqrt(d), sqrt(4), d) #random.normal takes stdev not var, so use sqrt(var)
        x_minus = random.normal(-5/sqrt(d), 1, d)
        z = random.normal(5/sqrt(d), sqrt(4), d)
        
        count = count + 1 if linalg.norm(x_plus-z) >= linalg.norm(x_minus-z) else count
        
    return(float(count)/1000)
        

def prob_2a():
    print("Problem 2a")
    print("d\tfreq")
    for d in [1,2,5,10,20,33,50,100,200,500,1000]:
        print(d, "\t{0:.3f}".format(errorProb(d)))
    #probability approaches 1

main()

def test():
    startTime = datetime.now()
    for i in range(10):
        cartVectorGen(2, 10000)
    print("Part a - runtime:", datetime.now() - startTime)
        
    startTime = datetime.now()
    for i in range(10):
        sphereVectorGen(2, 10000)
    print("Part a - runtime:", datetime.now() - startTime)
    

    
