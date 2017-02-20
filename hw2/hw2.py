# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def main():
    prob1c()

def prob1c():

    x = np.random.random()
    
    p_y = 0.9 if x<0.2 or x>0.8 else 0.2
    y = 1 if np.random.random() < p_y else 0
                             
    print(x, y)
    
main()