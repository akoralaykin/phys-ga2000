#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def partA(a,b,c):
    x_plus = ( -b + np.sqrt(b**2 - 4*a*c) ) / (2*a)
    x_minus = ( -b - np.sqrt(b**2 - 4*a*c) ) / (2*a)
    return x_plus,x_minus


# In[3]:


def partB(a,b,c):
    x_plus = (2*c) / ( -b - np.sqrt(b**2 - 4*a*c))
    x_minus = (2*c) / ( -b + np.sqrt(b**2 - 4*a*c))
    return x_plus,x_minus


# In[4]:


print(partA(0.001,1000,0.001))
print(partB(0.001,1000,0.001))


# In[5]:


def quadratic(a,b,c):
    x_plus = (2*c) / ( -b - np.sqrt(b**2 - 4*a*c))
    x_minus = c/(x_plus*a) # An alternative: ( -b - np.sqrt(b**2 - 4*a*c) ) / (2*a) 
    # In this way, we avoid subtracting two nearly equal quantities.
    return x_plus,x_minus 


# In[6]:


print(partA(0.001,1000,0.001))
print(partB(0.001,1000,0.001))
print(quadratic(0.001,1000,0.001))

