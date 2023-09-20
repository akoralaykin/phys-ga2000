#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


N=1000
A=np.zeros([N+1,N+1],float)
def mendelbrot(x,y):
    c=complex(x,y)
    z=c
    n=0
    while n<100:
        z=z*z+c
        if np.absolute(z)>2:
            return 0
        n+=1
    else:
        return 1
i=0
j=0
for x in np.linspace(-2,2,N):
    for y in np.linspace(-2,2,N):
        a=mendelbrot(x,y)
        A[i,j]=a
        j +=1
    i +=1 
    j =0
plt.imshow(A)
plt.xlabel('# of the element in X-direction')
plt.ylabel('# of the element in Y-direction')
plt.gray()
plt.savefig('mendelbrot2.png')
plt.show()


# In[ ]:




