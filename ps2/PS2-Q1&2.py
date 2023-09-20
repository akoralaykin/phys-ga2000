#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:


value32 = np.float32(100.98763)
diff = 100.98763 - value32
print(np.absolute(diff))


# In[3]:


def M_constant(number_of_atoms):
    e = 1 # Electron charge. 
    a = 1 # Lattice constant.
    epsilon_zero = 1 # Permittivity of vacuum.
    #Their values are trivial since our aim is to calculate M.
    L = number_of_atoms
    V = 0 # Potential is zero initially.
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if i == 0 and j == 0 and k == 0: # Skip the atom located at the origin.
                    continue
                elif np.absolute(i+j+k) % 2 == 0: 
                    V += e / (4*np.pi*epsilon_zero*a*np.sqrt(i**2+j**2+k**2)) # Postively charged atom. 
                else:
                    V -= e / (4*np.pi*epsilon_zero*a*np.sqrt(i**2+j**2+k**2)) # Negatively charged atom. 
    
    M_constant = V*4*np.pi*epsilon_zero*a / e # Madelung constant.
    return M_constant
start_time = time.time()
print ("Madelung Constant is:", M_constant(100))
print("It takes %s seconds" % (time.time() - start_time))


# In[4]:


L = 100
nx, ny, nz = (L, L, L)
x = np.linspace(-L, L, 2*nx+1, dtype=np.float32)
y = np.linspace(-L, L, 2*ny+1, dtype=np.float32)
z = np.linspace(-L, L, 2*nz+1, dtype=np.float32)

xv, yv, zv = np.meshgrid(x, y, z)



start_time = time.time()

M = (-1)**(xv + yv + zv)/np.sqrt(xv**2 + yv**2 + zv**2)
M[(xv == 0)*(yv == 0)*(zv == 0)] = 0
M = np.sum(M)
print ("Madelung Constant is:", M)
print("It takes %s seconds" % (time.time() - start_time))



# In[ ]:




