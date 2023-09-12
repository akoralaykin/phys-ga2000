#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


# In[3]:


x_values = np.arange(-10, 10, 0.001, dtype=np.float32)
plt.plot(x_values, gaussian(x_values, 0, 3))
plt.xlabel('X (a.u.)')
plt.ylabel('Y (a.u.)')
plt.savefig('gaussian.png')
plt.show()


# In[ ]:




