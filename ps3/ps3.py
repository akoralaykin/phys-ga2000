#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Question-1

def f(x):
    return x*(x - 1)


def derivative(x, d):
    return (f(x + d) - f(x))/ d

print(derivative(1, 1e-2))
print(derivative(1, 1e-4))
print(derivative(1, 1e-6))
print(derivative(1, 1e-8))
print(derivative(1, 1e-10))
print(derivative(1, 1e-12))
print(derivative(1, 1e-14))


# In[2]:


# Question-2

N = 100
time_array = np.zeros(N)
for i in np.arange(N):
    matrix = np.random.randint(10, size=(i+1, i+1))
    C = np.zeros([i+1,i+1] ,float)
    start_time = time.time()
    for j in range(i):
        for k in range(i):
            for l in range(i):
                C[j,k] += matrix[j,l]*matrix[l,k]
    time_array[i] = time.time() - start_time
N_3_array = (np.arange(N)+1)**3
print(time_array[-1])

time_array_2 = np.zeros(N)
for i in np.arange(N):
    matrix = np.random.randint(10, size=(i+1, i+1))
    start_time = time.time()
    C = np.dot(matrix, matrix)
    time_array_2[i] = time.time() - start_time
print(time_array_2[-1])


plt.plot(np.arange(N)+1, time_array, color = 'r', label = 'N vs. t (explicit)')
plt.plot(np.arange(N)+1, time_array_2, color = 'b',label = 'N vs. t (dot)')
plt.plot(np.arange(N)+1, (N_3_array/N_3_array[-1])*time_array[-1], color = 'g', label = 'N vs. N^3 (scaled)')
plt.xlabel('N')
plt.ylabel('t (s)')
plt.legend()
plt.savefig('q2.png')
plt.show()


# In[3]:


# Question-3
h = 1 # Time interval (s).

Bi209 = 0
Pb209 = 0
Tl209 = 0
Bi213 = 10000

pPb = 1 - 2**(-h/3.3/60)
pTl = 1 - 2**(-h/2.2/60)
pBi = 1 - 2**(-h/46/60)

Bi209_list = []
Pb209_list = []
Tl209_list = []
Bi213_list = []

t = np.arange(0,20000,h)
for ti in t:
    Bi209_list.append(Bi209)
    Pb209_list.append(Pb209)
    Tl209_list.append(Tl209)
    Bi213_list.append(Bi213)
    
    for i in np.arange(Bi213):
        if np.random.random()<pBi:
            Bi213 -=1
            if np.random.random()<0.9791:
                Pb209+=1
            else:
                Tl209+=1

    for i in np.arange(Pb209):
        if np.random.random()<pPb:
            Pb209-=1
            Bi209+=1

    for i in np.arange(Tl209):
        if np.random.random()<pTl:
            Tl209-=1
            Pb209+=1

    

plt.plot(t,Bi209_list,label='Bi209')
plt.plot(t,Pb209_list,label='Pb209')
plt.plot(t,Tl209_list,label='Tl209')
plt.plot(t,Bi213_list,label='Bi213')
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('Number of atoms')
plt.savefig('q3.png')
plt.show()


# In[4]:


# Question-4
N = 1000
tau = 3.053*60
mu = np.log(2)/tau

z = np.random.random(N)
x = np.log(1-z)/(-1*mu)
decay_times = np.sort(x)
decayed_atoms = np.arange(N)+1
not_decayed_atoms = (decayed_atoms - N)*(-1)



plt.plot(decay_times,not_decayed_atoms)
plt.xlabel('t (s)')
plt.ylabel('Number of atoms')
plt.savefig('q4.png')
plt.show()


# In[ ]:




