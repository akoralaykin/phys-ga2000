#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open('signal.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i)) 

data = np.array(data, dtype='float')
time = data[::2]
signal = data[1::2]


# In[3]:


plt.plot(time, signal,'.')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.savefig('1.png')
plt.show()


# In[4]:


x = time/(np.max(time))
y = signal
A = np.zeros((len(x), 4))
A[:, 0] = 1.
A[:, 1] = x
A[:, 2] = x**2
A[:, 3] = x**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)

print("Condition Number =",np.max(w)/np.min(w))

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(y)
ym = A.dot(c) 
plt.plot(x, y, '.', label='data')
plt.plot(x, ym, '.', label='model')
plt.xlabel('Time/Time_Max')
plt.ylabel('Signal')
plt.legend()
plt.title('Polynomial Fit with Order of 3')
plt.savefig('2.png')
plt.show()


plt.plot(x, y-ym, '.', label='data')
plt.xlabel('Time/Time_Max')
plt.ylabel('Residuals')
plt.savefig('3.png')
plt.show()


# In[5]:


order = 30
A = np.zeros((len(x), order + 1))

for i in np.arange(order+1):
    A[:,i] = x**i


(u, w, vt) = np.linalg.svd(A, full_matrices=False)

print("Condition Number =",np.max(w)/np.min(w))

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(y)
ym = A.dot(c) 
plt.plot(x, y, '.', label='data')
plt.plot(x, ym, '.', label='model')
plt.xlabel('Time/Time_Max')
plt.ylabel('Signal')
plt.legend()
plt.title('Polynomial Fit with Order of 30')
plt.savefig('4.png')
plt.show()


# In[6]:


omega = 2*np.pi/((np.max(x)-np.min(x))/2)
modes = 6
A = np.zeros((len(x), 2*modes+1))
A[:, 0] = 1.

for i in np.arange(1,2*modes,2):
    A[:, i] = np.cos((omega*x)*((i+1)/2))
    A[:, i+1] = np.sin((omega*x)*((i+1)/2))

(u, w, vt) = np.linalg.svd(A, full_matrices=False)

print("Condition Number =",np.max(w)/np.min(w))

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(y)
print(c)
ym = A.dot(c) 
plt.plot(x, y, '.', label='data')
plt.plot(x, ym,'.', label='model')
plt.xlabel('Time/Time_Max')
plt.ylabel('Signal')
plt.legend()
plt.title('Sin & Cos Fit with 6 modes')
plt.savefig('5.png')
plt.show()


plt.plot(x, y-ym, '.', label='data')
plt.xlabel('Time/Time_Max')
plt.ylabel('Residuals')
plt.show()


# In[7]:


def integrand(x, a):
    return np.exp(-x)*x**(a-1)
a1 = 2
a2 = 3
a3 = 4
x_val = np.arange(0,5,0.01)
y_val_a1 = integrand(x_val,a1)
y_val_a2 = integrand(x_val,a2)
y_val_a3 = integrand(x_val,a3)
plt.plot(x_val, y_val_a1, label='a = 2')
plt.plot(x_val, y_val_a2, label='a = 3')
plt.plot(x_val, y_val_a3, label='a = 4')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('6.png')
plt.show()


# In[8]:


from gaussxw import gaussxwab 

def integrand_better(x, a):
    return np.exp(-x + (a-1)*np.log(x))

# Gaussian Quadrature

z,w = gaussxwab(20,0,1)
a = 1.5
s = 0.
for k in np.arange(20):
    s += ((a-1)*w[k]*integrand_better(((a-1)*z[k])/(1-z[k]),a))/((1-z[k])**2)
    
print(s)


# In[9]:


a_val = np.array([3, 6, 10])
s = np.zeros(a_val.size)
for i in np.arange(a_val.size):           
    for k in np.arange(20):
        s[i] += ((a_val[i]-1)*w[k]*integrand_better(((a_val[i]-1)*z[k])/(1-z[k]),a_val[i]))/((1-z[k])**2)
    
print(s)


# In[ ]:




