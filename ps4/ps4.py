#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
from gaussxw import gaussxwab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from math import factorial 


# In[2]:


def integrand(x):
    return ((np.exp(x))*(x**4))/(np.exp(x)-1)**2

N = 50
T = 100

def cv(Temp, N):
    T = Temp
    theta_D = 428.
    ro = 6.022e28
    V = 1000.0e-6
    k_B = 1.38e-23
    a = 0.
    b = theta_D/T
    x,w = gaussxwab(N,a,b)
    s = 0.    
    for k in np.arange(N):
        s += w[k]*integrand(x[k])
    return s*9*V*ro*k_B*(T/theta_D)**3

T_i = 5
T_f = 500
delta_T = 0.1
T_values = np.arange(T_i,T_f+delta_T,delta_T)
N_values = np.arange(1,101)
cv_N = np.zeros(N_values.size)
cv_T = np.zeros(T_values.size)


for i in np.arange(N_values.size):
    cv_N[i] = cv(T, N_values[i])
    
for i in np.arange(T_values.size):
    cv_T[i] = cv(T_values[i], N)
    


plt.plot(T_values,cv_T,label='N = 50') 
plt.xlabel('T (K)')
plt.ylabel('C_V (J/K)')
plt.legend()
plt.savefig('1.png')
plt.show()



plt.plot(N_values,cv_N, label='T = 100 K') 
plt.xlabel('N')
plt.ylabel('C_V (J/K)')
plt.legend()
plt.savefig('2.png')
plt.show()


# In[3]:


def potential(x):
    return x**4

N = 20

def period(Amp):
    m = 1
    a = 0
    b = Amp
    x,w = gaussxwab(N,a,b)
    s = 0.
    for k in np.arange(N):
        s += w[k]/np.sqrt(potential(b)-potential(x[k]))
    return s*np.sqrt(8*m)


a_i = 0.01
a_f = 2
delta_a = 0.01
a_values = np.arange(a_i,a_f+delta_a,delta_a)
period_a = np.zeros(a_values.size)

for i in np.arange(a_values.size):
    period_a[i] = period(a_values[i])


plt.plot(a_values,period_a) 
plt.xlabel('Amplitude (m)')
plt.ylabel('T (s)')
plt.savefig('3.png')
plt.show()


# In[4]:


def H(n,x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*H(n-1,x)-2*(n-1)*H(n-2,x)

    
def wave_func(x,n):
    coef = 1/np.sqrt(factorial(n)*np.sqrt(np.pi)*(2**n))
    return coef*np.exp((-x**2)/2)*H(n,x)

x_i = -4
x_f = 4
delta_x = 0.01
x_values = np.arange(x_i,x_f+delta_x,delta_x)
wave_func_x_0 = wave_func(x_values,0)
wave_func_x_1 = wave_func(x_values,1)
wave_func_x_2 = wave_func(x_values,2)
wave_func_x_3 = wave_func(x_values,3)

    
plt.plot(x_values,wave_func_x_0,label = 'n = 0') 
plt.plot(x_values,wave_func_x_1,label = 'n = 1') 
plt.plot(x_values,wave_func_x_2,label = 'n = 2') 
plt.plot(x_values,wave_func_x_3,label = 'n = 3')
plt.legend()
plt.xlabel('x')
plt.ylabel('Wave Function')
plt.savefig('4.png')
plt.show()


# In[5]:


x_ic = -10
x_fc = 10
delta_xc = 0.01
x_valuesc = np.arange(x_ic,x_fc+delta_xc,delta_xc)
wave_func_x_30 = wave_func(x_valuesc,30)
    
    
plt.plot(x_valuesc,wave_func_x_30,label = 'n = 30') 
plt.legend()
plt.xlabel('x')
plt.ylabel('Wave Function')
plt.savefig('5.png')
plt.show()


# In[6]:


# Gaussian Quadrature
a = -1
b = 1
N = 100
z,w = gaussxwab(N,a,b)
s = 0.
for k in np.arange(N):
    s += w[k]*((1 + z[k]**2)/(1-z[k]**2)**2)*((wave_func((z[k]/(1-z[k]**2)),5))**2)*((z[k]/(1-z[k]**2))**2)
print(np.sqrt(s))


# In[7]:


# numpy used to get sample points and weights for Gauss-Hermite Quadrature
N = 80

x,w = np.polynomial.hermite.hermgauss(N)
s = 0.
for k in np.arange(N):
    s += w[k]*(x[k]**2)*H(5,x[k])**2

    
print(np.sqrt(s/(factorial(5)*np.sqrt(np.pi)*(2**5))))


# In[8]:


# scipy used to get sample points and weights for Gauss-Hermite Quadrature
N = 80

x,w = sp.special.roots_hermite(N)
s = 0.
for k in np.arange(N):
    s += w[k]*(x[k]**2)*H(5,x[k])**2

    
print(np.sqrt(s/(factorial(5)*np.sqrt(np.pi)*(2**5))))


# In[ ]:




