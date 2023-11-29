#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp


# In[2]:


import pandas as pd

piano = pd.read_csv('piano.txt', header = None).to_numpy()
trumpet = pd.read_csv('trumpet.txt', header = None).to_numpy()
pianodata = np.ndarray.flatten(piano)
trumpetdata = np.ndarray.flatten(trumpet)


# In[13]:


plt.plot(np.arange(0, len(piano))/44100, pianodata)
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title('Piano')
plt.savefig('1.png',bbox_inches='tight')
plt.show()


# In[12]:


plt.plot(np.arange(0, len(trumpet))/44100, trumpet)
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title('Trumpet')
plt.savefig('2.png',bbox_inches='tight')
plt.show()


# In[5]:


data =  pianodata
t = np.arange(0,len(pianodata))/44100

N = len(data)

# sample spacing

T = t[1]-t[0]

yf = fft(data)

xf = fftfreq(N, T)[:N//2]

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

plt.grid()
yplot = 2.0/N * np.abs(yf[0:N//2])
arg = np.argmax(yplot)

plt.vlines(xf[arg], 0, yplot[arg], color = 'red')
#plt.vlines(3, 0, 1, color = 'red')

print(xf[arg])


plt.xlabel('frequency [Hz]')
plt.ylabel('magnitude')
plt.savefig('3.png')
plt.title('Piano')
plt.show()


# In[6]:


data =  trumpetdata
t = np.arange(0,len(trumpetdata))/44100

N = len(data)

# sample spacing

T = t[1]-t[0]

yf = fft(data)

xf = fftfreq(N, T)[:N//2]

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

plt.grid()
yplot = 2.0/N * np.abs(yf[0:N//2])
arg = np.argmax(yplot)

plt.vlines(xf[arg], 0, yplot[arg], color = 'red')
#plt.vlines(3, 0, 1, color = 'red')

print(xf[arg])


plt.xlabel('frequency [Hz]')
plt.ylabel('magnitude')
plt.savefig('4.png')
plt.title('Trumpet')
plt.show()


# In[7]:


def derivative_lorenz(t,r):
 
    sigma_ = 10
    r_=28
    b_=8/3
    x = r[0]
    y = r[1]
    z = r[2]
    fx = sigma_*(y-x)
    fy = r_*x - y - x*z
    fz = x*y - b_*z
    return [fx, fy, fz] 


# In[8]:


def numerical_traj_ex(t_span, y0, t):

    sol4 = solve_ivp(derivative_lorenz, t_span, y0, t_eval = t,  \
                     method = 'LSODA')
    # Radau and LSODA yield the same results

    t = sol4.t
    y = sol4.y
    x = y[0,:]
    yy = y[1,:]
    z = y[2,:]
    return t, yy, x, z


# In[9]:


exp_fps = 10000 # samples per second
t_span = [0, 50] # simulate system for 3 seconds
t = np.arange(*t_span, 1/exp_fps)

y0 = [0,1,0]
t, yy, x, z = numerical_traj_ex(t_span, y0, t)


# In[10]:


# plot the trajectory
plt.plot(t,yy)
plt.ylabel('y', fontsize = 12)
plt.xlabel('t', fontsize = 12)
plt.savefig('5.png')
plt.show()


# In[11]:


plt.plot(x,z)
plt.ylabel('z', fontsize = 12)
plt.xlabel('x', fontsize = 12)
plt.savefig('6.png')
plt.show()


# In[ ]:





# In[ ]:




