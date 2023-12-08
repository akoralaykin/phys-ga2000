#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from IPython.display import HTML


# In[2]:


h = 1e-18*10
hbar = 1.0546e-36
L = 1e-8
M = 9.109e-31
N = 1000 # Grid slices

a = L/N

a1 = 1 + h*hbar/2/M/a**2*1j

a2 = -h*hbar*1j/4/M/a**2

b1 =  1 - h*hbar/2/M/a**2*1j

b2 =  h*hbar*1j/4/M/a**2

ksi = np.zeros(N+1,complex)


# In[3]:


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

aa1 = np.zeros(N+1,dtype=complex)
aa2 = np.zeros(N,dtype=complex)
bb1 = np.zeros(N+1,dtype=complex)
bb2 = np.zeros(N,dtype=complex)

aa1[:] = a1
aa2[:] = a2
bb1[:] = b1
bb2[:] = b2

A = tridiag(aa2, aa1, aa2)
B = tridiag(bb2, bb1, bb2)


# In[4]:


def ksi0(x):
    x0 = L/2
    sigma = 1e-10
    k = 5e10
    return np.exp(-(x-x0)**2/2/sigma**2)*np.exp(1j*k*x)


# In[5]:


x = np.linspace(0,L,N+1)
ksi[:] = ksi0(x)
ksi[[0,N]] = 0
tsteps = 10000
ksimatrix = np.zeros((ksi.size,tsteps+1),dtype=complex)
ksimatrix[:,0] = ksi


# In[6]:


for i in np.arange(0,tsteps):
    ksimatrix[:,i+1]= np.linalg.solve(A,np.dot(B,ksimatrix[:,i]))
    
plt.plot(x, np.absolute(ksimatrix[:,1]))


# In[16]:


figure, ax = plt.subplots()
 
# Setting limits for x and y axis
ax.set_xlim(-0.1e-8, 1.1e-8)
ax.set_ylim(-0.1, 1.1)
 
# Since plotting a single graph
line,  = ax.plot(0, 0) 
 
def animation_function(i):
   
 
    line.set_xdata(x)
    line.set_ydata(np.absolute(ksimatrix[:,i]))
    return line,
 
anim = animation.FuncAnimation(figure,
                          func = animation_function,
                          frames = np.arange(0, tsteps+1, 1), 
                          interval = 10)
HTML(anim.to_html5_video())
writervideo = animation.FFMpegWriter(fps=60) 
anim.save('norm.mp4', writer=writervideo) 
plt.close()


# In[17]:


figure, ax = plt.subplots()
 
# Setting limits for x and y axis
ax.set_xlim(-0.1e-8, 1.1e-8)
ax.set_ylim(-1.1, 1.1)

# Since plotting a single graph
line,  = ax.plot(0, 0) 
 
def animation_function(i):
   
 
    line.set_xdata(x)
    line.set_ydata(np.real(ksimatrix[:,i]))
    return line,
 
anim = FuncAnimation(figure,
                          func = animation_function,
                          frames = np.arange(0, tsteps+1, 1), 
                          interval = 10)
HTML(anim.to_html5_video())
writervideo = animation.FFMpegWriter(fps=60) 
anim.save('real.mp4', writer=writervideo) 
plt.close()


# In[42]:


t1 = 0*h
t2 = 1000*h
t3 = 2000*h
t4 = 3e-14
t5 = 4000*h
t6 = 5000*h
plt.plot(x, np.absolute(ksimatrix[:,0]),label=f't = {t1}s')
plt.plot(x, np.absolute(ksimatrix[:,1000]),label=f't = {t2}s')
plt.plot(x, np.absolute(ksimatrix[:,2000]),label=f't = {t3}s')
plt.plot(x, np.absolute(ksimatrix[:,3000]),label=f't = {t4}s')
plt.plot(x, np.absolute(ksimatrix[:,4000]),label=f't = {t5}s')
plt.plot(x, np.absolute(ksimatrix[:,5000]),label=f't = {t6}s')
plt.title('Norm as a function of time and x')
plt.xlabel('x (m)')
plt.ylabel('Norm of the wavefunction')
plt.legend()
plt.savefig('1.png')
plt.show()


# In[43]:


plt.plot(x, np.real(ksimatrix[:,0]),label=f't = {t1}s')
plt.plot(x, np.real(ksimatrix[:,1000]),label=f't = {t2}s')
plt.plot(x, np.real(ksimatrix[:,2000]),label=f't = {t3}s')
plt.plot(x, np.real(ksimatrix[:,3000]),label=f't = {t4}s')
plt.plot(x, np.real(ksimatrix[:,4000]),label=f't = {t5}s')
plt.plot(x, np.real(ksimatrix[:,5000]),label=f't = {t6}s')
plt.title('Real part as a function of time and x')
plt.xlabel('x (m)')
plt.ylabel('Real part of the wavefunction')
plt.legend()
plt.savefig('2.png')
plt.show()


# In[ ]:




