#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def parabolic_step(func=None, a=None, b=None, c=None):
    """returns the minimum of the function as approximated by a parabola"""
    fa = func(a)
    fb = func(b)
    fc = func(c)
    denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b - c)**2 * (fb - fa)
    # If singular, just return b 
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)

def func(x):
    return((x-0.3)**2 * np.exp(x))

def func_2(x):
    return((x-1)*(x-3))

xgrid = np.arange(-0.5,1,0.01)
plt.plot(xgrid,func(xgrid))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('1.png')
plt.show()


# In[3]:


def optimize():
    gsection = (3. - np.sqrt(5)) / 2
    #define interval
    a = -1 
    b = 1.1
    c = 4
    tol = 1e-7

    flag = True
    
    err = abs(c-a)
    err_list, b_list = [err], [b]
    while err > tol:
        s = parabolic_step(func=func, a=a, b=b, c=c)
        if ((s >= b))\
            or ((flag == True) and (abs(s-c) >= abs(c-b)))\
            or ((flag == False) and (abs(s-c) >= abs(b-d))):
            if((b - a) > (c - b)):
                x = b
                b = b - gsection * (b - a)
            else:
                x = b + gsection * (c - b)
            fb = func(b)
            fx = func(x)
            if(fb < fx):
                c = x
            else:
                a = b
                b = x 
            flag = True
        else:
            flag = False
        err = abs(c-a) #update error to check for convergence 
        err_list.append(err)
        b_list.append(b)
    print(f'minimum = {b}')
    return b_list, err_list

def plot(b_list, err_list):
    log_err = [np.log10(err) for err in err_list]
    fig, axs = plt.subplots(2,1, sharex=True)
    ax0, ax1 = axs[0], axs[1]
    #plot root
    ax0.scatter(range(len(b_list)), b_list, marker = 'o', facecolor = 'red', edgecolor = 'k')
    ax0.plot(range(len(b_list)), b_list, 'r-', alpha = .5)
    ax1.plot(range(len(err_list)), log_err,'.-')
    ax1.set_xlabel('number of iterations')
    ax0.set_ylabel(r'$x_{min}$')
    ax1.set_ylabel(r'$\log{\delta}$')
    plt.savefig('convergence.png')
    
if __name__ == "__main__":
    b_list, err_list = optimize()
    plot(b_list, err_list)


# In[13]:


from scipy import optimize
def f(x):
    return((x-0.3)**2 * np.exp(x))

minimizer = optimize.brent(f)
print(minimizer)


# In[6]:


def p(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))


# In[7]:


x = 50

beta_0 = np.linspace(-5,5, 100)
beta_1 = np.linspace(-5,5,100)
beta = np.meshgrid(beta_0, beta_1)
p_grid = p(x, *beta)
plt.pcolormesh(*beta, p_grid)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$p(y_i|x_i=50,\beta_0, \beta_1)$', fontsize = 16)
plt.savefig('3.png')


# In[8]:


import pandas as pd
data = pd.read_csv('survey.csv')  
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xs = xs[x_sort]
ys = ys[x_sort]


# In[9]:


def log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return ll # return log likelihood


# In[14]:


ll = log_likelihood(beta, xs, ys)
plt.pcolormesh(*beta, ll)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.savefig('4.png')


# In[12]:


from scipy import optimize
def negative_log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -1*ll # return log likelihood



xdata = xs
ydata = ys

# Covariance matrix of parameters
def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

#Error of parameters
def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))

result = optimize.minimize(fun = lambda beta, xs, ys: negative_log_likelihood(beta, xs, ys), x0 = np.array([0,0]),args = (xdata, ydata))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(ydata)-len(np.array([0,0]))) 
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))

print(result['x'][1])


# In[ ]:


plt.plot(xs,ys,'.',label='Data')
plt.plot(xs,p(xs,result['x'][0],result['x'][1]),label='Logistic Model')
plt.xlabel('x (age)')
plt.ylabel('p(x)')
plt.legend()
plt.savefig('2.png')
plt.show()


# In[ ]:


grad_ll_arr = np.gradient(ll)


# In[ ]:


plt.pcolormesh(*beta, grad_ll_arr[0])
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\frac{\partial}{\partial \beta_0} \mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.colorbar()


# In[ ]:


plt.pcolormesh(*beta, grad_ll_arr[1])
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\frac{\partial}{\partial \beta_1} \mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.colorbar()


# In[ ]:


def hessian(x):
    """
    https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


# In[ ]:


hess_ll = hessian(ll)


# In[ ]:




