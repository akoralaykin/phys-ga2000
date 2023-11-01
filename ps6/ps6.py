#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.io import fits
# documentation: see https://docs.astropy.org/en/stable/io/fits/
import matplotlib.pyplot as plt
import numpy as np
hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data


# In[2]:


print(flux.shape)
plt.plot(logwave, flux[0, :])
plt.plot(logwave, flux[1, :])
plt.plot(logwave, flux[2, :])
plt.plot(logwave, flux[3, :])
plt.plot(logwave, flux[4, :])
plt.ylabel('flux [$10^{−17}$ erg s$^{−1}$ cm$^{−2}$ A$^{-1}$]', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.savefig('1.png')
plt.show()


# In[3]:


# find normalization over wavelength for each galaxy
flux_sum = np.sum(flux, axis = 1)
flux_normalized = flux/np.tile(flux_sum, (np.shape(flux)[1], 1)).T

# check that the data is properly "normalized"
plt.plot(np.sum(flux_normalized, axis = 1))
plt.ylabel('Sum over each wavelength', fontsize = 16)
plt.xlabel('Corresponding galaxy', fontsize = 16)
plt.ylim(0,2)
plt.savefig('2.png')
plt.show()


# In[4]:


# subtract off mean
means_normalized = np.mean(flux_normalized, axis=1)
flux_normalized_0_mean = flux_normalized-np.tile(means_normalized, (np.shape(flux)[1], 1)).T
plt.plot(logwave, flux_normalized_0_mean[0,:]*(10**4))
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.ylabel('normalized 0-mean flux x10$^{-4}$', fontsize = 16)
plt.savefig('3.png')
plt.show()


# In[5]:


def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec


# In[21]:


r = flux_normalized_0_mean 
C = r.T@r # correlation matrix, dimension # wavelengths x # wavelengths
Uc, Sc, Vhc = np.linalg.svd(C, full_matrices=True)
print("Condition Number C =",np.max(Sc)/np.min(Sc))


# In[7]:


eigvals, eigvecs = sorted_eigs(r, return_eigvalues = True)


# In[8]:


U, S, Vh = np.linalg.svd(r, full_matrices=True)
print("Condition Number R =",np.max(S)/np.min(S))
# rows of Vh are eigenvectors
eigvecs_svd = Vh.T
eigvals_svd = S**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]


# In[9]:


plt.plot(logwave, eigvecs_svd[:,0],label = '1st')
plt.plot(logwave, eigvecs_svd[:,1],label = '2nd')
plt.plot(logwave, eigvecs_svd[:,2],label = '3rd')
plt.plot(logwave, eigvecs_svd[:,3],label = '4th')
plt.plot(logwave, eigvecs_svd[:,4],label = '5th')
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.ylabel('Eigenvectors', fontsize = 16)
plt.legend()
plt.title('SVD')
plt.show()


plt.plot(logwave, eigvecs[:,0],label = '1st')
plt.plot(logwave, eigvecs[:,1],label = '2nd')
plt.plot(logwave, eigvecs[:,2],label = '3rd')
plt.plot(logwave, eigvecs[:,3],label = '4th')
plt.plot(logwave, eigvecs[:,4],label = '5th')
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.ylabel('Eigenvectors', fontsize = 16)
plt.legend()
plt.title('EIG')
plt.savefig('4.png')
plt.show()


# In[10]:


[plt.plot(eigvecs_svd[:,i], eigvecs[:,i], 'o')for i in np.arange(100)]
plt.plot(np.linspace(-0.2, 0.2), np.linspace(-0.2, 0.2))
plt.xlabel('SVD eigenvectors', fontsize = 16)
plt.ylabel('Eig eigenvectors', fontsize = 16)
plt.savefig('5.png')
plt.show()

plt.plot(eigvals_svd, eigvals, 'o')
plt.xlabel('SVD eigenvalues', fontsize = 16)
plt.ylabel('Eig eigenvalues', fontsize = 16)
plt.savefig('6.png')
plt.show()


# In[11]:


def PCA(l, r, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum


# In[38]:


pca_5 = PCA(5,r)

plt.plot(logwave, pca_5[0,:], label = 'l = 5')
plt.plot(logwave, r[0,:], label = 'original data')
plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.legend()
plt.savefig('7.png')
plt.show()


# In[13]:


pca_5_c = PCA(5,r,project=False)


# In[55]:


c0 = pca_5_c[:,0]
c1 = pca_5_c[:,1]
c2 = pca_5_c[:,2]
plt.plot(c0,c1,'.')
plt.xlabel('c0',fontsize = 16)
plt.ylabel('c1',fontsize = 16)
plt.savefig('8.png')
plt.show()
plt.plot(c0,c2,'.')
plt.xlabel('c0',fontsize = 16)
plt.ylabel('c2',fontsize = 16)
plt.savefig('9.png')
plt.show()


# In[44]:


res = np.zeros((20,4001))
for i in np.arange(1,21):
    res[i-1,:] = (PCA(i,r)[0,:] - r[0,:])/r[0,:]


# In[54]:


plt.plot(logwave,res[0,:]**2,'.',label='Nc=1')
plt.plot(logwave,res[1,:]**2,'.',label='Nc=2')
plt.plot(logwave,res[19,:]**2,'.',label='Nc=20')
plt.ylabel('Square of the fractional residuals', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.legend()
plt.savefig('10.png')
plt.show()
print(res[19,:])


# In[ ]:




