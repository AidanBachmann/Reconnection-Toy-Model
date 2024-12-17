# Description: Visualize vector potential and magnetic field for parallel and antiparallel line currnents.

# ---------- Imports ----------

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# ---------- Functions ----------

@jit(nopython = True)
def computeA(x,y,a,f): # Compute vector potential. a is x position of line current, f is strength.
        return f*np.log(np.sqrt(pow(x-a,2)+pow(y,2))) + np.log(np.sqrt(pow(x+a,2)+pow(y,2)))

@jit(nopython = True)
def computeBx(x,y,a,f): # Compute x component of magnetic field
        return y*((f/(pow(x-a,2)+pow(y,2)))+(1/(pow(x+a,2)+pow(y,2))))

@jit(nopython = True)
def computeBy(x,y,a,f): # Compute y compute of magnetic field
        return -(((f*(x-a))/(pow(x-a,2)+pow(y,2)))+((x+a)/(pow(x+a,2)+pow(y,2))))
    
@jit(nopython = True)
def computeAll(x,y,a,f,n,N,func,xMin,yMin,threshold): # Compute fields across entire grid. Argument func specifies which quantity to compute (A, Bx, or By).
    v = np.zeros((x.shape[0],y.shape[0]))
    for i in np.linspace(0,n,N).astype('int'):
        for j in np.linspace(0,n,N).astype('int'):
            if (abs(x[i,j] - a) < xMin + threshold) and (abs(y[i,j]) < yMin + threshold): # This line is to avoid sigularities for too fine a grid
                v[i,j] = 0
            else:
                v[i,j] = func(x[i,j],y[i,j],a,f)
    return v

def normalize(x,y): # Normalize all magnetic field vectors to unit length
    norm = np.sqrt(pow(x,2) + pow(y,2))
    x /= norm
    y /= norm
    return x,y

def findClosestGridPoint(a,X,n): # Find grid point closest to position of line currents
    dOld = np.abs(X[0] - a)
    d = dOld
    for i in np.linspace(1,n,n,dtype='int'):
        dNew = np.abs(X[i] - a)
        if dNew < dOld:
            d = dNew
        dOld = dNew
    return d

# ---------- Main----------   

N = 101 # Number of grid points
lb = -10 # Lower bound
ub = 10 # Upper bound
n = N-1 # Number of points minus 1, useful for indexing
dn = n/ub # Points per unit length

X = np.linspace(lb,ub,N) # 1D position array
x,y = np.meshgrid(X,X) # Square meshgrid
a = 2.25 # Position of line currents along x axis
f = 1  # Current direction (+1 or -1) for time varying current

save = False # Save figure

useNorm = True # Use normalized arrows on quiver plot
threshold = 0 #Threshold for where to set A,B = 0, use to avoid singularities with a fine meshgrid

xMin = findClosestGridPoint(a,X,n) # Find grid point closest to line currents
yMin = 1/dn

tA = computeAll(x,y,a,f,n,N,computeA,xMin,yMin,threshold) # Vector potential, shear
rA = computeAll(x,y,a,-f,n,N,computeA,xMin,yMin,threshold) # Vector potential, mirror

tBx = computeAll(x,y,a,f,n,N,computeBx,xMin,yMin,threshold) # Magnetic field along x, shear
tBy = computeAll(x,y,a,f,n,N,computeBy,xMin,yMin,threshold) # Magnetic field along x, mirror

rBx = computeAll(x,y,a,-f,n,N,computeBx,xMin,yMin,threshold) # Magnetic field along y, shear
rBy = computeAll(x,y,a,-f,n,N,computeBy,xMin,yMin,threshold) # Magnetic field along y, mirror

if useNorm == True: # Normalize vectors
    tBx,tBy = normalize(tBx,tBy)
    rBx,rBy = normalize(rBx,rBy)

size = 10 # Size of points that denote wire locations

fig,ax = plt.subplots(2,2,figsize=(12,10))

ax[0,0].set_title('Az for Parallel Currents')
im = ax[0,0].imshow(tA/np.max(tA))
ax[0,0].scatter((-a*dn)/2+n/2,n/2,c='r',s=size,label=r'$J\hat{z}$')
ax[0,0].scatter((a*dn)/2+n/2,n/2,c='r',s=size,label=r'$J\hat{z}$')
fig.colorbar(im,ax=ax[0,0])
ax[0,0].legend()

ax[0,1].set_title('Az for Antiparallel Currents')
im = ax[0,1].imshow(rA/np.max(rA))
ax[0,1].scatter((-a*dn)/2+n/2,n/2,c='r',s=size,label=r'$J\hat{z}$')
ax[0,1].scatter((a*dn)/2+n/2,n/2,c='black',s=size,label=r'-$J\hat{z}$')
fig.colorbar(im,ax=ax[0,1])
ax[0,1].legend()

Scale = 65 # Scale of quiver arrows

ax[1,0].quiver(x,y,tBx,tBy,scale=Scale)
ax[1,0].scatter(-a,0,c='r',label=r'$J\hat{z}$')
ax[1,0].scatter(a,0,c='r',label=r'$J\hat{z}$')
ax[1,0].set_title('Magnetic Shear (Parallel Currnents)')
ax[1,0].legend()

ax[1,1].quiver(x,y,rBx,rBy,scale=Scale)
ax[1,1].scatter(-a,0,c='r',label=r'$J\hat{z}$')
ax[1,1].scatter(a,0,c='green',label=r'$-J\hat{z}$')
ax[1,1].set_title('Magnetic Mirror (Antiparallel Currents)')
ax[1,1].legend()

if save:
    plt.savefig('Fields.png',dpi=300)
else:
    plt.show()