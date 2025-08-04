# Description: Definitions of equations of motion and field time series visualization, to be used in Boris.py.

# ---------- Imports ----------

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# ---------- Units ----------

usePhysicalUnits = False # Use SI units for physical constants
factor = 2 # Multiplier for tau
J = 10 # Magnitude of current
d = 4 # Length scale of domain

if usePhysicalUnits:
    e = 1.6e-19 #Charge in C
    mu0 = 12.57e-7 #Permiability of free space
    _coeff = (pow(e,2)*pow(mu0,2)*pow(J,2))/(8*np.pi) #Leading coefficient of equations of motion
    
    tau = 50e-6 #Reconnection timescale
    _w = (2*np.pi)/(factor*tau) #Frequency 
    
    a = 0.5*0.03 + 0.5*0.076 # x position of line currents in meters
else:
    _coeff = J #Setting all united quantities to 1
    tau = 1 #Reconnection timescale
    #tau = 50e-6
    _w = (2*np.pi)/(factor*tau) #Frequency 
    _phi = 0
    
    a = 1 # x position of line currents
    
dutyCycle = 0.75 # Duty cycle for square wave current, 0.9

# ---------- Visualization Parameters ----------

N = 101 # Number of grid points
lb = -d/2 # Domain Lower bound
ub = d/2 # Domain upper bound
n = N-1 # Number of points minus 1, useful for indexing
dn = n/ub # Points per unit length

X = np.linspace(lb,ub,N) # Create array for meshgrid based on parameters
x,y = np.meshgrid(X,X) # Initialize meshgrid
sign = 1 # Controls sign of the current

useNorm = True # Use normalized arrows on quiver plot
threshold = 0 # Threshold for where to set A,B = 0 (0.99), used to avoid singularities

currentProfile = 'square' # Current profile. Options are: 'sine', 'square'
dir = 'Fields/' # Directory to save figures

# ---------- Equations of Motion ----------

if currentProfile == 'sine': # Set current profile
    def f(t,w=_w,phi=_phi): # Sinusoidal profile
        return np.cos(w*t+phi)
elif currentProfile == 'square':
    def f(t,w=_w,factor=factor,threshold=threshold): # Square wave
        tau = (2*np.pi)/(factor*w)
        t = t%tau
        if t < dutyCycle*tau: # Returns +/-J based on duty cycle. Use dutyCycle = 0.5 for regular square wave
            return 1
        else:
            return -1

def pxDot(x,y,t,a=a,f=f,coeff=_coeff): # Equation of motion in x
    return -1*coeff*( f(t)*np.log(pow((x-a),2) + pow(y,2)) + np.log(pow((x+a),2) + pow(y,2)) )*( (f(t)*(x-a))/(pow(x-a,2) + pow(y,2)) + (x+a)/(pow(x+a,2) + pow(y,2)) )
    
def pyDot(x,y,t,a=a,f=f,coeff=_coeff): # Equation of motion in y
    return -1*coeff*y*( f(t)*np.log(pow((x-a),2) + pow(y,2)) + np.log(pow((x+a),2) + pow(y,2)) )*( f(t)/(pow(x-a,2) + pow(y,2)) + 1/(pow(x+a,2) + pow(y,2)) )

# ---------- Generate A and B Time Series ----------

def normalize(x,y): # Normalize vector field
    x /= np.sqrt(x**2 + y**2)
    y /= np.sqrt(x**2 + y**2)
    return x,y

def computeTimeSeries(x,y,t,computeQ,a=a,f=f): # Compute time series of quantity (computeQ is a function that represents the equation for some quantity Q)
    array = np.empty([len(t)],dtype='object') # Initialize array to store results
    for i in np.linspace(0,len(t)-1,len(t),dtype='int'):
        array[i] = computeQ(x,y,a,f(t[i])) # Compute quantity for len(t) time steps
    return array
        
def generateTimeSeries(x,y,t0,dt,N,computeQ): # Generate the time series for a quantity Q
    t = np.linspace(t0,t0+(N-1)*dt,N) # Generate time data
    return computeTimeSeries(x,y,t,computeQ),t # Compute time evolution of quantity

def plotBTimeSeries(x,y,_Bx,_By,t,Scale,dir=dir): # Plot magnetic field time series
    plt.figure(figsize=(12,10))
    for i in np.linspace(0,len(t)-1,len(t),dtype='int'):
        Bx,By = normalize(_Bx[i],_By[i])
        quiv = plt.quiver(x,y,Bx,By,scale=Scale) # Plot magnetic field
        plt.title(f'B Field @ t = {t[i]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('step'+"{:05d}".format(i+1)+'.png') # Save figure
        quiv.remove() # Remove quiver plot from figure

# ---------- Computation and Visualization of Fields ----------

@jit(nopython = True)
def computeA(x,y,a,f,J=J): # Compute vector potential (f indicates sign of variable current)
    return f*np.log(np.sqrt(pow(x-a,2)+pow(y,2))) + np.log(np.sqrt(pow(x+a,2)+pow(y,2)))

@jit(nopython = True)
def computeBx(x,y,a,f,J=J): # Compute x component of magnetic field
    return y*J*((f/(pow(x-a,2)+pow(y,2)))+(1/(pow(x+a,2)+pow(y,2))))

@jit(nopython = True)
def computeBy(x,y,a,f,J=J): # Compute y component of magnetic field
    return -J*(((f*(x-a))/(pow(x-a,2)+pow(y,2)))+((x+a)/(pow(x+a,2)+pow(y,2))))

@jit(nopython = True)
def computeNormB(x,y,t,f=f,a=a): # Compute the norm of B
    return np.sqrt(pow(computeBx(x,y,a,f(t)),2) + pow(computeBy(x,y,a,f(t)),2))

def computeAll(x,y,a,f,n,N,func,xMin,yMin,threshold): # Compute A, Bx, or By over whole grid
    v = np.zeros([x.shape[0],y.shape[0]])
    for i in np.linspace(0,n,N,dtype='int'):
        for j in np.linspace(0,n,N,dtype='int'):
            if (abs(x[i,j] - a) < xMin + threshold) and (abs(y[i,j]) < yMin + threshold):
                v[i,j] = 0
            else:
                v[i,j] = func(x[i,j],y[i,j],a,f)
    return v

def findClosestGridPoint(a,X,n): # Find grid point closest to wire location
    dOld = np.abs(X[0] - a)
    d = dOld
    for i in np.linspace(1,n,n,dtype='int'):
        dNew = np.abs(X[i] - a)
        if dNew < dOld:
            d = dNew
        dOld = dNew
    return d  

def showFields(x=x,y=y,a=a,f=sign,n=n,N=N,dn=dn,useNorm=useNorm): 
    xMin = findClosestGridPoint(a,X,n)
    yMin = 1/dn
    
    tA = computeAll(x,y,a,sign,n,N,computeA,xMin,yMin,threshold) #Tube vector potential
    rA = computeAll(x,y,a,-sign,n,N,computeA,xMin,yMin,threshold) #Reconnection vector potential

    tBx = computeAll(x,y,a,sign,n,N,computeBx,xMin,yMin,threshold) #Tube magnetic field
    tBy = computeAll(x,y,a,sign,n,N,computeBy,xMin,yMin,threshold)

    rBx = computeAll(x,y,a,-sign,n,N,computeBx,xMin,yMin,threshold) #Reconnection magnetic field
    rBy = computeAll(x,y,a,-sign,n,N,computeBy,xMin,yMin,threshold)

    size = 10 #Size of points that denote wire locations

    # Vector potential plotting
    
    plt.figure(1,figsize=(12,10)) 
    plt.imshow(tA/np.max(tA))
    plt.scatter((-a*dn)/2+n/2,n/2,c='r',s=size,label='J//z')
    plt.scatter((a*dn)/2+n/2,n/2,c='r',s=size,label='J//z')
    plt.title('Normalized Az for J1 and J2 out of page (Separate flux tubes)')
    plt.xlabel('x')
    plt.ylabel('y')
    c = plt.colorbar()
    c.set_label('Az',rotation=360)
    plt.legend()

    plt.figure(2,figsize=(12,10))
    #plt.imshow(rA/np.max(rA))
    plt.imshow(rA)
    plt.scatter((-a*dn)/2+n/2,n/2,c='r',s=size,label='J//z')
    plt.scatter((a*dn)/2+n/2,n/2,c='black',s=size,label='J//-z')
    plt.title('Normalized Az for J1 out of page, J2 into page (reconnection)')
    plt.xlabel('x')
    plt.ylabel('y')
    c = plt.colorbar()
    c.set_label('Az',rotation=360)
    plt.legend()

    # Magnetic field plotting
    
    Scale = 65
    
    if useNorm == True:
        tBx,tBy = normalize(tBx,tBy)
        rBx,rBy = normalize(rBx,rBy)

    plt.figure(3,figsize=(12,10))
    plt.quiver(x,y,tBx,tBy,scale=Scale)
    plt.scatter(-a,0,c='r',label='J//z')
    plt.scatter(a,0,c='r',label='J//z')
    if useNorm == True:
        plt.title('Normalized magnetic field of flux tubes with parallel current')
    else:
        plt.title('Magnetic field of flux tubes with parallel current')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.figure(4,figsize=(12,10))
    plt.quiver(x,y,rBx,rBy,scale=Scale)
    plt.scatter(-a,0,c='r',label='J//z')
    plt.scatter(a,0,c='green',label='J//-z')
    if useNorm:
        plt.title('Normalized magnetic field of flux tubes with anti-parallel current (reconnection)')
    else:
        plt.title('Magnetic field of flux tubes with anti-parallel current (reconnection)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()