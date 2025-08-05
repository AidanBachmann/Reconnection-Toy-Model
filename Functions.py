# Description: Function definitions for reconnection toy model.

# ---------- Imports ----------

import numpy as np
from scipy import signal
from numba import jit
import matplotlib.pyplot as plt

# ---------- Debugging ----------

def printParams(dt,tau,t0,N,current,Bz,J): # Print simulation parameters to terminal
    print(f'\033[1mSimulation Parameters\033[0m\nTime step: {dt}s\nNumber of Time Steps: {N}\nInitial Time: {t0}s\nFinal Time: {t0+(N+1)*dt}s')
    print(f'Reconnection Time Scale: {tau}s\nCurrent Profile: {current}\nAxial Magnetic Field: {Bz}\nCurrent Magnitude: {J}\n')

# ---------- Visualization Parameters ----------

def defineGrid(d,theshold=0,N=101,useNorm=True): # Define meshgrid parameters for visualizing fields
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

# ---------- Current Profile ----------
    
def currentProf(t,profileType,args): # Return current profile; t is time array, profileType is a string ('square' or 'sine'), and args as an array of parameters for the profile
    if profileType == 'square':
        # args must by an array of the form args = np.asarray([w,dutyCycle]), define in main
        w = args[0]
        dutyCycle = args[1]
        return 2*signal.square(2*np.pi*w*t,duty=dutyCycle) - 1 # Square wave with frequency 2πω and duty cycle dutyCycle, extends from -1 to 1
    elif profileType == 'sine':
        # args must by an array of the form args = np.asarray([w,phi]), define in main
        w = args[0]
        phi = args[1]
        return np.cos(w*t+phi) # Sinusoidal current profile with frequency ω and phase shift phi
        ## Add extra elif statements below to define new current profiles ##
    else:
        print('Unrecognized current profile, defaulting to constant.')
        return np.ones([len(t)]) # Constant current profile
    
