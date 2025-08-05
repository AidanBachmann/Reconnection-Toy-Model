# Description: Main function. Define simulation parameters here, run for simulation.

# ---------- Imports ----------

import numpy as np
import time
import Functions as funcs

# ---------- Simulation Parameters ----------

dir = 'Fields/' # Directory to save figures
showFields = False # Plot magnetic field and vector potential

## Defining Coefficients ##
usePhysicalUnits = False # True --> Use SI units for physical constants, False --> Set constants to 1
J = float(1) # Magnitude of current
Bz = float(1) # Axial magnetic field
d = 4 # Length scale of domain, 4

if usePhysicalUnits:
    e = 1.6e-19 # Charge in Coulombs
    mu0 = 1.257e-6 # Permiability of free space
    coeff = (pow(e,2)*pow(mu0,2)*pow(J,2))/(8*np.pi) # Leading coefficient of equations of motion
    tau = 50e-6 # Reconnection timescale
    w = (2*np.pi)/(tau) # Frequency 
    a = 0.5*0.03 + 0.5*0.076 # x position of line currents in meters
else:
    coeff = J # Setting all united quantities to 1, leading coefficient of equations of motion is just current magnitude
    tau = 2 # Reconnection timescale
    w = (2*np.pi)/(tau) # Frequency 
    a = 1.0 # x position of line currents

## Current Profile Settings ##
currentProfile = 'sine' # Current profile. Options are currently: 'sine', 'square'. Add more in Functions.py.
if currentProfile == 'square':
    dutyCycle = 0.75 # Duty cycle for square wave current
    args = np.asarray([w,dutyCycle]) # Arguments array for current profile function
elif currentProfile == 'sine':
    phi = 0 # Phase shift for sinusoid
    args = np.asarray([w,phi]) # Arguments array for current profile function
else:
    args = None

## Simulation Time ##
dt = float(1e-4) # Time step, float() casts to C++ size double
t0 = float(0) # Initial time
N = int(1e3) # Number of time steps
tArr = np.linspace(t0,t0+N*dt,N+1,dtype='double') # Define time array
currentArr = funcs.currentProf(tArr,currentProfile,args).astype('double') # Compute current profile

## Initial Conditions ##
r0 = np.asarray([a,1e-3,0],dtype='double') # Particle initial position
v0 = np.asarray([1,1,0.5],dtype='double') # Particle initial velocity

funcs.printParams(dt,tau,t0,N,currentProfile,Bz,J,r0,v0) # Print simulation parameters to terminal

# ---------- Simulation ----------

print('Starting simulation...')
start = time.time()
r,v,B = funcs.integrateBoris(r0,v0,dt,Bz,N,a,J,currentArr) # Integrate trajectory
normB = funcs.computeBNorm(B) # Compute field magnitude along trajectory
end = time.time()
print(f'Finished simulation in {end-start} seconds.\n')

#funcs.plotJ(tArr,currentArr,normB) # Plot results
#funcs.plot3D(r,v,normB,a)
funcs.plot2D(r,v,normB,N,a,J,tArr)
#funcs.plotE(v,tArr,normB)

if showFields: # Plot magnetic field and vector potential
    funcs.showFields(a,d,J)


# ---------- Multiple Trajectories ----------

'''
M = 10 #Number of trajectories to integrate
r0 = np.asarray([-EOM.a/2,-EOM.a/2,0])
v0 = np.asarray([1e-2,1e-2,0])
dr = EOM.a/10
dv = 0

_r0,_v0 = initICs(M,r0,dr,v0,dv)
integrateSeries(_r0,_v0,t0,dt,Bz,N,M)
'''