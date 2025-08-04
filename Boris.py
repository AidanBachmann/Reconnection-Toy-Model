# Description: Integrator of single particle trajectory in time varying magnetic field.

# ---------- Imports ----------

import numpy as np
import matplotlib.pyplot as plt

import EOM

# ---------- Boris Integration ----------

def computeStep(rn,vn,dt,tn,Bz,f=EOM.f,a=EOM.a): # Compute single time step using Boris integrator
    tau = dt/2 # Half time step
    B = np.asarray([EOM.computeBx(rn[0],rn[1],a,f(tn)),EOM.computeBy(rn[0],rn[1],a,f(tn)),Bz]) # Compute magnetic field at particle position
    t = B*tau # t vector
    s = 2*t/(1+np.dot(t,t)) # s vector
    vnew = vn + np.cross(vn + np.cross(vn,t),s) # Update velocity
    r = rn + vnew*dt # Update positio
    tn += dt # Iterate time step
    
    return r,vnew,B,tn
    
def integrateBoris(r0,v0,t0,dt,Bz,N,f=EOM.f,a=EOM.a): # Integrate trajectory for N time steps
    r,v,B = np.zeros([3,N+1]),np.zeros([3,N+1]),np.zeros([3,N+1]) # Initialize arrays for position, velocity, and magnetic field
    r[:,0],v[:,0],B[:,0] = r0,v0,np.asarray([EOM.computeBx(r0[0],r0[1],a,f(t0)),EOM.computeBy(r0[0],r0[1],a,f(t0)),Bz]) # Initial conditions
    t = t0 # Initial time
    for i in np.linspace(1,N,N,dtype='int'): # Evolve system for N time steps
        r[:,i],v[:,i],B[:,i],t = computeStep(r[:,i-1],v[:,i-1],dt,t,Bz)    
    return r,v,B

def computeBNorm(B,N): # Compute magnitude of magnetic field
    normB = np.zeros([N+1])
    for i in np.linspace(0,N,N+1,dtype='int'):
        normB[i] = np.sqrt(np.dot(B[:,i],B[:,i]))
    return normB

# ---------- Integrating Multiple Trajectories ----------

def initICs(M,r0,dr,v0,dv,dimr='x',dimv='x'): # Initialize initial conditions for a collection of simulations
    return initRVICs(M,r0,dr,dimr),initRVICs(M,v0,dv,dimv)

def initRVICs(M,r0,dr,dim):
    _r0 = np.zeros([3,M+1])
    _r0[:,0] = r0
    if dim == 'x':
        for i in np.linspace(1,M,M,dtype='int'):
            _r0[:,i] = np.asarray([_r0[0,i-1]+dr,_r0[1,i-1],_r0[2,i-1]])
    elif dim == 'y':
        for i in np.linspace(1,M,M,dtype='int'):
            _r0[:,i] = np.asarray([_r0[0,i-1],_r0[1,i-1]+dr,_r0[2,i-1]])
    elif dim == 'z':
        for i in np.linspace(1,M,M,dtype='int'):
            _r0[:,i] = np.asarray([_r0[0,i-1],_r0[1,i-1],_r0[2,i-1]+dr])
    return _r0

def integrateSeries(_r0,_v0,t0,dt,Bz,N,M,f=EOM.f,a=EOM.a): # Run simulations for a collection of initial conditions
    fig,ax = None,None
    for i in np.linspace(0,M,M+1,dtype='int'):
        r0,v0 = _r0[:,i],_v0[:,i]
        r,v,B = integrateBoris(r0,v0,t0,dt,Bz,N,f=EOM.f,a=EOM.a)
        normB = computeBNorm(B,N)
        fig,ax = plot2D(r,v,normB,fig,ax)
        plot3D(r,v,normB)

# ---------- Plotting ----------

def computeAutoScaling(x,y): # Auto scaling for axes (This function was stolen from Pierre's intro to computational plasma notes: https://gourdain.pas.rochester.edu/index.php/introduction-to-plasma-physics)
    xc = (x.max() + x.min())/2.
    x_low = xc - (x.max() - x.min())/2.*1.1
    x_high = xc + (x.max() - x.min())/2.*1.1
    yc = (y.max() + y.min())/2.
    y_low = yc - (y.max() - y.min())/2.*1.1
    y_high = yc + (y.max()-y.min())/2.*1.1
    
    return x_low,x_high,y_low,y_high

def plot2D(r,v,normB,N,fig=None,ax=None): # Make two dimensional phase space plots p_i versus r_i for i = x,y,z; plot for real-space particle trajectory flattened along z
    if (fig is None) and (ax is None):
        fig,ax = plt.subplots(2,2,figsize=(16,9.5)) # Define axes
    if N > 10000: # Set size of points for plotting
        size = 0.1
    else:
        size = 5
    
    J = EOM.J # Grab peak current
    a = EOM.a # Grab position of line current along x axis
    
    # All particle trajectories have a colormap set by the magnitude of the magnetic field at the location of the particle for the time step in question. 
    # This is to help understand why specific kinks or deflections occur in the trajectory (for example, locations where the field is large correspond 
    # to the most extreme bending of the particle trajectory).

    ax[0,0].scatter(r[0,0],v[0,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot x phase space
    c = ax[0,0].scatter(r[0,1:],v[0,1:],c=normB[1:],s=size)
    c = fig.colorbar(c,ax=ax[0,0])
    c.set_label('|B|',rotation=360)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel(r'$P_x$')
    ax[0,0].set_title(f'x Phase Space, J = {round(J,2)}')
    ax[0,0].grid()
    ax[0,0].legend()
    
    ax[0,1].scatter(r[1,0],v[1,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot y phase space
    c = ax[0,1].scatter(r[1,1:],v[1,1:],c=normB[1:],s=size)
    c = fig.colorbar(c,ax=ax[0,1])
    c.set_label('|B|',rotation=360)
    ax[0,1].set_xlabel('y')
    ax[0,1].set_ylabel(r'$P_y$')
    ax[0,1].set_title(f'y Phase Space, J = {round(J,2)}')
    ax[0,1].grid()
    ax[0,1].legend()
    
    ax[1,0].scatter(r[2,0],v[2,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot z phase space
    c = ax[1,0].scatter(r[2,1:],v[2,1:],c=normB[1:],s=size)
    c = fig.colorbar(c,ax=ax[1,0])
    c.set_label('|B|',rotation=360)
    ax[1,0].set_xlabel('z')
    ax[1,0].set_ylabel(r'$P_z$')
    ax[1,0].set_title(rf'z Phase Space, J = {round(J,2)}')
    ax[1,0].grid()
    ax[1,0].legend()
  
    ax[1,1].scatter(r[0,0],r[1,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot real-space trajectory flattened to xy plane
    c = ax[1,1].scatter(r[0,1:],r[1,1:],c=normB[1:],s=size)
    c = fig.colorbar(c,ax=ax[1,1])
    c.set_label('|B|',rotation=360)
    ax[1,1].scatter(-a,0,marker='+',c='b',s=50) # Draw position of lines currents in xy plane
    ax[1,1].scatter(a,0,marker='+',c='b',s=50,label='Line Current')
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    ax[1,1].set_title(f'Real Space Particle Trajectory, J = {round(J,2)}')
    ax[1,1].grid()
    ax[1,1].legend(loc=1)
    xmin,xmax,ymin,ymax = computeAutoScaling(r[0,1:],r[1,1:])
    ax[1,1].set_xlim(xmin=xmin,xmax=xmax)
    ax[1,1].set_ylim(ymin=ymin,ymax=ymax)
    plt.show()
    return fig,ax
    
def plot3D(r,v,normB,fig=None,ax=None): # Plot 3D real-space and momentum-space plots
    fig = plt.figure(figsize=(16,9.5)) # Set figure size
    size = 5 # Set size of points to plot
    J = EOM.J # Grab peak current
    a = EOM.a # Grab position of line current along x axis
    
    ax = fig.add_subplot(1,2,1,projection='3d') # Plot position space
    ax.scatter(r[0,0],r[1,0],r[2,0],c='r',marker='*',s=15*size,label='Initial Position') # Highlight initial position of particle
    aArr = -a*np.ones(2)
    zroArr = np.zeros(2)
    zArr = np.asarray([min(r[2,:]),max(r[2,:])])
    ax.plot(aArr,zroArr,zArr,c='b',label='Line Current') # Draw position of line currents
    ax.plot(-aArr,zroArr,zArr,c='b')
    c = ax.scatter(r[0,1:],r[1,1:],r[2,1:],c=normB[1:],s=size) # Plot trajectory
    c = fig.colorbar(c,ax=ax,fraction=0.046,pad=0.1)
    c.set_label('|B|',rotation=360)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Real Space Trajectory')
    ax.grid()
    ax.legend()
    margin = 1e-4
    ax.set_xlim(xmin=np.min(r[0,1:])-margin,xmax=np.max(r[0,1:])+margin)
    ax.set_ylim(ymin=np.min(r[1,1:])-margin,ymax=np.max(r[1,1:])+margin)
    ax.set_zlim(zmin=np.min(r[2,1:])-margin,zmax=np.max(r[2,1:])+margin)
    
    ax = fig.add_subplot(1,2,2,projection='3d') # Plot momentum space
    ax.scatter(v[0,0],v[1,0],v[2,0],c='r',marker='*',s=15*size,label='Initial Position') # Highlight initial position of particle
    c = ax.scatter(v[0,1:],v[1,1:],v[2,1:],c=normB[1:],s=size)
    c = fig.colorbar(c,ax=ax,fraction=0.046,pad=0.04)
    c.set_label('|B|',rotation=360)
    ax.set_xlabel(r'$P_x$')
    ax.set_ylabel(r'$P_y$')
    ax.set_zlabel(r'$P_z$')
    ax.set_title('3D Momentum Space Trajectory')
    ax.grid()
    ax.legend()
    plt.show()
    
def plotJ(t0,dt,N,normB): # Plot current profile
    time = np.zeros([N+1]) # Time array
    func = np.zeros([len(time)]) # Array to store current values
    time[0] = t0 # Get initial time
    func[0] = EOM.f(t0) # Compute curent at t0 (f is a current profile defined in EOM.py)
    t = t0 + dt
    for i in np.linspace(1,len(time)-1,len(time),dtype='int'): # Compute current profile along trajectory
        func[i] = EOM.f(t)
        time[i] = t
        t += dt
    plt.figure(figsize=(12,9.5))
    plt.plot(time,func)
    plt.scatter(time,func,c=normB,s=5) # Plot current profile
    plt.xlabel('Time (s)')
    plt.ylabel('Current')
    plt.title('Current Profile')
    plt.grid()
    plt.show()
    
def plotE(p,t0,dt,N,normB): # Plot total energy as a function of time, compute varaince to check for conservation
    E = np.zeros([p.shape[1]])
    time = np.linspace(t0,t0+(N+1)*dt,N+1)
    for i in np.linspace(0,len(E)-1,len(E),dtype='int'): # Compute total energy for each time step
        E[i] = np.sqrt( np.dot(p[:,i],p[:,i]) )
    var = str(np.var(E)) # Compute variance of energy
    length = len(var)
    plt.figure(figsize=(12,9.5))
    plt.plot(time,E,label=f'Variance of Energy = {var[0:6]+var[length-4:length]}') # Plot energy time series
    plt.scatter(time,E,c=normB) # Scatter plot on top of line to color with norm of B
    c = plt.colorbar()
    c.set_label('|B|',rotation=360) # Define colorbar
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy')
    plt.title('Time Evolution of Total Energy')
    plt.legend()
    plt.show()

# ---------- Simulation Parameters ----------

dt = 1e-4 # Time step, 1e-3
t0 = 0 # Initial time
N = 250000 # Number of time steps

print(f'Time step: {dt}s, Reconnection Time Scale: {EOM.tau}, Initial Time: {t0}s, Final Time: {t0+(N+1)*dt}s')

r0 = np.asarray([EOM.a,1e-3,0]) # Particle initial position
v0 = np.asarray([3,-1,1]) # Particle initial velocity

Bz = 1 # Axial magnetic field, 0.003

# ---------- Simulation ----------

r,v,B = integrateBoris(r0,v0,t0,dt,Bz,N) # Integrate trajectory
normB = computeBNorm(B,N) # Compute field magnitude along trajectory

plotJ(t0,dt,N,normB) # Plot results
plot3D(r,v,normB)
plot2D(r,v,normB,N)
plotE(v,t0,dt,N,normB)

# ---------- Multiple Trajectories ----------

'''M = 10 #Number of trajectories to integrate
r0 = np.asarray([-EOM.a/2,-EOM.a/2,0])
v0 = np.asarray([1e-2,1e-2,0])
dr = EOM.a/10
dv = 0

_r0,_v0 = initICs(M,r0,dr,v0,dv)'''

#integrateSeries(_r0,_v0,t0,dt,Bz,N,M)