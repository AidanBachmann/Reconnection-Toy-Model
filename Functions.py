# Description: Function definitions for reconnection toy model.

# ---------- Imports ----------

import numpy as np
import matplotlib as mpl
from scipy import signal
from numba import jit
import matplotlib.pyplot as plt

# ---------- Debugging ----------

def printParams(dt,tau,t0,N,current,Bz,J,r0,v0): # Print simulation parameters to terminal (yes, this is unwieldy)
    print(f'\033[1mSimulation Parameters\033[0m\nTime step: {dt}s\nNumber of Time Steps: {N}\nInitial Time: {t0}s\nFinal Time: {t0+(N+1)*dt}s')
    print(f'Reconnection Time Scale: {tau}s\nCurrent Profile: {current}\nAxial Magnetic Field: {Bz}\nCurrent Magnitude: {J}')
    print(f'Initial Position: ({r0[0]},{r0[1]},{r0[2]})m\nInitial Velocity: ({v0[0]},{v0[1]},{v0[2]})m/s\n')

# ---------- Current Profile ----------
    
def currentProf(t,profileType,args): # Return current profile; t is time array, profileType is a string ('square' or 'sine'), and args as an array of parameters for the profile
    if profileType == 'square':
        # args must by an array of the form args = np.asarray([w,dutyCycle]), define in main
        w = args[0]
        dutyCycle = args[1]
        return signal.square(w*t,duty=dutyCycle).astype('double') # Square wave with frequency 2πω and duty cycle dutyCycle, extends from -1 to 1
    elif profileType == 'sine':
        # args must by an array of the form args = np.asarray([w,phi]), define in main
        w = args[0]
        phi = args[1]
        return np.cos(w*t+phi).astype('double') # Sinusoidal current profile with frequency ω and phase shift phi
        ## Add extra elif statements below to define new current profiles ##
    else:
        print('Unrecognized current profile, defaulting to constant.')
        return np.ones([len(t)]).astype('double') # Constant current profile
    
# ---------- Equations of Motion ----------
# These functions are deprecated, they were to be used in an old version with a leapfrog integrator. 
# I'll leave them here in case I want to add a leapfrog integrator in the future.
def pxDot(x,y,a,idx,current,coeff): # Equation of motion in x
    return -1*coeff*( current[idx]*np.log(pow((x-a),2) + pow(y,2)) + np.log(pow((x+a),2) + pow(y,2)) )*( (current[idx]*(x-a))/(pow(x-a,2) + pow(y,2)) + (x+a)/(pow(x+a,2) + pow(y,2)) )
    
def pyDot(x,y,a,idx,current,coeff): # Equation of motion in y
    return -1*coeff*y*( current[idx]*np.log(pow((x-a),2) + pow(y,2)) + np.log(pow((x+a),2) + pow(y,2)) )*( current[idx]/(pow(x-a,2) + pow(y,2)) + 1/(pow(x+a,2) + pow(y,2)) )

# ---------- Generate A and B Time Series ----------

def normalize(x,y): # Normalize vector field
    norm =  np.sqrt(pow(x,2) + pow(y,2))
    x /= norm
    y /= norm
    return x,y

def computeTimeSeries(x,y,t,computeQ,a,J,current): # Compute time series of quantity (computeQ is a function that represents the equation for some quantity Q)
    array = np.empty([len(t)],dtype='object') # Initialize array to store results
    for i in np.linspace(0,len(t)-1,len(t)).astype('int'):
        array[i] = computeQ(x,y,a,J,current[i]) # Compute quantity for len(t) time steps
    return array
        
def generateTimeSeries(x,y,t,a,J,current,computeQ): # Generate the time series for a quantity Q
    return computeTimeSeries(x,y,t,computeQ,a,J,current) # Compute time evolution of quantity

def plotBTimeSeries(x,y,_Bx,_By,t,Scale,dir): # Plot magnetic field time series
    plt.figure(figsize=(12,10))
    for i in np.linspace(0,len(t)-1,len(t),dtype='int'):
        Bx,By = normalize(_Bx[i],_By[i])
        quiv = plt.quiver(x,y,Bx,By,scale=Scale) # Plot magnetic field
        plt.title(f'B Field @ t = {t[i]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('step'+"{:05d}".format(i+1)+'.png') # Save figure
        quiv.remove() # Remove quiver plot from figure

# ---------- Computation of Fields ----------

# Below, f = f(t) is the time-dependence of the current. So, for a sinusoid, I = J*f(t) = J*sin(wt + phi).

@jit(nopython = True)
def computeA(x,y,a,J,f): # Compute vector potential (f indicates sign of variable current)
    return J*f*np.log(np.sqrt(pow(x-a,2)+pow(y,2))) + J*np.log(np.sqrt(pow(x+a,2)+pow(y,2)))

@jit(nopython = True)
def computeBx(x,y,a,J,f): # Compute x component of magnetic field
    return y*J*((f/(pow(x-a,2)+pow(y,2)))+(1/(pow(x+a,2)+pow(y,2))))

@jit(nopython = True)
def computeBy(x,y,a,J,f): # Compute y component of magnetic field
    return -J*(((f*(x-a))/(pow(x-a,2)+pow(y,2)))+((x+a)/(pow(x+a,2)+pow(y,2))))

@jit(nopython = True)
def computeNormB(x,y,a,J,f): # Compute the norm of B
    return np.sqrt(pow(computeBx(x,y,a,J,f),2) + pow(computeBy(x,y,a,J,f),2))

def computeAll(x,y,a,J,f,n,N,func,xMin,yMin,threshold): # Compute A, Bx, or By over whole grid
    v = np.zeros((x.shape[0],y.shape[0]))
    for i in np.linspace(0,n,N,dtype='int'):
        for j in np.linspace(0,n,N,dtype='int'):
            if (abs(abs(x[i,j]) - a) <= xMin + threshold) and (abs(y[i,j]) < yMin + threshold):
                v[i,j] = 0
            else:
                v[i,j] = func(x[i,j],y[i,j],a,J,f)
    return v

def findClosestGridPoint(a,X,n): # Find grid point closest to wire location
    dOld = np.abs(X[0] - a)
    d = dOld
    for i in np.linspace(1,n,n).astype('int'):
        dNew = np.abs(X[i] - a)
        if dNew < dOld:
            d = dNew
        dOld = dNew
    return d
    
# ---------- Boris Integration ----------

@jit(nopython = True)
def computeStep(rn,vn,dt,Bz,a,J,f): # Compute single time step using Boris integrator
    dt2 = dt/2 # Half time step
    B = np.asarray((computeBx(rn[0],rn[1],a,J,f),computeBy(rn[0],rn[1],a,J,f),Bz)) # Compute magnetic field at particle position
    tv = B*dt2 # t vector
    s = 2*tv/(1+np.dot(tv,tv)) # s vector
    vnew = vn + np.cross(vn + np.cross(vn,tv),s) # Update velocity
    r = rn + vnew*dt # Update position
    return r,vnew,B

@jit(nopython = True) 
def integrateBoris(r0,v0,dt,Bz,N,a,J,current): # Integrate trajectory for N time steps
    r,v,B = np.zeros((3,N+1)).astype('double'),np.zeros((3,N+1)).astype('double'),np.zeros((3,N+1)).astype('double') # Initialize arrays for position, velocity, and magnetic field
    r[:,0],v[:,0],B[:,0] = r0,v0,np.asarray((computeBx(r0[0],r0[1],a,J,current[0]),computeBy(r0[0],r0[1],a,J,current[0]),Bz)) # Initial conditions
    for i in np.linspace(1,N,N).astype('int'): # Evolve system for N time steps
        r[:,i],v[:,i],B[:,i] = computeStep(r[:,i-1],v[:,i-1],dt,Bz,a,J,current[i])    
    return r,v,B

def computeBNorm(B): # Compute magnitude of magnetic field
    return np.sqrt(pow(B[0,:],2) + pow(B[1,:],2) + pow(B[2,:],2))

# ---------- Integrating Multiple Trajectories ----------

'''
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
'''
    
# ---------- Visualization ----------

def computeAutoScaling(x,y): # Auto scaling for axes (This function was stolen from Pierre's intro to computational plasma notes: https://gourdain.pas.rochester.edu/index.php/introduction-to-plasma-physics)
    xc = (x.max() + x.min())/2.
    x_low = xc - (x.max() - x.min())/2.*1.1
    x_high = xc + (x.max() - x.min())/2.*1.1
    yc = (y.max() + y.min())/2.
    y_low = yc - (y.max() - y.min())/2.*1.1
    y_high = yc + (y.max()-y.min())/2.*1.1
    
    return x_low,x_high,y_low,y_high

def plot2D(tArr,r,v,normB,N,a,J,plotColor,fig=None,ax=None): # Make two dimensional phase space plots p_i versus r_i for i = x,y,z; plot for real-space particle trajectory flattened along z
    print('Plotting two-dimensional trajectories.')
    if (fig is None) and (ax is None):
        fig,ax = plt.subplots(2,2,figsize=(16,9.5)) # Define axes
    if N > 10000: # Set size of points for plotting
        size = 0.1
    else:
        size = 5

    if plotColor == 'time': # Set plot color
        colors = plt.cm.gist_rainbow(tArr/np.max(tArr))
    elif plotColor == 'Bfield':
        colors = normB
    
    # All particle trajectories have a colormap set by the magnitude of the magnetic field at the location of the particle for the time step in question. 
    # This is to help understand why specific kinks or deflections occur in the trajectory (for example, locations where the field is large correspond 
    # to the most extreme bending of the particle trajectory).

    ax[0,0].scatter(r[0,0],v[0,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot x phase space
    ax[0,0].scatter(r[0,1:],v[0,1:],c=colors[1:],s=size)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel(r'$P_x$')
    ax[0,0].set_title(f'x Phase Space, J = {round(J,2)}')
    ax[0,0].grid()
    ax[0,0].legend()
    
    ax[0,1].scatter(r[1,0],v[1,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot y phase space
    ax[0,1].scatter(r[1,1:],v[1,1:],c=colors[1:],s=size)
    ax[0,1].set_xlabel('y')
    ax[0,1].set_ylabel(r'$P_y$')
    ax[0,1].set_title(f'y Phase Space, J = {round(J,2)}')
    ax[0,1].grid()
    ax[0,1].legend()
    
    ax[1,0].scatter(r[2,0],v[2,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot z phase space
    ax[1,0].scatter(r[2,1:],v[2,1:],c=colors[1:],s=size)
    ax[1,0].set_xlabel('z')
    ax[1,0].set_ylabel(r'$P_z$')
    ax[1,0].set_title(rf'z Phase Space, J = {round(J,2)}')
    ax[1,0].grid()
    ax[1,0].legend()
  
    ax[1,1].scatter(r[0,0],r[1,0],c='r',marker='*',s=55*size,label='Initial Position') # Plot real-space trajectory flattened to xy plane
    ax[1,1].scatter(r[0,1:],r[1,1:],c=colors[1:],s=size)
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

    if plotColor == 'time':
        c = fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=tArr[1],vmax=tArr[-1]),cmap='gist_rainbow'),ax=ax,fraction=0.046,pad=0.1)
        c.set_label('Time (s)',rotation=360)
    elif plotColor == 'Bfield':
        c = fig.colorbar(c,ax=ax)
        c.set_label('|B|',rotation=360)

    plt.show()

    return fig,ax
    
def plot3D(tArr,r,v,normB,a,plotColor,fig=None,ax=None): # Plot 3D real-space and momentum-space plots
    print('Plotting three-dimensional trajectories.')
    fig = plt.figure(figsize=(16,9.5)) # Set figure size
    size = 5 # Set size of points to plot

    if plotColor == 'time': # Set plot color
        colors = plt.cm.gist_rainbow(tArr/np.max(tArr))
    elif plotColor == 'Bfield':
        colors = normB
    
    ax = fig.add_subplot(1,2,1,projection='3d') # Plot position space
    ax.scatter(r[0,0],r[1,0],r[2,0],c='r',marker='*',s=15*size,label='Initial Position') # Highlight initial position of particle
    aArr = -a*np.ones(2)
    zroArr = np.zeros(2)
    zArr = np.asarray([min(r[2,:]),max(r[2,:])])
    ax.plot(aArr,zroArr,zArr,c='b',label='Line Current') # Draw position of line currents
    ax.plot(-aArr,zroArr,zArr,c='b')
    ax.scatter(r[0,1:],r[1,1:],r[2,1:],c=colors[1:],s=size) # Plot trajectory
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
    ax.scatter(v[0,1:],v[1,1:],v[2,1:],c=colors[1:],s=size)
    ax.set_xlabel(r'$P_x$')
    ax.set_ylabel(r'$P_y$')
    ax.set_zlabel(r'$P_z$')
    ax.set_title('3D Momentum Space Trajectory')
    ax.grid()
    ax.legend()

    if plotColor == 'time':
        c = fig.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=tArr[1],vmax=tArr[-1]),cmap='gist_rainbow'),ax=ax,fraction=0.046,pad=0.1)
        c.set_label('Time (s)',rotation=360)
    elif plotColor == 'Bfield':
        c = fig.colorbar(c,ax=ax)
        c.set_label('|B|',rotation=360)

    plt.show()
    
def plotJ(time,current,normB): # Plot current profile
    print('Plotting current profile.')
    plt.figure(figsize=(12,9.5))
    plt.plot(time,current)
    plt.scatter(time,current,c=normB,s=5) # Plot current profile
    plt.xlabel('Time (s)')
    plt.ylabel('Current')
    plt.title('Current Profile')
    plt.grid()
    plt.show()
    
def plotE(p,time,normB): # Plot total energy as a function of time, compute varaince to check for conservation
    print('Plotting total energy time series.')
    E = np.sqrt(pow(p[0,:],2) + pow(p[1,:],2) + pow(p[2,:],2))
    var = str(np.var(E)/pow(np.min(E),2)) # Compute variance of energy normalized by smallest energy scale
    length = len(var)
    plt.figure(figsize=(12,9.5))
    plt.plot(time,E,label=f'Normalized Variance of Energy = {var[0:6]+var[length-4:length]}') # Plot energy time series
    plt.scatter(time,E,c=normB) # Scatter plot on top of line to color with norm of B
    c = plt.colorbar()
    c.set_label('|B|',rotation=360) # Define colorbar
    plt.grid()
    padding = 0.05 # Padding for y limits
    plt.ylim([min(E)*(1-padding),max(E)*(1+padding)])
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy')
    plt.title('Time Evolution of Total Energy')
    plt.legend()
    plt.show()

def defineGrid(d,N): # Define meshgrid parameters for visualizing fields
    # d is domain length, N is number of grid points
    lb = -d/2 # Domain Lower bound
    ub = d/2 # Domain upper bound
    n = N-1 # Number of points minus 1, useful for indexing
    dn = n/ub # Points per unit length

    X = np.linspace(lb,ub,N) # Create array for meshgrid based on parameters
    x,y = np.meshgrid(X,X) # Initialize meshgrid
    sign = 1 # Controls sign of the current
    return x,y,sign,n,dn,X

def showFields(a,d,J,save=False,threshold=0.0,N=101,useNorm=True): # Plot magnetic field and vector potential
    x,y,sign,n,dn,X = defineGrid(d,N)

    xMin = findClosestGridPoint(a,X,n)
    yMin = 1/dn

    tA = computeAll(x,y,a,J,sign,n,N,computeA,xMin,yMin,threshold) # Vector potential, shear
    rA = computeAll(x,y,a,J,-sign,n,N,computeA,xMin,yMin,threshold) # Vector potential, mirror

    tBx = computeAll(x,y,a,J,sign,n,N,computeBx,xMin,yMin,threshold) # Magnetic field along x, shear
    tBy = computeAll(x,y,a,J,sign,n,N,computeBy,xMin,yMin,threshold) # Magnetic field along y, shear

    rBx = computeAll(x,y,a,J,-sign,n,N,computeBx,xMin,yMin,threshold) # Magnetic field along x, mirror
    rBy = computeAll(x,y,a,J,-sign,n,N,computeBy,xMin,yMin,threshold) # Magnetic field along y, mirror

    size = 10 #Size of points that denote wire locations

    # Plotting
    
    Scale = 65
    
    if useNorm == True: # Normalize vectors
        tBx,tBy = normalize(tBx,tBy)
        rBx,rBy = normalize(rBx,rBy)

    size = 10 # Size of points that denote wire locations

    fig,ax = plt.subplots(2,2,figsize=(12,10))

    ax[0,0].set_title('Az for Parallel Currents')
    im = ax[0,0].pcolormesh(x,y,tA/np.max(tA))
    ax[0,0].scatter(-a,0,c='r',label=r'$J\hat{z}$',s=size)
    ax[0,0].scatter(a,0,c='r',label=r'$J\hat{z}$',s=size)
    fig.colorbar(im,ax=ax[0,0])
    ax[0,0].legend()

    ax[0,1].set_title('Az for Antiparallel Currents')
    im = ax[0,1].pcolormesh(x,y,rA/np.max(rA))
    ax[0,1].scatter(-a,0,c='r',label=r'$J\hat{z}$',s=size)
    ax[0,1].scatter(a,0,c='black',label=r'-$J\hat{z}$',s=size)
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