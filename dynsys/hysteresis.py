# Hysteresis model
# http://eprints.lancs.ac.uk/1375/1/MFI_10c.pdf
# Identification of Hysteresis Functions Using a Multiple Model Approach
# Mihaylova, Lampaert et al

import numpy as npy
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy

#%%

plt.close('all')

class HysteresisModel:
    """
    Hysteresis model comprising a number of elementary Maxwell-slip models
    refer http://eprints.lancs.ac.uk/1375/1/MFI_10c.pdf
    """
    
    def __init__(self,N,K,W=None,delta=None):
        
        self.N = N
        """
        Integer number of elementary models
        """
        
        self.K = npy.ravel(npy.abs(K))
        """
        Array of stiffness values for each elementary model
        """
        
        if delta is None :
            # K and W specified
            
            if W is None:
                raise ValueError("Error: either delta or W arguments "+
                                 "must be provided!")
            else:
                W = npy.ravel(npy.abs(W))          # limiting friction values
                
        else:
            # K and delta specified
            # W to be inferred, given this input
            delta = npy.abs(npy.ravel(delta))
            W = self.K * delta
            
        self.W = W
        """
        Array of limiting friction values for each elementary model
        """
        
        # Initialise matrices F and C, which do not vary with input
        self.F = npy.asmatrix(npy.identity(self.N))
        self.C = npy.asmatrix(npy.diag(-self.K))
        
        # Initialise matrices G and D, as empty
        self.G = npy.asmatrix(npy.empty((self.N,1)))
        self.D = npy.asmatrix(npy.empty((self.N,1)))
        
        # Initialise array to contain case indexs
        self.case = npy.zeros((self.N,),dtype=int)
        
                
        
    @property
    def x0(self):
        return self._x0        

    @x0.setter
    def x0(self,x0):
        """
        Set initial states
        """
        self._x0 = npy.asmatrix(npy.ravel(x0)).T
        self.x =self.x0
        
        if self.x.shape[0] != self.N:
            raise ValueError("Error: x0 wrong shape!")
            
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self,val):
        #print("States updated")
        self._x = npy.asmatrix(val)
        
            
    def update(self,u,save_states=True):
        """
        Function to advance state-space description of model dynamics 
        by a single time step, returning next state and output vectors
        """
    
        x = copy.deepcopy(self.x)
        
        # Define G and D matrix entries
        for i in range(self.N): # loop over all elementary models
            
            Wi = self.W[i]
            Ki = self.K[i]
            
            # Evaluate switching parameter
            fi = Ki * (u - x[i])
            
            if fi > Wi:
                # Case 2
                self.case[i] = 2
                self.G[i] = 1
                self.D[i] = 0
                x[i] = -Wi/Ki
                
            
            elif fi < -Wi:
                # Case 3
                self.case[i] = 3
                self.G[i] = 1
                self.D[i] = 0   
                x[i] = +Wi/Ki
            
            else:
                # Case 1
                self.case[i] = 1
                self.G[i] = 0
                self.D[i] = Ki
        
        # Compute next states and output
        # using eqns (10) and (11) in Mihaylova's paper
        x_next = self.F * x + self.G * u
        y_k = self.C * x + self.D * u
        
        Fh_k = y_k.sum() # total hysteresis force
                
        # Update states
        if save_states:
            self.x = x_next

        return x_next, y_k, Fh_k
    
        
    def run(self,x0,uVals):
        """
        Run simulation from initial conditions, given inputs u
            x0    : column vector [Nx1]
            u     : list or vector of length (nSteps,1)
        """
        
        # Convert and check shape of u
        uVals = npy.ravel(uVals)
        nSteps = uVals.shape[0] 
        
        # Initialise state space eqns
        self.x0 = x0
        
        # Step through state space eqns
        xVals = npy.zeros((nSteps,self.N))
        yVals = npy.zeros((nSteps,self.N))
        Fh_vals = npy.zeros((nSteps,))
        
        for k, u_k in enumerate(uVals):
            
            # Get next states and output
            x_k, y_k, Fh_k = self.update(u_k)
            
            # Store
            xVals[k,:] = npy.ravel(x_k)
            yVals[k,:] = y_k.T
            Fh_vals[k] = Fh_k
        
        # Store results
        self.uVals = uVals
        self.xVals = xVals
        self.yVals = yVals
        self.FhVals = Fh_vals
        
        # Return states and output for each step
        return xVals, yVals, Fh_vals
    
    
    def write_results(self,
                      fname='results.csv',
                      delimiter=','):
        
        arr = npy.asmatrix(self.uVals).T
        titles = ["u"]
        N = self.N
        
        arr = npy.hstack((arr,self.xVals))
        titles += ["x%d" % (i+1) for i in range(N)]
        
        arr = npy.hstack((arr,self.yVals))
        titles += ["y%d" % (i+1) for i in range(N)]
        
        arr = npy.hstack((arr,npy.asmatrix(self.FhVals).T))
        titles += ["Fh"]
        
        npy.savetxt(fname=fname,
                    X=arr,
                    delimiter=delimiter,
                    header=delimiter.join(str(x) for x in titles))
    
    
    def PlotResults_timeSeries(self,tVals):
        """
        Plot results as time series
        [t,u], [t,x], [t,y], [t,Fh]
        """
        
        fig, axarr = plt.subplots(4,sharex=True)
        fig.set_size_inches(16,9,forward=True)
        
        ax1 = axarr[0]
        ax1.plot(tVals,self.uVals)
        ax1.xaxis.set_visible(False)
        ax1.set_ylabel("u")
        ax1.set_xlabel("Input displacement, u(t)")
        
        ax2 = axarr[1]
        ax2.plot(tVals,self.xVals)
        ax2.xaxis.set_visible(False)
        ax2.set_ylabel("x")
        ax2.set_title("States of\nelementary models, x(t)")
        
        ax3 = axarr[2]
        ax3.plot(tVals,self.yVals)
        ax3.xaxis.set_visible(False)
        ax3.set_ylabel("y")
        ax3.set_title("Outputs from\nelementary models, y(t)")
        
        ax4 = axarr[3]
        ax4.plot(tVals,self.FhVals)
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("F$_h$")
        ax4.set_title("Net output F$_h$")
        
        
    def PlotResults(self):
        """
        Plot results as [u,x], [u,y], [u,Fh] plots
        """
        
        fig, axarr = plt.subplots(1,3,sharex=True)
        fig.set_size_inches(16,9,forward=True)
        
        ax1 = axarr[0]
        ax1.plot(self.uVals,self.xVals)
        ax1.set_xlabel("Input u")
        ax1.set_title("States of\nelementary models, x")
        
        ax2 = axarr[1]
        
        ax2.plot(self.uVals,self.yVals)
        ax2.set_xlabel("Slip (u-x)")
        ax2.set_title("Outputs from\nelementary models, y")
        
        ax3 = axarr[2]
        ax3.plot(self.uVals,self.FhVals)
        ax3.set_xlabel("Input u")
        ax3.set_title("Net output F$_h$")
    

class static_response():
    """
    Class used to compute response to forcing input
    """

    def __init__(self,hys_obj,K1, K2):
        self.hys_obj = hys_obj
        self.K1 = K1
        self.K2 = K2
    
    def net_force(self,d,F_ext,verbose=False):
        """
        Function which defines net force 
        given position 'u' and external force 'F_ext'
        """
        
        u = d[0] - d[1]  # relative displacement at friction interface
        F_hys = self.hys_obj.update(u=u,save_states=False)[2]

        F_net_1 = self.K1 * d[0] + F_hys - F_ext
        F_net_2 = self.K2 * d[1] - F_hys
        F_net = npy.array([F_net_1,F_net_2])
        
        if verbose:
            print("u = %.3e" % u)
            print("x = {0}".format(self.hys_obj.x))
            print("F_hys = {0}".format(F_hys))
            print("F_net = {0}".format(F_net))
        
        return F_net
    
    def run(self,F_vals,x0=None,d0=None):
        
        # Define function to solve for next u
        def solve(d_last,F_k,hys_obj):
    
            # Determine next u to satify equilibrium - i.e. zero net force
            sol = root(fun=self.net_force,x0=d_last,args=(F_k,))
            d_k = sol.x
            u_k = d_k[0]-d_k[1]
            
            F_net = self.net_force(d_k,F_k)
            
            if not sol.success:
                pass#print(sol.message)
                
            x_k, y_k, F_hys_k = hys_obj.update(u=u_k,save_states=True)
            
            return F_hys_k, d_k, u_k, x_k, y_k, F_net
        
        # Set initial conditions
        if x0 is None:
            x0 = npy.zeros((self.hys_obj.N,))
        self.hys_obj.x0 = x0
        
        if d0 is None:
            d0 = npy.array([0.0,0.0])
        d_j = d0 # initial guess
        
        # Run step by step 
        F_hys_vals = []
        x_vals = []
        u_vals = []
        y_vals = []
        F_net_vals = []
        
        for j, F_j in enumerate(F_vals):
            
            #print("--- Step #%d ---" % j)
            F_hys_j, d_j, u_j, x_j, y_j, F_net = solve(d_j,F_j,self.hys_obj)
            
            F_hys_vals.append(F_hys_j)
            x_vals.append(npy.ravel(x_j))
            y_vals.append(npy.ravel(y_j))
            u_vals.append(u_j)
            F_net_vals.append(F_net)
        
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.u_vals = u_vals
        self.F_hys_vals = F_hys_vals
        self.F_vals = F_vals
        self.F_net_vals = F_net_vals
        
    
    def plot(self):
        
        fig, axarr = plt.subplots(3,2,sharex='col')
        fig.set_size_inches(14,8)
        
        ax = axarr[0,0]
        ax.plot(self.F_vals,label='$F_{external}$')
        ax.plot(self.F_hys_vals,label='$F_{hysteresis}$')
        ax.legend()
        ax.set_ylabel("Forces")
        
        ax = axarr[1,0]
        ax.plot(self.u_vals)
        ax.set_ylabel("Displacement, u")
        
        ax = axarr[2,0]
        ax.plot(self.x_vals)
        ax.set_xlabel("Step index")
        ax.set_ylabel("States, x")
        
        ax = axarr[0,1]
        ax.plot(self.u_vals,self.y_vals)
        ax.set_ylabel("Outputs, y")
        
        ax = axarr[1,1]
        ax.plot(self.u_vals,self.F_hys_vals)
        ax.set_ylabel("$F_{hysteresis}$")
        
        ax = axarr[2,1]
        ax.plot(self.u_vals,self.F_vals)
        ax.set_xlabel("Displacement, u")
        ax.set_ylabel("$F_{external}$")
        
        return fig
        
    
# -------- TEST ROUTINE ----------

if __name__ == "__main__":
    
    test_routine = 1
    
    if test_routine == 0:
            
        # Define hysteresis model
        K = [1000,2000,3000] 
        delta = [1,2,3]
        Ne = len(K)
        hys = HysteresisModel(Ne,K,delta=delta)
        
        # Define displacement inputs
        dt = 0.02
        tmax = 10
        u0 = 10
        
        import random

        def randomWalk(N,normalise=True):
                
            x= [0]
            
            for j in range(N-1):
                step_x = random.randint(0,1)
                if step_x == 1:
                    x.append(x[j] + 1 + 0.05*npy.random.normal())
                else:
                    x.append(x[j] - 1 + 0.05*npy.random.normal())
                 
            x = npy.asarray(x)
            
            if normalise:
                absmaxVal = npy.max([npy.max(x),-npy.min(x)])
                x = x / absmaxVal
            
            return x
        
        tVals = npy.arange(0,tmax,dt)
        uVals = u0*randomWalk(tVals.shape[0])
        
        #uVals = 4.5*npy.sin(2*npy.pi*0.5*tVals)
        
        # Obtain states and outputs by state space stepping
        hys.run(npy.zeros((Ne,)),uVals)
        
        # Plot results
        hys.PlotResults()
        hys.PlotResults_timeSeries(tVals)
          
        #hys.write_results()
    
    elif test_routine==1:
        
        # Define hysteresis model
        K = [1000,2000,3000] 
        W = [1000,1000,1000]
        Ne = len(K)
        hys = HysteresisModel(Ne,K,W=W)
        
        # Define force function
        # Define displacement inputs
        dt = 0.02
        tmax = 10
        u0 = 10
        F0 = 3000
        
        t_vals = npy.arange(0,tmax,dt)
        F_vals = F0 * (npy.sin(2*npy.pi*t_vals) + npy.sin(2*npy.pi*3.2*t_vals))
    
        # Define spring
        K_spring = 1500
        
        # Define and run analysis
        analysis = static_response(hys_obj=hys,K_spring=K_spring)
        analysis.run(F_vals=F_vals)
        analysis.plot()            
    
    else:
        raise ValueError("No test selected!")
    
#%%

#
