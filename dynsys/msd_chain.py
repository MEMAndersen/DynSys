# -*- coding: utf-8 -*-
"""
Class definition and test functions for msdChain, a class used to define
a chain of SDOF mass-spring-dashpot systems
"""

from __init__ import __version__ as currentVersion

# Std library imports
import numpy as npy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Other imports
from dynsys import DynSys
from dynsys import (angularFreq,
                    SDOF_stiffness, 
                    SDOF_dashpot, 
                    SDOF_dampingRatio, 
                    SDOF_frequency)

class MSD_Chain(DynSys):
    """
    Defines a chain of SDOF mass-spring-dashpot systems
    """
    
    description="Mass-spring-dashpot chain system"
    
    def __init__(self, M_vals,
                 f_vals=None,eta_vals=None,
                 K_vals=None,C_vals=None,
                 **kwargs):
        """
        Initialisation function
        ***
        
        All inputs are (in general) _array-like_ and relate to each 
        of the SDOF systems in the chain. All arguments are required to be of 
        the same shape (this is checked).
        
        ***
        Required:
            
        * `M_vals`, mass (kg) corresponding to each DOF
        
        ***
        Optional:
        
        * `f_vals`, undamped natural frequency (Hz)
        
        * `K_vals`, stiffness (N/m) applicable to the each DOF linkage
        
        Either `K_vals` or `f_vals` must be provided. If both are provided, 
        `K_vals` will be used.
        
        * `eta_vals`, damping ratio (1.0 = critical)
        
        * `C_vals`, dashpot rate (Ns/m) applicable to the each DOF linkage
        
        Either `C_vals` or `eta_vals` must be provided. If both are provided, 
        `C_vals` will be used.
        
        Additional keyword arguments may be provided; these will be passed to 
        `DynSys.__init__()` method; refer [docs](..\docs\dynsys.html) 
        for that class for further details
        
        """

        if M_vals is None:
            raise ValueError("`M_vals` cannot be 'None'")

        # Flatten input and establish nDOFs
        M_vals = npy.ravel(npy.asarray(M_vals))
        
        if f_vals is not None:      f_vals = npy.ravel(npy.asarray(f_vals))
        if eta_vals is not None:    eta_vals = npy.ravel(npy.asarray(eta_vals))
        if K_vals is not None:      K_vals = npy.ravel(npy.asarray(K_vals))
        if C_vals is not None:      C_vals = npy.ravel(npy.asarray(C_vals))
        
        nDOF = M_vals.shape[0]
        
        # Determine K matrix
        if K_vals is None:
            
            if f_vals is None:
                raise ValueError("Either `f_vals` or `K_vals` is required")
                
            else:
                f_vals = SDOF_frequency(M_vals,K_vals)
                omega_vals = angularFreq(f_vals)
                K_vals = SDOF_stiffness(M_vals,omega=omega_vals)                
        
        # Determine C matrix
        if C_vals is None:
            
            if eta_vals is None:
                raise ValueError("Either `eta_vals` or `C_vals` is required")
                
            else:
                C_vals = SDOF_dashpot(M_vals,K_vals,eta_vals)
                    
        # Define system matrices
        
        M_mtrx = npy.asmatrix(npy.diag(M_vals))
        C_mtrx = npy.zeros_like(M_mtrx)
        K_mtrx = npy.zeros_like(M_mtrx)
    
        for n in range(nDOF):
            
            # On diagonal terms
            C_mtrx[n,n]=C_vals[n]
            K_mtrx[n,n]=K_vals[n]
            
            # Off diagonal and appended terms
            
            if n!=0:
                
                C_mtrx[n-1,n]=-C_vals[n]
                C_mtrx[n,n-1]=-C_vals[n]
                C_mtrx[n-1,n-1]+=C_vals[n]
                
                K_mtrx[n-1,n]=-K_vals[n]
                K_mtrx[n,n-1]=-K_vals[n]
                K_mtrx[n-1,n-1]+=K_vals[n]
    
        # ---- Write details into object as attributes -----
        
        # Write details into object using parent init function
        super().__init__(M_mtrx,C_mtrx,K_mtrx,**kwargs)
        
    
    def PlotSystem(self,ax,**kwargs):
        
        x_vals = npy.arange(0,self.nDOF)
        y_vals = npy.ravel(kwargs.get("v"))
        ax.plot(x_vals,y_vals.T,"r.",markersize=50)        
        
        x_vals = npy.vstack((x_vals,x_vals))
        y_vals = npy.vstack((npy.zeros_like(y_vals),y_vals)).T
        ax.plot(x_vals,y_vals.T,"b-")
        
        ax.set_xlabel("DOF index")
        ax.set_ylabel("Displacement")
        
        
# ******************* FUNCTIONS ************************

def warburton_TMD(m_M = 2000,       # main mass (kg)
                  f_M = 1.0,        # system undamped natural freq (Hz)
                  mu = 0.05,        # mass ratio
                  f = 1.0,          # tuning ratio
                  gamma_M = 0.003,  # damping ratio of main system in isolation
                  gamma_A = 0.05,   # damping ratio of TMD system in isolation
                  fmax=5.0          # max frequency to evaluate G_f at
                  ):
    """
    Define a two-mass system as per Figure 1 of Warburton and Ayorinde (1980)
    ***
    
    Notation used in the implementation is as per paper:
    [PDF](../references/Warburton and Ayorinde (1980).pdf)
    
    """
            
    # Derive other terms required
    m_A = mu * m_M
    omega_M = angularFreq(f_M)
    omega_A = f * omega_M
    f_A = f * f_M
    k_M = SDOF_stiffness(m_M,omega=omega_M)
    k_A = SDOF_stiffness(m_A,omega=omega_A)
    c_M = SDOF_dashpot(m_M,k_M,gamma_M)
    c_A = SDOF_dashpot(m_A,k_A,gamma_A)
    
    # Create object instqnce
    msd_sys = MSD_Chain(M_vals = [m_M,m_A],
                        K_vals = [k_M,k_A],
                        C_vals = [c_M,c_A]
                        )
    
    # Define frequency values to evaluate G_f at
    freqVals = msd_sys.freqVals(fmax=fmax)
    omega = angularFreq(freqVals)
    
    # Define expected frequency response using eqns in Warburton's paper
    numerator = k_A - m_A * omega**2 + 1j * omega * c_A
    denom_term_11 = k_M + k_A - m_M * omega**2 + 1j * omega * (c_M + c_A)
    denom_term_12 = k_A - m_A * omega**2 + 1j * omega * c_A
    denom_term_2 = k_A + 1j * omega * c_A
    H_f = numerator / (denom_term_11 * denom_term_12 - denom_term_2**2)

    # Returns key variables as dict
    names= ['m_M', 'm_A', 'k_M', 'k_A', 'c_M', 'c_A']
    items = locals()
    d = {}
    for name in names:
        d[name] = items[name]
        
    # Return statement
    return d, msd_sys, freqVals, H_f
        
        
# ********************** TEST ROUTINES ****************************************
# (Only execute when running as a script / top level)
if __name__ == "__main__":
    
    M_vals = npy.asarray([100,100,100,100])
    f_vals = npy.asarray([1,2,3,4])
    eta_vals = npy.asarray([0.03,0.04,0.05,0.01])

    dynSys1 = MSD_Chain(M_vals,f_vals,eta_vals)
    dynSys1.PrintSystemMatrices()
    
    # Produce some nice plots
    fig = plt.figure(figsize=(18,9))
    gs = gridspec.GridSpec(1, 1)
    
    ax1 = plt.subplot(gs[0])
    dynSys1.PlotSystem(ax1,v=[0.2,0.3,-1.0,-0.2])
