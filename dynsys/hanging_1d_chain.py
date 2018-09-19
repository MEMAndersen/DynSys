# -*- coding: utf-8 -*-
"""
Class definition and test functions for hanging1DChain, a class used to define
a chain hanging vertically under self-weight
"""

from __init__ import __version__ as currentVersion

# Std library imports
import numpy as npy

# Other imports
import dynsys

class hanging1DChain(dynsys.DynSys):
    """
    Chain hanging vertically under self-weight
    ***
    
    DOFs are transverse displacements of lumped masses.
    
    
    *Note*: Model is linearised; small rotations and displacements assumed
    """
    
    def __init__(self,nDOF,mass_per_length,length,modal_eta=0,g=10,constrainFixedEnd=True):
        """
        Initialisation function
        ***
        
        Required:
            
        * `mass_per_length`, mass per unit length (kg/m)
        
        * `length`, length of chain (m)
        
        * `nDOF`, number of lumped masses to be used to define subdivide chain
        
        
        `nDOF=0` denotes the top of the chain. By default `nDOF=0` is 
        constrained to not displace.
        """    
        
        self.mass_per_length = mass_per_length
        self.length = length
        
        # Define mass matrix
        M_vals = mass_per_length*length / (nDOF-1) * npy.ones((nDOF,))
        M_vals[0]=M_vals[0]/2               # halve first mass to account for half-length
        M_vals[nDOF-1]=M_vals[nDOF-1]/2     # halve last mass to account for half-length
        M_mtrx = npy.asmatrix(npy.diag(M_vals))
        
        # Derive tensions in links between masses
        T_vals = npy.zeros((nDOF,1))
        for i in range(nDOF-1,-1,-1):
    
            T_vals[i] = M_vals[i]*g
            
            if i!=nDOF-1:
                T_vals[i] = T_vals[i] + T_vals[i+1]
        #print(T_vals)
        
        # Define length of chain segments between masses
        L_vals = length / (nDOF-1) * npy.ones((nDOF-1,))
        
        # Define stiffness matrix
        K_mtrx = npy.zeros_like(M_mtrx)
        
        for i in range(nDOF):
            
            if i!=0:        
                K_mtrx[i,i]+=T_vals[i-1]/L_vals[i-1]
                K_mtrx[i,i-1]+=-T_vals[i-1]/L_vals[i-1]
    
            if i!=nDOF-1:
                K_mtrx[i,i]+=T_vals[i]/L_vals[i]
                K_mtrx[i,i+1]+=-T_vals[i]/L_vals[i]
        
        # Define modal damping ratios and hence C matrix
        if modal_eta==0:
            C_mtrx = npy.zeros_like(M_mtrx)
        else:
            raise ValueError("C_mtrx for non-zero modal_eta not yet implemented!")
        
        # Write details into object using parent init function
        super().__init__(M_mtrx,C_mtrx,K_mtrx,isLinear=True)
        
        # Define constraint equation to restrain node 0
        # Which would be normal for chain where pivot point is fixed in space
        if constrainFixedEnd:
            Jrow = npy.asmatrix(npy.zeros((1,self.nDOF)))
            Jrow[0,0]=1
        else:
            Jrow = npy.asmatrix(npy.zeros((0,self.nDOF)))
        
        self.AddConstraintEqns(Jnew=Jrow,Jkey='0',checkConstraints=False)
        
        
    def PlotSystem_init_plot(self,ax,plot_env=True):
        """
        Method for initialising system displacement plot
        ***
        (Will usually be overriden by derived class methods)
        """
                
        # Variables used to generate plot data
        self.x = npy.arange(self.nDOF)/(self.nDOF-1) * self.length
        self.v_env_max = npy.zeros((self.nDOF,))
        self.v_env_min = npy.zeros_like(self.v_env_max)

        # Define drawing artists
        self.lines = {}
        
        self.lines['v'] = ax.plot([], [],'ko-',label='$v(t)$')[0]
    
        self.plot_env = plot_env
        if plot_env:        
            
            self.lines['v_env_max'] = ax.plot(self.x,
                                              self.v_env_max,
                                              color='r',alpha=0.3,
                                              label='$v_{max}$')[0]
            
            self.lines['v_env_min'] = ax.plot(self.x,
                                              self.v_env_min,
                                              color='b',alpha=0.3,
                                              label='$v_{min}$')[0]
        
        # Set up plot parameters
        ax.grid(axis='x')
        ax.axhline(0.0,color='k')
        ax.set_xlim(0,self.length)
        ax.set_xticks(self.x)
        ax.set_xlabel("Distance along chain")
        ax.set_ylabel("Displacement (m)")
        
    
    def PlotSystem_update_plot(self,v):
        """
        Method for updating system displacement plot given displacements `v`
        ***
        (Will usually be overriden by derived class methods)
        """
        
        # Update envelopes
        self.v_env_max = npy.maximum(v,self.v_env_max)
        self.v_env_min = npy.minimum(v,self.v_env_min)       
        
        # Update plot data
        self.lines['v'].set_data(self.x,v)
        
        if self.plot_env:
            self.lines['v_env_max'].set_data(self.x,self.v_env_max)
            self.lines['v_env_min'].set_data(self.x,self.v_env_min)
        
        return self.lines
        
        

        
# ********************** TEST ROUTINES ****************************************
# (Only execute when running as a script / top level)
if __name__ == "__main__":
    
    import tstep
    
    nDOF = 21
    mass_per_length = 60
    length = 10
    
    dynSys1 = hanging1DChain(nDOF,mass_per_length,length)
    dynSys1.PrintSystemMatrices()
    
    x0 = npy.ravel([npy.sin(npy.pi*(npy.arange(nDOF)/nDOF)),npy.zeros((nDOF,))])
    my_results = tstep.TStep(dynSys1,x0=x0).run()
    
    #my_results.PlotStateResults()
    
    anim = my_results.AnimateResults()
    #%%
    #my_results.PlotEnergyResults()