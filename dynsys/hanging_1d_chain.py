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
        self.AddConstraintEqns(Jrow,checkConstraints=False)
        
        
        
# ********************** TEST ROUTINES ****************************************
# (Only execute when running as a script / top level)
if __name__ == "__main__":
    
    dynSys1 = hanging1DChain(7,60,12)
    dynSys1.PrintSystemMatrices()