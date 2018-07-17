# -*- coding: utf-8 -*-
"""
Class definitions and functions for definition of damper systems e.g. TMDs

Damper systems are not usually analysed in isolation; rather they would normally 
be appended to another system (e.g. a modal system)
"""

from __init__ import __version__ as currentVersion

# Standard imports


# DynSys imports
import msd_chain
from dynsys import SDOF_stiffness, SDOF_dashpot

class TMD(msd_chain.MSD_Chain):
    """
    Defines a single-mass tuned mass damper system
    """
    
    description="Tuned mass damper"
    
    def __init__(self,sprung_mass:float,nat_freq:float,
                 fixed_mass:float=None,damping_ratio:float=0.0,
                 **kwargs):
        """
        Initialisation method
        
        A 2DOF system will be initialised, the first freedom of which will
        ordinarily be appended to another system (e.g. modal system)
        
        ***
        **Required:**
        
        * `sprung_mass`, _float_, mass (in kg) of sprung mass component of TMD
        
        * `nat_freq`, _float_, undamped natural frequency (in Hz) of TMD system 
          in isolation
        
        ***
        **Optional:**
        
        * `fixed_mass`, _float_, mass (in kg) of fixed part of TMD, which will 
          be treated as rigidly fixed to the structure to which the TMD system 
          is appended.
          
            * A zero value should be avoided, to avoid ill-conditioning of the 
              mass matrix. The support frames of most real TMDs will in any case 
              usually have a non-negligible mass.
              
            * If _None_ is provided (as per default argument) the fixed mass 
              will be assumed to be 1/100th of the sprung mass
          
        * `damping_ratio`, _float_, damping ratio (1.0=critical) of TMD system 
          in isolation
          
        Additional keyword arguments may be provided; these will be passed to 
        `MSD_Chain.__init__()` method; refer [docs](..\docs\msd_chain.html) 
        for that class for further details
        
        """
        
        # Handle optional arguments
        if fixed_mass is None:
            fixed_mass = sprung_mass / 100
        
        # Define masses, stiffnesses and damping dashpot of msd_chain system
        K = SDOF_stiffness(M=sprung_mass,f=nat_freq)
        C = SDOF_dashpot(M=sprung_mass,K=K,eta=damping_ratio)
        
        M_vals = [fixed_mass,sprung_mass]
        K_vals = [0.0,K]
        C_vals = [0.0,C]
    
        # Invoke msd_chain's init method
        super().__init__(M_vals=M_vals,K_vals=K_vals,C_vals=C_vals,**kwargs)
        
        # Add output matrix entry to compute relative displacement
        # (as this is commonly of interest)
        self.AddOutputMtrx(output_mtrx=[-1,1,0,0,0,0],
                           output_names=["Relative disp [m]"])
        
        
    
# TEST ROUTINES
if __name__ == "__main__":
    
    my_tmd = TMD(sprung_mass=1000,nat_freq=1.2,damping_ratio=0.1,name='test')
    my_tmd.PrintSystemMatrices(printValues=True)
    
    print("Output matrix:\n{0}".format(my_tmd.output_mtrx))
    