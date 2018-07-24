# -*- coding: utf-8 -*-
"""
Class definitions and functions for definition of damper systems e.g. TMDs

Damper systems are not usually analysed in isolation; rather they would normally 
be appended to another system (e.g. a modal system)
"""

from __init__ import __version__ as currentVersion

# Standard imports
import numpy
import matplotlib.pyplot as plt
import pandas

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
        
        if fixed_mass is None:
            fixed_mass = sprung_mass / 100 # reasonable value, non-zero required
        
        if fixed_mass == 0.0:
            raise ValueError("`fixed_mass` cannot be zero!")
                
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
        
        # Add output matrix to compute damper force
        # given by K.v_relative + C.vdot_relative
        self.AddOutputMtrx(output_mtrx=[-K,+K,-C,+C,0,0],
                           output_names=["TMD linkage force [N]"])
        
        
    def PlotSystem(self,ax,v,
                   relative=True,
                   seperation=1.0,
                   direction='x',
                   **kwargs):
        """
        Plot deformed configuration of the TMD system
        
        ***
        **Required:**
        
        * `ax`, axes object, onto which system will be plotted
        
        * `v`, _array_ of displacement results, defining the position of 
          analysis freedoms
          
        ***
        **Optional:**
        
        * `relative`, _boolean_, if True relative position between sprung mass 
          and fixed mass will be plotted. Otherwise displacement of both masses 
          will be plotted
          
        * `seperation`, _float_ defining initial seperation of masses when v=0
        
        * `direction`, _string_, direction in which 1D displacement should be 
          plotted. Either 'x' or 'y' required.
          
        """
        
        # Convert to flattened array, if not already
        v = numpy.ravel(v)
        
        # Define initial positions
        v0 = numpy.array([0.0,seperation])
        
        # Determine relative displacement, if required
        if relative:
            v = v - v[0]
            
        # Define deformed position to plot
        v = v0 + v
        
        # Plot mass positions
        z = [0.0,0.0]
        
        if direction == 'x':
            h = ax.plot(v,z,'b.')
        else:
            h = ax.plot(z,v,'b.')
            
        return h
        
# --------------- FUNCTIONS -------------------
        
def append_TMDs(modal_sys,
                fname:str,
                append:bool=True,
                verbose:bool=True):
    """
    Define and append multiple TMDs to 'parent' modal system.
    TMD 'child' systems are defined using inputs read-in from .csv file
    
    ***
    Required:
        
    * `modal_sys`, instance of `ModalSys` class; 'parent' modal system to which 
      TMDs are to be appended
      
    * `fname`, _string_; filename of file in which TMD definitions are provided
      
    ***
    Optional:
        
    * `append`, _boolean_; if False then TMD systems will only be defined, not 
      appended to the parent modal system. In this case the `modal_sys` 
      argument should be set as `None`.
    
    ***
    Returns:
        
    List of `TMD()` instances i.e. objects created by this function
    
    """
    
    if verbose:
        print("Defining TMD system using input provided in '%s'..." % fname)
    
    # Read in TMD defintions from datafile
    TMD_defs = pandas.read_csv(fname)
    
    # Parse dataframe for specific details
    M = TMD_defs["Mass (kg)"].values
    f = TMD_defs["Freq (Hz)"].values
    eta = TMD_defs["Damping ratio"].values
    
    Xpos = TMD_defs["Chainage (m)"].values
    modeshapes_TMD = TMD_defs.values[:,4:]
    
    if modeshapes_TMD.shape[1]==0:
        modeshapes_TMD = None #' no input provided
      
    # Loop through to define all TMD systems
    sys_list = []
    
    for i, (_M, _f, _eta) in enumerate(zip(M,f,eta)):
        
        sys_list.append(TMD(sprung_mass=_M,
                            nat_freq=_f,
                            damping_ratio=_eta,
                            name="TMD#%d" % (i+1)))    
    
    
    nTMD = len(sys_list)
    print("Number of TMDs defined: %d" % nTMD)  
    
    # Append TMDs to parent modal system
    if append:
        
        # Check modalsys is as expected type
        if modal_sys.__class__.__name__ != 'ModalSys':
            raise ValueError("Error with `modal_sys` argument:" + 
                             "must be instance of 'ModalSys' class!")
            
        # Loop over all new TMD systems to append
        for i, TMD_sys in enumerate(sys_list):
            
            if modeshapes_TMD is None:
                mTMD = None
            else:
                mTMD = modeshapes_TMD[i,:]
                        
            modal_sys.AppendSystem(child_sys=TMD_sys,
                                   Xpos_parent=Xpos[i],
                                   modeshapes_parent=mTMD,
                                   DOF_child=0)
                
    # Return list of TMD system objects
    return sys_list
        
 
# TEST ROUTINES
if __name__ == "__main__":
    
    my_tmd = TMD(sprung_mass=1000,nat_freq=1.2,damping_ratio=0.1,name='test')
    my_tmd.PrintSystemMatrices(printValues=True)
    
    print("Output matrix:\n{0}".format(my_tmd.output_mtrx))
    
    fig, ax = plt.subplots()
    my_tmd.PlotSystem(ax,[0.1,0.2])
    