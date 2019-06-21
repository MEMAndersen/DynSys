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
from dynsys import SDOF_stiffness, SDOF_dashpot, SDOF_dampingRatio

class TMD(msd_chain.MSD_Chain):
    """
    Defines a single-mass tuned mass damper system
    """
    
    description="Tuned mass damper"
    
    def __init__(self,sprung_mass:float,nat_freq:float,
                 fixed_mass:float=None,damping_ratio:float=0.0,
                 dashpot:float=None,
                 outputs:dict={},
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
          in isolation. Alternatively `dashpot` argument can be used to define 
          the TMD damping
          
        * `dashpot`, _float_, defines dashpot rate (in Ns/m). Use in place of 
          `damping_ratio` argument.
          
        Additional keyword arguments may be provided; these will be passed to 
        `MSD_Chain.__init__()` method; refer [docs](..\docs\msd_chain.html) 
        for that class for further details
        
        """
        
        self._initialise(sprung_mass,nat_freq,fixed_mass,
                         damping_ratio,dashpot,outputs,**kwargs)
    
                
                
    def _initialise(self,sprung_mass,nat_freq,fixed_mass,
                    damping_ratio,dashpot,outputs,**kwargs):
        """
        Method used to actually initialise class. This is called from 
        __init__().
        
        The reason for defining as seperate (private) method is to allow 
        re-initialisation of properties, e.g. for when defining damper in a 
        probabilistic (Monte Carlo) sense
        """
        
        # Manipulate inputs as necessary, to define key properties of damper
        self._define_properties(fixed_mass,sprung_mass,
                                nat_freq,
                                damping_ratio,dashpot)
        
        # Invoke msd_chain's init method
        M_vals = [self.M_fixed,self.M_sprung]
        K, C = self.K, self.C
        K_vals = [0.0,K]
        C_vals = [0.0,C]
        super().__init__(M_vals=M_vals,K_vals=K_vals,C_vals=C_vals,**kwargs)
        
        # Add output matrix entry to compute relative displacement
        # (as this is commonly of interest)
        
        # Define default outputs and merge with any passed-in
        default_outputs = {}
        default_outputs['disp'] = True
        default_outputs['rel disp'] = True
        default_outputs['linkage force'] = False
        outputs = {**default_outputs, **outputs}
        
        if outputs is not None:
            
            #print("Defining default output matrices for '%s'..." % self.name)
            
            # Add output matrix for sprung mass displacement
            if outputs['disp']:
                self.add_outputs(output_mtrx=[0,1,0,0,0,0],
                                   output_names=["Disp [m]"])
            
            # Add output matrix for relative displacement with attachment point
            if outputs['rel disp']:
                self.add_outputs(output_mtrx=[-1,1,0,0,0,0],
                                   output_names=["Relative disp [m]"])
            
            # Add output matrix to compute damper force
            # given by K.v_relative + C.vdot_relative
            if outputs['linkage force']:
                self.add_outputs(output_mtrx=[-K,+K,-C,+C,0,0],
                                   output_names=["TMD linkage force [N]"])
                
                
        
    def _define_properties(self,fixed_mass,sprung_mass,
                           nat_freq,damping_ratio,C):
        """
        Define key properties of the damper, given inputs provided
        
        Refer docstring for __init__() method for details of inputs
        """
                
        # Define masses, stiffnesses and damping dashpot of msd_chain system
        K = SDOF_stiffness(M=sprung_mass,f=nat_freq)
        
        if C is None:
            C = SDOF_dashpot(M=sprung_mass,K=K,eta=damping_ratio)
        else:
            damping_ratio = SDOF_dampingRatio(sprung_mass,K,C)
            
        if fixed_mass is None:
            fixed_mass = sprung_mass / 100 # reasonable value, non-zero required
        elif fixed_mass == 0.0:
            raise ValueError("`fixed_mass` cannot be zero!")
        
        # Store inputs as attributes
        self._M = sprung_mass
        self._M_fixed = fixed_mass
        self._fn = nat_freq
        self._eta = damping_ratio
        self._C = C
        self._K = K
        
        
    @property
    def M_sprung(self):
        """
        Sprung component of TMD mass [kg]
        """
        return self._M
    
    @property
    def M_fixed(self):
        """
        Fixed component of TMD mass [kg] (e.g. mass of attachment assembly)
        """
        return self._M_fixed
    
    @property
    def fn(self):
        """
        Undamped natural frequency [Hz] of TMD when taken in isolation
        """
        return self._fn
    
    @property
    def damping_ratio(self):
        """
        Damping ratio (critical=1) for TMD taken in isolation
        """
        return self._eta
    
    @property
    def C(self):
        """
        Viscous dashpot rate [Ns/m] for TMD linkage
        """
        return self._C
        
    @property
    def K(self):
        """
        Spring rate [N/m] for TMD linkage
        """
        return self._K
        
        
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
                fname:str=None,
                tmd_list=None,
                tmd_pos=None,
                append:bool=True,
                verbose:bool=True):
    """
    Define and append multiple TMDs to 'parent' modal system.
    TMD 'child' systems are either provided via `tmd_list` or else 
    are defined using inputs read-in from .csv file
    
    ***
    Required:
        
    * `modal_sys`, instance of `ModalSys` class; 'parent' modal system to which 
      TMDs are to be appended
      
    ***
    Optional:
        
    * `tmd_list`, list of TMD instances to append to modal_sys. If None 
      (default) then TMD objects will be defined based on input from `fname`
      
    * `tmd_pos`, list of locations of TMD instances. Only required if TMDs 
      defined via `tmd_list`. Otherwise will be read-in from .csv file
      
    * `fname`, _string_; filename of file in which TMD definitions are provided
        
    * `append`, _boolean_; if False then TMD systems will only be defined, not 
      appended to the parent modal system. In this case the `modal_sys` 
      argument should be set as `None`.
    
    ***
    Returns:
        
    List of `TMD()` instances i.e. objects created by this function
    
    """
    
    
    
    if not append and tmd_list is not None:
        append = True # override input parameter if tmds provided via tmd_list
    
    if tmd_list is None:
        
        tmd_list = []
        
        if verbose:
            print("Defining TMD system using input provided in '%s'" % fname)
    
        # Read in TMD defintions from datafile
        TMD_defs = pandas.read_csv(fname)
        
        # Parse dataframe for specific details
        M = TMD_defs["Mass (kg)"].values
        f = TMD_defs["Freq (Hz)"].values
        eta = TMD_defs["Damping ratio"].values
        
        tmd_pos = TMD_defs["Chainage (m)"].values
        modeshapes_TMD = TMD_defs.values[:,4:]
        
        if modeshapes_TMD.shape[1]==0:
            modeshapes_TMD = None #' no input provided
          
        # Loop through to define all TMD systems
        for i, (_M, _f, _eta) in enumerate(zip(M,f,eta)):
            
            tmd_list.append(TMD(sprung_mass=_M,
                                nat_freq=_f,
                                damping_ratio=_eta,
                                name="TMD#%d" % (i+1)))    
        
    
        nTMD = len(tmd_list = [])
        print("Number of TMDs defined: %d" % nTMD)  
    
    else:    
        modeshapes_TMD=None # ensures modeshapes at TMD positions determined 
                            # from supplied modal system
        
    
    if tmd_pos is None:
        raise ValueError("TMD locations must be specified via `tmd_pos`")
        
    if len(tmd_pos)!=len(tmd_list):
        raise ValueError("Inconsistent `tmd_pos` and `tmd_list` list lengths")
    
    # Append TMDs to parent modal system
    if append:
        
        # Check modalsys is as expected type
        if modal_sys.__class__.__name__ != 'ModalSys':
            raise ValueError("Error with `modal_sys` argument:" + 
                             "must be instance of 'ModalSys' class!")
            
        # Loop over all new TMD systems to append
        for i, TMD_sys in enumerate(tmd_list):
            
            if modeshapes_TMD is None:
                mTMD = None
            else:
                mTMD = modeshapes_TMD[i,:]
                        
            modal_sys.AppendSystem(child_sys=TMD_sys,
                                   Xpos_parent=tmd_pos[i],
                                   modeshapes_parent=mTMD,
                                   DOF_child=0)
                
    # Return list of TMD system objects
    return tmd_list
        
 
# TEST ROUTINES
if __name__ == "__main__":
    
    my_tmd = TMD(sprung_mass=1000,nat_freq=1.2,damping_ratio=0.1,name='test')
    my_tmd.PrintSystemMatrices(printValues=True)
    
    print("Output matrix:\n{0}".format(my_tmd.output_mtrx))
    
    fig, ax = plt.subplots()
    my_tmd.PlotSystem(ax,[0.1,0.2])
    