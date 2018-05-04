# -*- coding: utf-8 -*-
"""
Class definition and test functions for ModalSys, a class used to define
a second order dynamic system via its (usually truncated) modal properties 
"""

from __init__ import __version__ as currentVersion

# Std library imports
import numpy as npy
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import warnings
import deprecation # not in anaconda distribution - obtain this from pip

# Other imports
from dynsys import DynSys
from dynsys import (angularFreq,
                    SDOF_stiffness,
                    SDOF_dashpot)

class ModalSys(DynSys):
    """
    Dynamic system represented by its (usually truncated) modal properties 
    """
    
    description="Modal dynamic system"
    
    def __init__(self,
                 name=None,
                 isSparse=False,
                 fname_modalParams="modalParams.csv",
                 fname_modeshapes="modeshapes.csv",
                 output_mtrx=None,
                 output_names=[],
                 fLimit=None,
                 ):
        """
        Initialisation function used to define decoupled (diagonalised) system
        
        
        
        Optional arguments:
        
        * `isSparse`, denotes whether sparse matrix representation should be 
          used for system matrices and numerical operations.
        
        * `outputsList`, list of output matrices
        
        * `fLimit`, maximum natural frequency for modes: modes with fn in 
          excess of this value will not be included in analysis
        
        
        An _output matrix_ defines a linear mapping between modal results 
        (displacements, velocities, accelerations) and real-world outputs, 
        which need not be displacements. For example real-world displacements, 
        forces/moments, reactions etc. may all be obtained by linear 
        combination of modal displacement results. Similarly real-world 
        velocities and accelerations relate linearly to modal velocities and 
        accelerations.
        
        
        For a system with _n_ degrees of freedom, the required shape of 
        each listed output matrix is _[mx3n]_, where _m_ is the number of 
        outputs defined. The output matrix consists of horiztonally-stacked 
        submatrices, each of shape _[Oxn]_, which are used to pre-multiply 
        modal displacements, velocities and accelerations respectively.
        
        """
        
        # Import data from input files
        d = self._DefineModalParams(fName=fname_modalParams,fLimit=fLimit)
        
        mode_IDs = d["mode_IDs"]
        M_mtrx = d["M_mtrx"]
        C_mtrx = d["C_mtrx"]
        K_mtrx = d["K_mtrx"]
        J_dict = d["J_dict"]
        
        # Write details into object using parent init function
        super().__init__(M_mtrx,C_mtrx,K_mtrx,
                         J_dict=J_dict,
                         output_mtrx=output_mtrx,
                         output_names=output_names,
                         isLinear=True,
                         isModal=True,
                         isSparse=isSparse,
                         name=name)
    
        self.mode_IDs = mode_IDs
        """
        Labels to describe modal dofs
        """
        
        # Read modeshape data
        if fname_modeshapes is not None:
            self.DefineModeshapes(fname_modeshapes)
        
        
    def _DefineModalParams(self,fName='modalParams.csv', fLimit=None):
        """
        It is generally most convenient to define modal parameters by reading
        data in from .csv file. Comma-delimited data in the following table
        format is expected:
            | ModeID | Frequency | Mass | Damping ratio |
            | ---    | ---       | ---  | ---           |
            | Mode_1 | 1.25      | 2500 | 0.002         |
            | Mode_2 | 1.75      | 1600 | 0.015         |
            | ...    | ...       | ...  | ...           |
            | Mode_N | 8.2       | 7400 | 0.03          |
            
            Frequency:       Mode undamped natural frequency (in Hz)      
            Mass:            Mode-generalised mass (in kg)
            Damping ratio:   Modal damping ratio 'eta', as fraction of critical
        """
        
        # Read data from csv
        modalParams = pd.read_csv(fName,header=0,index_col=0)
        mode_IDs    = modalParams.index.tolist()
        f_vals      = npy.asarray(modalParams["Frequency"])       
        M_vals      = npy.asarray(modalParams["Mass"]) 
        eta_vals    = npy.asarray(modalParams["Damping ratio"]) 
        nDOF = M_vals.shape[0]
        
        
        
        # Only define for modes with f < fLimit, if specified
        if fLimit is not None:
            
            # Sort input into ascending frequency order
            indexs = npy.argsort(f_vals)
            mode_IDs = mode_IDs[indexs]
            f_vals = f_vals[indexs]
            M_vals = M_vals[indexs]
            eta_vals = eta_vals[indexs]
            
            # Only retain entries with f_vals < fLimit
            Nm = npy.searchsorted(f_vals,fLimit,side='right')
            f_vals = f_vals[:Nm]
            M_vals = M_vals[:Nm]
            eta_vals = eta_vals[:Nm]
            nDOF_full = nDOF
            nDOF = M_vals.shape[0]
            
            print("fLimit = %.2f specified. " % fLimit + 
                  "Only the first #%d of #%d modes " % (Nm+1,nDOF_full+1) + 
                  "will be included.")
            
        # Check required input is valid and consistent
        if f_vals.shape[0]!=nDOF:
            raise ValueError("Error: length of f_vals " + 
                             "does not agree with expected nDOF!")
        if eta_vals.shape[0]!=nDOF:
            raise ValueError("Error: length of eta_vals " + 
                             "does not agree with expected nDOF!")
            
        # Calculate circular natural freqs
        omega_vals = angularFreq(f_vals)
        
        # Calculate SDOF stiffnesses and dashpot constants
        K_vals = SDOF_stiffness(M_vals,omega=omega_vals)
        C_vals = SDOF_dashpot(M_vals,K_vals,eta_vals)
        
        # Store modal properties no longer required as attributes anyway
        self.M_generalised = M_vals
        """
        Mode-generalised mass (kg) of modes, as defined by input file
        """
        
        self.fn = f_vals
        """
        Undamped natural frequencies of modes, as defined by input file
        """
        
        self.eta = eta_vals
        """
        Damping ratios for modes, as defined by input file
        """
        
        # Assemble system matrices, which are diagonal due to modal decomposition
        M_mtrx = npy.asmatrix(npy.diag(M_vals))
        C_mtrx = npy.asmatrix(npy.diag(C_vals))
        K_mtrx = npy.asmatrix(npy.diag(K_vals))
        
        # Return matrices and other properties using dict
        d = {}
        d["mode_IDs"]=mode_IDs
        d["M_mtrx"]=M_mtrx
        d["C_mtrx"]=C_mtrx
        d["K_mtrx"]=K_mtrx
        d["J_dict"]={} # no constraints
        return d
            
                
    def DefineModeshapes(self,fName='modeshapes.csv',saveAsAttr=True):
        """
        Function to allow 1-dimensional line-like modeshapes to be defined
        e.g. for use in calculating mode-generalised forces
        
        It is generally most convenient to define such systems by reading
        data in from .csv file. Comma-delimited data in the following table
        format is expected:
            
            | Chainage | Mode_1 | Mode_2 | Mode_3 | ... | Mode_N |
            | ---      | ---    | ---    | ---    | --- | ---    |
            | ...      | ...    | ...    | ...    | --- | ...    |
            
            Chainage:   Column defining common chainage for modeshape ordinates
            Mode_i:     Modeshape ordinates for given ModeID
        
        """
    
        # Read in data from .csv file
        df = pd.read_csv(fName,delimiter=',',header=0,index_col=0)
        
        chainageVals = df.index
        modeshapeVals = npy.asarray(df)
        mode_IDs = df.columns.values.tolist()
        
        # Adjust chainage values to start from zero
        chainageVals = chainageVals - min(chainageVals)
        
        # Get length of track along dynamic system as defined by modeshapes
        Ltrack = max(chainageVals)-min(chainageVals)
            
        # Set up interpolation function: linear interpolation between modeshape ordinates provided
        modeshapeFunc = scipy.interpolate.interp1d(chainageVals,
                                                   modeshapeVals,
                                                   axis=0,
                                                   bounds_error=False,
                                                   fill_value=0)
        
        # Check modeNames agree with modalParams input
        if not npy.array_equal(mode_IDs, self.mode_IDs):
            warnings.warn("Unexpected mode_IDs in {0}\n".format(fName) + 
                          "mode_IDs: {0}\n".format(mode_IDs) + 
                          "self.mode_IDs: {0}\n".format(self.mode_IDs))
        
        # Define class properties
        if saveAsAttr:
            self.modeshapeFunc = modeshapeFunc
            self.Ltrack = Ltrack
    
        return modeshapeFunc, Ltrack
    
    def PlotModeshapes(self,
                       num:int = 50,
                       L:float = 100.0,
                       ax=None,
                       plotAttached=True):
        """
        Plot modeshapes vs chainage using 'modeshapeFunc'
        
        
        Optional:
            
        * `ax`: axes object onto which plot should be produced. If `None` then 
          new figure will be produced.
          
        * `L`: chainage is defined in the range [0,L]. L=100.0m is default.
          If `Ltrack` attribute is defined, this value will be used instead.
        
        * `num`: number of intermediate chainages to interpolate modeshapes at
        
        * `plotAttached`: if `modeshape_attachedSystems` and 
          `Xpos_attachedSystems` attributes exist, modeshape ordinates at 
          attachment positions will be overlaid as red dots (usually attached 
          systems will represent damper systems)
        
        """
        
        # Configure plot
        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(16,4)
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.gcf()
            
        # Use Ltrack instead of L passed, if attribute defined
        attr="Ltrack"
        obj=self
        if hasattr(obj,attr):
            L=getattr(obj,attr)
            
        # Use interpolation function to obtain modeshapes at
        x = npy.linspace(0,L,num,endpoint=True)
        m = self.modeshapeFunc(x)
        
        # Get mode IDs to use as labels
        if hasattr(self,"mode_IDs"):
            modeNames = self.mode_IDs
        else:
            modeNames = npy.arange(1,m.shape[1],1)
            
        # Plot modeshapes vs chainage
        ax.plot(x,m,label=modeNames)
        ax.set_xlim([0,L])
        ax.set_xlabel("Longitudinal chainage [m]")
        ax.set_ylabel("Modeshape ordinate")
        ax.set_title("Modeshapes along loading track")
        
        # Overlaid modeshape ordinates at attachment positions, if defined
        if plotAttached and len(self.DynSys_list)>1:
            
            attr1 = "Xpos_attachedSystems"
            attr2 = "modeshapes_attachedSystems"
            makePlot = True
            
            if hasattr(self,attr1):
                X_TMD = getattr(self,attr1)
            else:
                makePlot = False
                print("Warning: {0} attribute not defined\n".format(attr1) + 
                      "Modeshape ordinates at attached system locations " + 
                      "cannot be plotted")
                
            if hasattr(self,attr2):
                modeshape_TMD = getattr(self,attr2)
            else:
                makePlot = False
                print("Warning: {0} attribute not defined\n".format(attr2) + 
                      "Modeshape ordinates at attached system locations " + 
                      "cannot be plotted")
                    
            if makePlot:
                ax.plot(X_TMD,modeshape_TMD,'xr',label="Attached systems")
        
        # Prepare legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, modeNames, loc='best',fontsize='xx-small',ncol=5)
        
        # Return objects
        return fig, ax
    
    
    def CalcModalForces(self,loading_obj,
                        loadVel:float=5.0,
                        Ltrack:float=None,
                        dt:float=0.01,
                        use_abs_modeshape:bool=False):
        """
        Calculates the mode-generalised forces due to series of point loads
        ***
        
        Practically this is done by evaluating the following summation:
            
        $$ Q_j = \sum_{k=1}^{N} \phi_j(x_k) F_k $$
        
        ***
        Required:
            
        * `loadtrain_obj`, instance of `LoadTrain` class, defines load pattern
        
        ***
        Optional:
        
        * `loadVel`, _float_, velocity of moving load along track
            
        * `Ltrack`, _float_, length of track along which load pattern is 
          running. If _None_ then `Ltrack` will be sought from class attributes.
            
        * `dt`, _float_, time increment at which to evaluate mode-generalised 
          forces at
          
        * `use_abs_modeshape`, _boolean_: if True then absolute values of 
          modeshape vector will be used; required for some forms of analysis e.g. 
          pedestrian moving load analysis. Default is 'False'.
        
        ***
        Returns:
            
        Function f(t) as expected by `tstep.__init__()`
        
        """
        
        # Get loading details
        loadX = loading_obj.loadX
        loadVals = loading_obj.loadVals(t=0.0)
        
        # Check shapes of loadX and loadVals agree
        if not npy.array_equal(loadX.shape,loadVals.shape):
            raise ValueError("Shapes of `loadX` and `loadVals` do not agree!")
            
        # Check modeshapes are defined
        attr = "modeshapeFunc"
        if not hasattr(self,attr):
            raise ValueError("Error: `modeshapeFunc` not defined! " + 
                             "Cannot compute mode-generalised forces")
        else:
            modeshapeFunc = getattr(self,attr)
            
        # Define function in the form expected by tstep
        def ModalForces(t):
            
            leadX = loadVel*t
            xPos = leadX + loadX
            
            modeshapeVals = modeshapeFunc(xPos)
            
            if use_abs_modeshape:
                modeshapeVals = npy.abs(modeshapeVals)
            
            QVals = npy.asmatrix(modeshapeVals.T) @ loading_obj.loadVals(t=t)
            QVals = npy.ravel(QVals)
            return QVals
        
        return ModalForces
            
    @deprecation.deprecated(deprecated_in="1.0.0",current_version=currentVersion)
    def AppendTMDs(self,chainage_TMD,mass_TMD,freq_TMD,eta_TMD,
                   defineRelDispOutputs=True):
        """
        Function appends a set of TMDs (a list of simple mass-spring-dashpot
        systems with 1DOF each) to the current DynSys object
        ***
        Note in contrast to class method `AppendSystem()` constraint equations
        are not used, but rather system matrices are edited to reflect the
        attachment of TMD freedoms
        
        ***This function is now marked as deprecated. Update still required to 
        male class method `CalcEigenproperties()` usuable for systems with 
        constraint equations. However functionality provided in updated 
        `AppendSystem()` means that function should generally be use***
        
        For a full description of the method (and notation) adopted, refer
        *The Lateral Dynamic Stability of Stockton Infinity Footbridge
        Using Complex Modes*
        by Alan McRobie. 
        [PDF](../references/The Lateral Dynamic Stablity of Stockton 
        Infinity Footbridge using Complex Modes.pdf)
        
        ***
        Optional:
            
        * `defineRelDispOutputs`, option to create new rows in (enlarged) output
          matrix, to define relative displacement between TMD and structure at 
          attachment point
        
        """
        
        # Check no constraints define (method won't work in this case!)
        if self.hasConstraints():
            raise ValueError("Error: cannot use function 'AppendTMS' " + 
                             "for systems with constraints")
            
        # Define modeshape interpolation function (if not done so already)
        if not hasattr(self,"modeshapeFunc"):
            raise ValueError("Error: you must run 'DefineModeshapes' first!")
            
        # Define diagonal mass matrix for modes and TMDs
        M_a = self.M_mtrx
        C_a = self.C_mtrx
        K_a = self.K_mtrx
        Nm = M_a.shape[0]
        
        # Define TMD matrices
        omega_TMD = npy.asmatrix(npy.diagflat(angularFreq(freq_TMD)))
        eta_TMD = npy.asmatrix(npy.diagflat(eta_TMD))
        
        M_T = npy.asmatrix(npy.diagflat(mass_TMD))
        K_T = omega_TMD * omega_TMD * M_T
        C_T = 2 * eta_TMD * omega_TMD * M_T
        N_T = M_T.shape[0]

        # Use modeshape function to obtain modeshape ordinates at TMD positions
        modeshape_TMD = self.modeshapeFunc(chainage_TMD)
        self.chainage_TMD = chainage_TMD
        self.modeshape_TMD = modeshape_TMD
        
        # Determine mode-TMD mass matrix
        m1 = npy.hstack((M_a,npy.asmatrix(npy.zeros((Nm,N_T)))))
        m2 = npy.hstack((npy.asmatrix(npy.zeros((N_T,Nm))),M_T))
        M_aT = npy.vstack((m1,m2))
        
        # Define B_aT matrix
        B_aT = npy.hstack((modeshape_TMD,-npy.identity(N_T)))
        B_aT
        
        # Determine K_aT matrix
        k1 = npy.hstack((K_a,npy.asmatrix(npy.zeros((Nm,N_T)))))
        k2 = npy.asmatrix(npy.zeros((N_T,Nm+N_T))) 
        K_aT1 = npy.vstack((k1,k2))
        
        K_aT2 = B_aT.T * K_T * B_aT
        
        K_aT = K_aT1 + K_aT2
        
        # Determine C_aT matrix
        c1 = npy.hstack((C_a,npy.asmatrix(npy.zeros((Nm,N_T)))))
        c2 = npy.asmatrix(npy.zeros((N_T,Nm+N_T))) 
        C_aT1 = npy.vstack((c1,c2))
        
        C_aT2 = B_aT.T * C_T * B_aT
        
        C_aT = C_aT1 + C_aT2
        
        # Decompose system output matrix into blocks
        output_mtrx = self.output_mtrx
        nOutputs = output_mtrx.shape[0]
        o1 = output_mtrx[:,:Nm]
        o2 = output_mtrx[:,Nm:2*Nm]
        o3 = output_mtrx[:,2*Nm:]
        
        # Define null output matrix for TMDs
        oNew = npy.asmatrix(npy.zeros((nOutputs,N_T)))
        
        # Re-assemble output matrix for enlargened system
        output_mtrx = npy.hstack((o1,oNew,o2,oNew,o3,oNew))
        output_names = self.output_names
        
        if defineRelDispOutputs:
            
            # Define displacement block to give relative displacement at TMD locs
            z = npy.zeros((N_T,output_mtrx.shape[1]-Nm-N_T))
            
            # Define remainder blocks
            new_outputs = npy.hstack((-modeshape_TMD,npy.identity(N_T),z))
            
            # Name new outputs
            new_names = []
            for i in range(N_T):
                new_names.append("Relative displacement (m), TMD{0}".format(i+1))

            # Append as new rows
            output_names = output_names + new_names
            output_mtrx = npy.append(output_mtrx,new_outputs,axis=0)
        
        # Overwrite system matrices
        self.M_mtrx = M_aT
        self.C_mtrx = C_aT
        self.K_mtrx = K_aT
        
        nDOF = M_aT.shape[0]
        self.nDOF = nDOF
        
        self.J_mtrx = npy.asmatrix(npy.zeros((0,nDOF)))
        self.output_mtrx = output_mtrx
        self.output_names = output_names
    
  
# ********************** TEST ROUTINE ****************************************
# (Only execute when running as a script / top level)
        
if __name__ == "__main__":
    
    pass
    
    
    
    
