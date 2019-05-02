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
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings

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
                 name:str=None,
                 isSparse:bool=False,
                 fname_modalParams:str="modalParams.csv",
                 fname_modeshapes:str="modeshapes.csv",
                 modalParams_dict:dict=None,
                 modeshapeFunc=None,
                 Ltrack:float=None,
                 output_mtrx=None,
                 output_names:list=None,
                 fLimit:float=None,
                 **kwargs):
        """
        Initialisation function used to define decoupled (diagonalised) system
        
        ***
        Optional:
        
        * `isSparse`, denotes whether sparse matrix representation should be 
          used for system matrices and numerical operations.
                  
        * `fname_modalParams`, string, defines .csv file containing modal 
          parameter definitions. Refer docs for `ReadModalParams()` method for 
          expected file format
          
        * `fname_modeshapes`, string, defines .csv file containing modeshape 
          data, describing how modeshape ordinates vary with chainage along 
          loading track. Refer docs for `DefineModeshapes()` method for 
          expected file format
          
        * `modeshapeFunc`, function, used to define how modeshape varies with 
          chainage. Expected form is f(x), where x denotes chainage. `Ltrack` 
          must be supplied also (see below)
          
        * `Ltrack`, float, defines overall chainage along `modeshapeFunc`
          
        * `output_mtrx`, matrix, used when calculating responses
        
        * `output_names`, list, defines names of responses defined by rows of
          `output_mtrx`
        
        * `fLimit`, maximum natural frequency for modes: modes with fn in 
          excess of this value will not be included in analysis
          
        """
        
        if modalParams_dict is None:
            
            # Import data from input files
            modalParams_dict = self.ReadModalParams(fName=fname_modalParams)
            
        # Define mode IDs, if not already provided
        if not 'ModeIDs' in modalParams_dict.keys():
            
            mass_vals = modalParams_dict['Mass']
            
            if isinstance(mass_vals,(int,float)):
                nModes = 1
            else:
                nModes = len(mass_vals)
            
            mode_IDs = ["Mode %d" % (i+1) for i in range(nModes)]
            modalParams_dict['ModeIDs'] = mode_IDs
            
        
        self.modalParams_dict = modalParams_dict
        self._CheckModalParams()
        """
        _Dict_ containing modal parameters used to define system
        """
        
        
        self.mode_IDs = modalParams_dict['ModeIDs']
        """
        Labels to describe modal dofs
        """
            
        # Filter according to frequency
        if fLimit is not None:
            self._FilterByFreq(fLimit)
        
        # Calculate system matrices
        d = self._CalcSystemMatrices()
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
                         name=name,
                         **kwargs)
    
        # Define modeshapes
        if modeshapeFunc is None:
            if fname_modeshapes is not None:
                
                # Read data from file
                modeshapeFunc, Ltrack = self.DefineModeshapes(fname_modeshapes)
            
        else:
            
            if Ltrack is None:
                raise ValueError("As `modeshapeFunc` has been supplied "+ 
                                 "`Ltrack` must be provided also")
            
        self.modeshapeFunc = modeshapeFunc
        """
        Function defining how modeshapes vary with chainage
        """
        
        self.Ltrack = Ltrack
        """
        Defines overall chainage
        """
        
        
    def ReadModalParams(self,fName='modalParams.csv'):
        """
        It is often most convenient to define modal parameters by reading
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
        
        # Return as dict
        d = {}
        d['Mass'] = M_vals
        d['Freq'] = f_vals
        d['DampingRatio'] = eta_vals
        d['ModeIDs'] = mode_IDs
        return d
    
    
    def _CheckModalParams(self):
        
        # Bring modal parameters dict into function
        mp_dict = self.modalParams_dict
        
        # Check dict contains expected entries
        M_vals = mp_dict['Mass']
        mp_dict['Mass'] = npy.ravel(M_vals)
        
        if isinstance(M_vals,(int,float)):
            nDOF=1
        else:
            nDOF = M_vals.shape[0]
        
        # Handle pseudonyms of frequency
        freq_attr_list = ['freq','frequency']
        for attr in freq_attr_list:
            if hasattr(mp_dict,attr.lower()):
                mp_dict['Freq'] = npy.ravel(getattr(attr))
                
        # Handle pseudonyms of damping ratio
        damping_attr_list = ['damping_ratio','dampingratio','eta']
        for d in damping_attr_list:
            if hasattr(d,d.lower()):
                mp_dict['DampingRatio'] = npy.ravel(getattr(mp_dict,d))   
        
        # Check shapes are consistent
        attr_list = ['Freq','DampingRatio']
        
        for attr in attr_list:
            
            vals = mp_dict[attr]
            
            if vals.shape!=(nDOF,):
                raise ValueError("Error: shape of '%s' array " % attr + 
                                 "does not agree with expected nDOF!")
            
#            if nDOF==1 :
#                if not isinstance(vals,(int,float)):
#                    raise ValueError("Error: float expected for '%s' value" 
#                                     % attr)
#            
#            else:
#                
#            
                    
        # Update class attribute
        self.modalParams_dict = mp_dict
                
    
    def _FilterByFreq(fLimit,modalParams_dict=None):
        """
        Use to filter modalParams dict for only those modes with f < fLimit
        """
        
        # Bring modal parameters dict into function
        if modalParams_dict is None:
            modalParams_dict = self.modalParams_dict
            
        # Check how many modes included prior to filtering
        nModes_before = len(modalParams_dict['Mass'])
            
        # Convert dict to pandas dataframe and filter
        df = pd.DataFrame(modalParams_dict)
        df = df.loc[(df['Freq'] <= fLimit)]
        modalParams_dict = df.to_dict(orient='list')
        self.modalParams_dict
        
        # Check how many modes included after filtering
        nModes = len(modalParams_dict['Mass'])
            
        print("fLimit = %.2f specified. " % fLimit + 
              "Only #%d of #%d modes defined" % (nModes,nModes_before) + 
              "will be included.")
    
        return modalParams_dict
    
    
    def _CalcSystemMatrices(self,modalParams_dict=None):
        """
        Calculates diagonal system matrices from modal parameters
        """
        
        # Bring modal parameters dict into function
        if modalParams_dict is None:
            modalParams_dict = self.modalParams_dict
            
        M_vals = modalParams_dict['Mass']
        f_vals = modalParams_dict['Freq']
        eta_vals = modalParams_dict['DampingRatio']
        
        # Calculate circular natural freqs
        omega_vals = angularFreq(f_vals)
        
        # Calculate SDOF stiffnesses and dashpot constants
        K_vals = SDOF_stiffness(M_vals,omega=omega_vals)
        C_vals = SDOF_dashpot(M_vals,K_vals,eta_vals)
        
        # Assemble system matrices, which are diagonal due to modal decomposition
        nDOF = M_vals.shape[0]
        
        if nDOF != 0:
            M_mtrx = npy.asmatrix(npy.diag(M_vals))
            C_mtrx = npy.asmatrix(npy.diag(C_vals))
            K_mtrx = npy.asmatrix(npy.diag(K_vals))
            
        else:
            M_mtrx = M_vals
            C_mtrx = C_vals
            K_mtrx = K_vals
        
        # Return matrices and other properties using dict
        d = {}
        d["M_mtrx"]=M_mtrx
        d["C_mtrx"]=C_mtrx
        d["K_mtrx"]=K_mtrx
        d["J_dict"]={} # no constraints
        return d
        
        
    def DefineModeshapes(self,fName='modeshapes.csv'):
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
            
        return modeshapeFunc, Ltrack
    

    def PlotModeshapes(self,*args,**kwargs):
        """
        Note: DEPRECATED METHOD
        """
        print("PlotModeshapes() method is deprecated. " + 
              "Use plot_modeshapes instead")
        self.plot_modeshapes(*args,**kwargs)
    
    
    def plot_modeshapes(self,
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
            
        # Get ordinates to plot
        modeshape_func = self.modeshapeFunc
        
        if isinstance(modeshape_func,scipy.interpolate.interpolate.interp1d):
            
            # Retrieve modeshape ordinates defining interpolation function
            x = modeshape_func.x
            m = modeshape_func.y
            L = x[-1]
            
        else:
            
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
        loadVals = loading_obj.evaluate_loads(t=0.0)
        
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
            
            loads_arr = loading_obj.evaluate_loads(t=t)
            QVals = npy.asmatrix(modeshapeVals.T) @ loads_arr 
            QVals = npy.ravel(QVals)
            return QVals
        
        return ModalForces
            
    
    def AppendTMDs(self,chainage_TMD,mass_TMD,freq_TMD,eta_TMD,
                   modeshape_TMD=None,
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
        if not hasattr(self,"modeshapeFunc") and modeshape_TMD is None:
            raise ValueError("Error: you must run 'DefineModeshapes' first!")
            
        # Define diagonal mass matrix for modes and TMDs
        M_a = self._M_mtrx
        C_a = self._C_mtrx
        K_a = self._K_mtrx
        Nm = M_a.shape[0]
        
        # Define TMD matrices
        omega_TMD = npy.asmatrix(npy.diagflat(angularFreq(freq_TMD)))
        eta_TMD = npy.asmatrix(npy.diagflat(eta_TMD))
        
        M_T = npy.asmatrix(npy.diagflat(mass_TMD))
        K_T = omega_TMD * omega_TMD * M_T
        C_T = 2 * eta_TMD * omega_TMD * M_T
        N_T = M_T.shape[0]

        # Use modeshape function to obtain modeshape ordinates at TMD positions
        if modeshape_TMD is None:
            
            # Use interpolation function already defined to get TMD modeshapes
            modeshape_TMD = self.modeshapeFunc(chainage_TMD)
            
        else:
            
            # Modeshape data to be provided directly via array
            pass
        
        # Check dimesions ok
        if modeshape_TMD.shape!=(N_T,Nm):
            raise ValueError("Error: `modeshape_TMD` shape (N_T,Nm) required\n"+
                             "Shape: {0}".format(modeshape_TMD.shape))
        
        self.chainage_TMD = chainage_TMD
        self.modeshape_TMD = modeshape_TMD
        
        # Determine mode-TMD mass matrix
        m1 = npy.hstack((M_a,npy.asmatrix(npy.zeros((Nm,N_T)))))
        m2 = npy.hstack((npy.asmatrix(npy.zeros((N_T,Nm))),M_T))
        M_aT = npy.vstack((m1,m2))
        
        # Define B_aT matrix
        B_aT = npy.hstack((modeshape_TMD,-npy.identity(N_T)))
        
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
        self._M_mtrx = M_aT
        self._C_mtrx = C_aT
        self._K_mtrx = K_aT
        
        nDOF = M_aT.shape[0]
        self.nDOF = nDOF
        
        self.J_mtrx = npy.asmatrix(npy.zeros((0,nDOF)))
        self.output_mtrx = output_mtrx
        self.output_names = output_names
        
        
    def CalcModeshapeIntegral(self,weighting_func=None,track_length=None,num=1000,power:int=1):
        """
        Evaluates integral along modeshape
        
        Prior to integration, modeshape ordinates are raised to `power`. E.g. 
        use `power=2` to evaluating integral of modeshape-squared (which is a 
        common application for this method)
        """
        
        modeshape_func = self.modeshapeFunc
                    
        # Evaluate modeshape ordinates
        if isinstance(modeshape_func,scipy.interpolate.interpolate.interp1d):
            
            # Retrieve modeshape ordinates defining interpolation function
            x = modeshape_func.x
            vals = modeshape_func.y
            
        else:
            
            if track_length is None:
                raise ValueError("`track_length` to be defined!")
            
            x = npy.linspace(0,track_length,num)
            vals = modeshape_func(x)
        
        # Take square of modeshape
        vals = vals**power
        
        # Evaluate and multiply by weighting function, if defined:
        if weighting_func is not None:
            
            if isinstance(weighting_func,float):
                vals = vals * weighting_func
            else:
                weighting_vals = weighting_func(x)
                vals = vals * weighting_vals
        
        # Integrate along track
        integral = scipy.integrate.trapz(y=vals, x=x, axis=0)
        
        return integral
    
    
    def PlotSystem(self,ax,v):
        """
        Plot system in deformed configuration as given by `v`
        """
        
        self.PlotSystem_init_plot(ax)
        self.PlotSystem_update_plot(v)
        
    
    def PlotSystem_init_plot(self,ax,plot_env=True):
        """
        Method for initialising system displacement plot
        """
                
        # Get modeshape function and salient x coordinates to use
        self.x = self.modeshapeFunc.x

        # Variables used to generate plot data
        self.y_env_max = 0.0 * self.x
        self.y_env_min = 0.0 * self.x

        # Define drawing artists
        self.lines = {}
        
        self.lines['y_res'] = ax.plot([], [],'k-',label='y(t)')[0]
        
        self.plot_env = plot_env
        if plot_env:        
            
            self.lines['y_env_max'] = ax.plot(self.x,
                                              self.y_env_max,
                                              color='r',alpha=0.3,
                                              label='$y_{max}$')[0]
            
            self.lines['y_env_min'] = ax.plot(self.x,
                                              self.y_env_min,
                                              color='b',alpha=0.3,
                                              label='$y_{min}$')[0]
        
        # Set up plot parameters
        ax.set_xlim(0, self.Ltrack)
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel("Displacement (m)")
        
    
    def PlotSystem_update_plot(self,v):
        """
        Method for updating system displacement plot given displacements `v`
        """
        
        # Calculate displacements along structure at time t, given modal disp v
        y = v.T @ self.modeshapeFunc(self.x).T

        # Update envelopes
        self.y_env_max = npy.maximum(y,self.y_env_max)
        self.y_env_min = npy.minimum(y,self.y_env_min)       
        
        # Update plot data
        self.lines['y_res'].set_data(self.x,y)
        
        if self.plot_env:
            self.lines['y_env_max'].set_data(self.x,self.y_env_max)
            self.lines['y_env_min'].set_data(self.x,self.y_env_min)
        
        return self.lines
    
# --------------- FUNCTIONS ------------------
        
def MAC(x1,x2):
    """
    Modal assurance criterion for comparing two complex-valued vectors 
    `x1` and `x2`
    
    ***
    
    $$ 
    MAC = (x_{2}^{H}.x_{1} + x_{1}^{H}.x_{2})/
    (x_{2}^{H}.x_{2} + x_{1}^{H}.x_{1}) 
    $$
    
    MAC is a scalar _float_ in the range [0.0,1.0]:
        
    * MAC = 1.0 implies vectors are exactly the same
    
    * MAC = 0.0 implies vectors are othogonal i.e. have no shared component
    
    """
       
    x1 = npy.asmatrix(x1)
    x2 = npy.asmatrix(x2)
    
    # Check dimensions are consistent
    if x1.shape!=x2.shape:
        raise ValueError("Error: x1 and x2 must be same shape!")
    
    # Calculate numerator and denominator of MAC function
    num = x2.H * x1 * x1.H * x2
    den = x2.H * x2 * x1.H * x1
    MAC = num/den
    MAC = npy.real(MAC)    # note should have negligible imag part anyway
        
    return MAC
    
  
# ********************** TEST ROUTINE ****************************************
# (Only execute when running as a script / top level)
        
if __name__ == "__main__":
    
    pass
    
    
    
    
