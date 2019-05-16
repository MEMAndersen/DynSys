# -*- coding: utf-8 -*-
"""
Classes and methods used to implement specific forms of dynamic analysis

@author: rihy
"""

from __init__ import __version__ as currentVersion

import numpy
import timeit
import itertools
import matplotlib.pyplot as plt
import dill
import pandas as pd
#import multiprocessing   # does not work with Spyder!

import tstep
import loading
import msd_chain
from common import chunks

#%%

class Dyn_Analysis():
    """
    Base class for dynamic analysis implementations
    
    _Inheritance expected_
    """
    
    def __init__(self,name:str,dynsys_obj,loading_obj):
        """
        Initialisation function
        
        _All derived classes are expected to run this function_
        """
        
        self.name = name
        """
        String identifier for object
        """
        
        self.dynsys_obj = dynsys_obj
        """
        Dynamic system to which analysis relates
        """
        
        self.loading_obj = loading_obj
        """
        Object defining applied loading
        """
    
    def run(self):
        """
        Runs dynamic analysis
        
        _Inheritance expected_
        """
        pass
    
    def _pickle_fName(self,fName):
        """
        Defines default filename for pickle files
        """
        if fName is None:
            
            fName = "{0}".format(self.__class__.__name__)
            
            if self.name is not None:
                fName += "{0}".format(self.name)
                
            fName += ".pkl" 
                
        return fName
    
    def save(self,fName=None):
        """
        Serialises object to file `fName`
        """
    
        fName=self._pickle_fName(fName)

        with open('{0}'.format(fName), 'wb') as dill_file:
            print("Serialising `{0}` object to `{1}`".format(self.__class__.__name__,fName))         
            dill.dump(self, dill_file)
            
        print("Serialisation complete!")
        
    
    
    
class MovingLoadAnalysis(Dyn_Analysis):
    """
    Class to implement moving load analysis
    ***
    
    _Moving load analysis_ involves the determining the response of the system 
    to groups of point loads moving in a pre-defined manner along a defined 
    track.
    
    _Moving loads_ include, but are not limited to:
    
    *   Train axles, which are usually at pre-defined spacings
    
    *   Vehicle axles
    
    *   Forces to represent the actions of walkers/joggers per UK NA to 
        BS EN 1991-2
        
    """
    
    def __init__(self,
                 modalsys_obj,
                 name=None,
                 loadtrain_obj=None,
                 loadtrain_fName="loadDefs.csv",
                 loadVel=5.0,
                 use_abs_modeshape=False,
                 tEpilogue=10.0,
                 dt=None,
                 dt_loads=0.01,
                 max_dt=None,
                 retainDOFTimeSeries=True,
                 retainResponseTimeSeries=True,
                 writeResults2File=False,
                 results_fName="results.csv"):
        """
        Initialisation function
        
        ***
        Required:
            
        * `modalsys_obj`, modal system to which analysis relates
        
        **Important note:** `modalsys_obj` must be a modal system, as the 
        action of the moving loads is determined by obtaining the 
        mode-generalised force functions for each mode. _This is checked_.
        
        ***
        Optional:
            
        * `loadtrain_obj`, load train object defining load pattern. If _None_ 
          then `loadtrain_fName` must be provided (see below)
          
        * `loadtrain_fName", file containing load train definition
            
        * `loadVel`, constant velocity of load pattern (m/s)
        
        * `loadDefs_fName`, file containing load definitions
        
        
        """
        
        # Handle None for loadtrain_obj
        if loadtrain_obj is None:
            loadtrain_obj = loading.LoadTrain(fName=loadtrain_fName)
        
        # Check class name of modalsys_obj
        if modalsys_obj.__class__.__name__ != "ModalSys":
            raise ValueError("`modalsys_obj`: instance of `ModalSys` class expected!")
            
        # Check loadtrain_obj is of class 'LoadTrain' or derived class

        if loadtrain_obj.__class__.__name__ != "LoadTrain":
        
            base_class_name = loadtrain_obj.__class__.__bases__[0].__name__

            if base_class_name != "LoadTrain":
                
                raise ValueError("`loadtrain_obj`: instance of `LoadTrain` "+
                                 "class (or derived classes) expected!\n" + 
                                 "`loadtrain_obj` base class: %s" 
                                 % base_class_name)
        
        # Run parent init
        super().__init__(name,modalsys_obj,loadtrain_obj)
        
        # Save details as attributes
        self.loadVel = loadVel
        """
        Velocity of load pattern along track
        """
        
        # Define time-stepping analysis
        
        tStart, tEnd = self._CalcSimDuration(loadVel=loadVel,
                                             tEpilogue=tEpilogue)
        
        # Define force function for parent system and subsystems
        modalForces_func = modalsys_obj.CalcModalForces(loading_obj=loadtrain_obj,
                                                        loadVel=loadVel,
                                                        dt=dt_loads,
                                                        use_abs_modeshape=use_abs_modeshape)
        
        force_func_dict = {modalsys_obj : modalForces_func}
        
        self.tstep_obj = tstep.TStep(modalsys_obj,
                                     name=name,
                                     tStart=tStart,
                                     tEnd=tEnd,
                                     dt=dt,
                                     max_dt=max_dt,
                                     force_func_dict=force_func_dict,
                                     retainDOFTimeSeries=retainDOFTimeSeries,
                                     retainResponseTimeSeries=retainResponseTimeSeries,
                                     writeResults2File=writeResults2File,
                                     results_fName=results_fName
                                     )
        """
        Time-stepping solver object
        """
        
        self.results_obj = self.tstep_obj.results_obj
        """
        Results object
        """
        
        # Create relationship to this analysis object
        self.results_obj.analysis_obj = self
        
        
    
    def run(self,
            verbose=True,
            saveResults=False,
            save_fName=None):
        """
        Runs moving load analysis, using `tstep.run()`
        
        _Refer documentation for that function for more details_
        """
        
        if verbose:
            print("***** Running `%s`..." % self.__class__.__name__)
            print("Dynamic system: '{0}'".format(self.dynsys_obj.name))
            print("Load pattern: '{0}'".format(self.loading_obj.name))
            print("Load velocity: %.1f" % self.loadVel)
        
        tic=timeit.default_timer()
        
        results_obj = self.tstep_obj.run(verbose=verbose)
        
        toc=timeit.default_timer()
        
        if verbose:
            print("***** Analysis complete after %.3f seconds." % (toc-tic))
               
        if saveResults:
            self.save(fName=save_fName)
            
        return results_obj
    
        
    def _CalcSimDuration(self,loadVel=10.0,tEpilogue=5.0):
        """
        Calculates the required duration for time-stepping simulation
        ***
        
        Optional:
         
        * `loadVel`, constant velocity (m/s) of the load pattern
        
        * `tEpilogue`, additional time to run following exit of last load from 
          track overlying the dynamic system in question.
          
        In the case of `Ltrack` and `loadLength`, if _None_ is provided then 
        function will attempt to obtain this information from class attributes.
          
        ***
        Returns:
            
        tStart, `tEnd`: start and end times (secs) for simulation
         
        """
        
        #print("tEpilogue = %.3f" % tEpilogue)
        
        modalsys_obj = self.dynsys_obj
        loadtrain_obj = self.loading_obj
        
        # Get length of track along which loading is running
        attr = "Ltrack"
        obj = modalsys_obj
        if hasattr(obj,attr):
            Ltrack = getattr(obj,attr)
        else:
            raise ValueError("`{0}` not defined!".format(attr))
            
        # Get length of load train
        attr = "loadLength"
        obj = loadtrain_obj
        if hasattr(obj,attr):
            loadLength = getattr(obj,attr)
        else:
            raise ValueError("`{0}` not defined!".format(attr))
      
        # Determine time required for all loads to pass along track
        tStart=0
        Ltotal = Ltrack + loadLength
        tEnd = Ltotal/loadVel + tEpilogue
        
        return tStart, tEnd
    
    
    def PlotResults(self,dofs2Plot=None):
        """
        Plots results using `tstep_results.PlotResults()`.
        
        _Refer documentation from that function for further details_
        """
        self.results_obj.PlotResults(dofs2Plot=dofs2Plot)
     
            

class Multiple():
    """
    Function to run multiple dynamic analyses and provide functionality to 
    store, plot and analyse results from multiple dynamic analyses in a 
    systematic manner
    """
    
    def __init__(self,
                 classDef,
                 dynsys_obj:object,
                 writeResults2File:bool=False,
                 retainDOFTimeSeries:bool=False,
                 retainResponseTimeSeries:bool=False,
                 **kwargs):
        """
        Initialisation function
        ****
        
        Required:
            
        * `classDef`, `Dyn_Analysis` class definition (usually inherited) 
          that implements the required analysis type
          
        * `dynsys_obj`, dynamic system to which analysis relates
        
        ***
        Optional:
            
        * `retainResponseTimeSeries`, _boolean_, denotes whether detailed 
          _response_ time series results can be deleted once summary 
          statistics have been computed
          
        * `retainDOFTimeSeries`, _boolean_, denotes whether detailed _DOF_ 
          time series results can be deleted once summary statistics have been 
          computed
            
        By default _False_ is assigned to be above (contrary to usual defaults) 
        as running multiple analyses would otherwise often lead to large 
        memory demands.
          
        """
    
        className = classDef.__name__
        print("Initialising multiple `{0}`".format(className))
    
        if className == "MovingLoadAnalysis":
            
            kwargs2permute = ["loadVel","loadtrain_obj"]
            
        elif className == "UKNA_BSEN1991_2_walkers_joggers":
            
            kwargs2permute = ["analysis_type","mode_index"]
        
        else:
            raise ValueError("Unsupported class name!")
        
        # Get input arguments to permute
        vals2permute={}
        for key in kwargs2permute:
            
            # Get list as supplied via **kwargs
            if key in kwargs:
                
                vals_list = kwargs[key]
                del kwargs[key]
                
                # Convert to list if single
                if not isinstance(vals_list,list):
                        vals_list = [vals_list]
                        
                vals2permute[key]=vals_list
                        
            else:
                raise ValueError("'{0}' ".format(key) + 
                                 "included in `kwargs2permute` list but " + 
                                 "list of values to permute not provided!")

        # Populate object array with initialised objects
        vals_list = list(itertools.product(*vals2permute.values()))
        
        analysis_list=[]
        nAnalysis = len(vals_list)
        for i in range(nAnalysis):
            
            # Prepare dict of key arguments to pass
            kwargs_vals = vals_list[i]
            kwargs_dict = dict(zip(kwargs2permute, kwargs_vals))
            
            # Append any general kwargs provided
            results_fName = "results/analysis%04d.csv" % (i+1)
            
            kwargs_dict.update(kwargs)
            kwargs_dict.update({"retainDOFTimeSeries":retainDOFTimeSeries,
                                "retainResponseTimeSeries":retainResponseTimeSeries,
                                "writeResults2File":writeResults2File,
                                "results_fName":results_fName})
            
            # Initialisise new analysis object
            analysis_list.append(classDef(name="%04d"% i,
                                           modalsys_obj=dynsys_obj,
                                           **kwargs_dict))
            
        
        self.dynsys_obj = dynsys_obj
        """
        `DynSys` object to which analysis relates
        """
        
        self.analysisType = className
        """
        Class name of `dyn_analysis` derived class
        """
                    
        self.vals2permute = vals2permute
        """
        Dictionary of keywords and values to permute in the multiple analyses 
        defined
        """
        
        self.vals2permute_shape = tuple([len(x) for x in self.vals2permute.values()])
        """
        Tuple to denote the shape of lists specified in `vals2permute`
        """
        
        self.results_arr=None
        """
        ndarray of `tstep_results` object instances
        """
        
        self.stats_dict=None
        """
        Dict of ndarrays containing stats (max, min, std, absmax) for each of 
        the analyses carried out, for each of the responses defined
        """
            
        self.analysis_list = analysis_list
        """
        List of `Dyn_Analysis` object instances, each of which defines a 
        dynamic analysis to be performed
        """
        
    def run(self,save=True,solveInParallel=False):
        """
        Runs multiple dynamic analyses, as defined by `__init__`
        ***
        
        In principle this can be done in parallel, to efficiently use all 
        avaliable cores and give improve runtime.
        
        _Parallel processing not yet implemented due to outstanding bug 
        associated with using `multiprocessing` module from within Spyder_
        """
        
        print("Running multiple `{0}`\n".format(self.analysisType))
        tic=timeit.default_timer()
            
        # Run analyses using parallel processing (if possible)
        if solveInParallel:
            print("Parallel processing not yet implemented due to " + 
                  "outstanding bug associated with using "+
                  "`multiprocessing` module from within Spyder\n"+
                  "A single-process analysis will be carried out instead.")    

        # Run all pre-defined analyses
        for i, x in enumerate(self.analysis_list):
            
            print("Analysis #%04d of #%04d" % (i+1, len(self.analysis_list)))
            x.run(saveResults=False)
            print("")#clear line for emphasis
        
        toc=timeit.default_timer()
        print("Multiple `%s` analysis complete after %.3f seconds!\n" 
              % (self.analysisType,(toc-tic)))
            
        # Reshape results objects in ndarray
        results_obj_list = [x.tstep_obj.results_obj for x in self.analysis_list]
        reqdShape = self.vals2permute_shape
        results_arr = numpy.reshape(results_obj_list,reqdShape)
        self.results_arr = results_arr
        
        # Collate statistics
        self.collate_stats()
        
        # Pickle results
        if save:
            self.save()
    
    
    def plot_stats(self,
                   stat='absmax',
                   sys=None,
                   **kwargs):
        """
        Produces a plot of a given statistic, taken across multiple analyses
        
        Optional:
        
        * `stat`, name of statistic to be plotted (e.g. 'max', 'min')
        
        * `sys`, instance of `DynSys` class, or string to denote name of system,
          used to select system whose outputs are to be plotted. 
          If None then seperate figures will be produced for each subsystem.
          
        _See docstring for `plot_stats_for_system()` for other keyword 
        arguments that may be passed._
        
        """
        
        # Get list of system names to loop over
        if sys is None:
            sys_names = [obj.name for obj in self.dynsys_obj.DynSys_list]
        else:
            
            # Get name of system
            if isinstance(sys,str):
                sys_name = sys
            else:
                sys_name = sys.name # get name from object
            
            sys_names = [sys_name] # list of length 1
            
        # Produce a seperate figure for each sub-system responses
        fig_list = []
        
        fig_list = []
        for sys_name in sys_names:
            
            _fig_list = self.plot_stats_for_system(sys_name=sys_name,
                                                   stat=stat,
                                                   **kwargs)
            
            fig_list.append(_fig_list)
            
        # Return list of figures, one for each subsystem
        return fig_list
    
    
    def plot_stats_for_system(self,sys_name,stat,
                              max_responses_per_fig:int=None,
                              subplot_kwargs={}
                              ):
        """
        Produces a plot of a given statistic, taken across multiple analyses
        for a specified sub-system
        
        Required:
        
        * `sys_name`, string giving name of system
        
        * `stat`, string to specify statistic to be plotted. E.g. 'absmax'
        
        Optional:
            
        * `max_responses_per_fig`, integer to denote maximum number of 
          responses to be plotted in each figure. If None, all responses will 
          be plotted via a single figure
          
        * `subplot_kwargs`, dict of keyword arguments to be passed to 
          `pyplot.subplots()` method, to customise subplots (e.g. share axes)        
          
        """
        
        # Re-collate statistics as required
        if self.stats_df is None:
            stats_df = self.collate_stats()
        else:
            stats_df = self.stats_df
        
        # Slice for requested stat
        try:
            stats_df = stats_df.xs(stat,level=-1,axis=1)
        except KeyError:
            raise KeyError("Invalid statistic selected!")
            
        # Get stats for just this system
        df_thissys = stats_df.xs(sys_name,level=0,axis=0)
            
        # Obtain responses names for this subsystem
        response_names = df_thissys.index.values
        nResponses = len(response_names)
        
        if max_responses_per_fig is None:
            max_responses_per_fig = nResponses
        
        fig_list = []
        
        for _response_names in chunks(response_names,max_responses_per_fig):
            
            # Create figure, with one subplot per response
            fig, axlist = plt.subplots(len(_response_names),
                                       sharex=True,
                                       **subplot_kwargs)
            
            fig_list.append(fig)
        
            for i, (r,ax) in enumerate(zip(_response_names,axlist)):
                                
                # Get series for this response
                df = df_thissys.loc[r]
                
                # Reshape such that index will be x-variable for plot
                df = df.unstack()
                
                # Make plot
                ax = df.plot(ax=ax,legend=False)
                
                ax.set_ylabel(r,
                              rotation=0,
                              fontsize='small',
                              horizontalAlignment='right',
                              verticalAlignment='center')
                
                if i==0:
                    # Add legend to figure
                    fig.legend(ax.lines, df.columns,fontsize='x-small') 
                    
            fig.subplots_adjust(left=0.15,right=0.95)
            fig.align_ylabels()
                    
        return fig_list            
        
    
    def _pickle_fName(self,fName):
        """
        Defines default filename for pickle files
        """
        if fName is None:
            
            fName = "{0}_{1}".format(self.__class__.__name__,self.analysisType)
            
            if self.dynsys_obj.name is not None:
                fName += "_{0}".format(self.dynsys_obj.name)
                
            fName += ".pkl" 
                
        return fName
    
    def save(self,fName=None):
        """
        Serialises object to file `fName`
        """
    
        fName=self._pickle_fName(fName)

        with open('{0}'.format(fName), 'wb') as dill_file:
            print("\nSerialising `{0}` object to `{1}`".format(self.__class__.__name__,fName))         
            dill.dump(self, dill_file)
            
        print("Serialisation complete!\n")
    
    
    def collate_stats(self):
        """
        Collates computed statistics into a Pandas DataFrame, as follows:
            
        * Index is a MultiIndex, comprising the following levels:
            
            * 0 : Name of sub-system
            
            * 1 : Name of output / response
            
        * Columns are a MultiIndex comprising the following levels:
            
            * 0 to -2 : Variables of analysis `vals2permute`
            
            * -1 : Statistic name (e.g. 'max', 'min')
            
        """
        
        print("Collating statistics...")
        
        # Get list of all tstep_results objects associate with Multiple()
        if self.results_arr is None:
            raise ValueError("self.results_arr=None! No results to collate!")
        
        # Get lists of inputs permuted for analyses
        kwargs2permute = list(self.vals2permute.keys())
        vals2permute = list(self.vals2permute.values())
        
        # Where list contains objects, get their names
        for i in range(len(vals2permute)):
            
            if hasattr(vals2permute[i][0],'name'):
                vals2permute[i] = [x.name for x in vals2permute[i]]
        
        # Get list of systems and subsystems
        DynSys_list = self.dynsys_obj.DynSys_list
        
        stats_df = None
        
        for i, (index, results_obj) in enumerate(numpy.ndenumerate(self.results_arr)):
            
            # Get combination of vals2permute for results_obj
            combination = [vals[i] for i, vals in zip(index, vals2permute)]
            
            # Get stats from results_obj
            stats_df_inner = results_obj.get_response_stats_df()
            
            # Loop over all sub-systems
            for df, sys in zip(stats_df_inner,DynSys_list):
                
                # Prepend system to index
                df = pd.concat([df],
                               axis=0,
                               keys=[sys.name],
                               names=['System','Response'])
                
                # Prepare combination values to column MultiIndex
                tuples = [(*combination,col) for col in df.columns]
                df.columns = pd.MultiIndex.from_tuples(tuples)
                df.columns.names = [*kwargs2permute,'Statistic']
                
                # Append results to DataFrame
                stats_df = pd.concat([stats_df, df],axis=1)
            
        self.stats_df = stats_df
            
        return stats_df
    
    
   #%% 

        
        
    
# ********************** FUNCTIONS   ****************************************
  

def load(fName):
    """
    De-serialises object from pickle file, given `fName`
    """
    
    with open(fName,'rb') as fobj:
        print("\nLoading serialised  object from `{0}`".format(fName))
        obj = dill.load(fobj)
        
    print("De-serialisation successful!\n")
    
    return obj
      

def ResponseSpectrum(accFunc,
                     tResponse,
                     T_vals=None,
                     eta=0.05,
                     makePlot=True,
                     **kwargs):
    """
    Function to express ground acceleration time series as a seismic response 
    spectrum
    ***
    
    _Seismic response spectra_ are used to summarises the vibration response 
    of a SDOF oscillator in response to a transient ground acceleration 
    time series. Seismic response spectra therefore represent a useful way of 
    quantifying and graphically illustrating the severity of a given 
    ground acceleration time series.
    
    
    
    ***
    Required:
    
    * `accFunc`, function a(t) defining the ground acceleration time series 
      (usually this is most convenient to supply via an interpolation function)
      
    * `tResponse`, time interval over which to carry out time-stepping analysis.
      Set this to be at least the duration of the input acceleration time 
      series!
      
    **Important note**:
        
    This routine expects a(t) to have units of m/s<sup>2</sup>. 
    It is common (at least in the field of seismic analysis) to quote ground 
    accelerations in terms of 'g'.
    
    Any such time series must be pre-processed by multiplying by 
    g=9.81m/s<sup>2</sup>) prior to using this function, such that the 
    supplied `accFunc` returns ground acceleration in m/s<sup>2</sup>.
      
    ***
    Optional:
        
    * `T_vals`, _list_, periods (in seconds) at which response spectra
      are to be evaluated. If _None_ will be set to logarithmically span the range 
      [0.01,10.0] seconds, which is suitable for most applications.
        
    * `eta`, damping ratio to which response spectrum obtained is applicable. 
      5% is used by default, as this a common default in seismic design.
      
     * `makePlot`, _boolean_, controls whether results are plotted
      
     `kwargs` may be used to pass additional arguments down to `TStep` object 
     that is used to implement time-stepping analysis. Refer `tstep` docs for 
     further details
     
    ***
    Returns:
        
    Values are returned as a dictionary, containing the following entries:
        
    * `T_vals`, periods at which spectra are evaluated.
    * `S_D`, relative displacement spectrum (in m)
    * `S_V`, relative velocity spectrum (in m/s)
    * `S_A`, absolute acceleration spectrum (in m/s<sup>2</sup>)
    * `PSV`, psuedo-velocity spectrum (in m/s)
    * `PSA`, psuedo-acceleration spectrum (in m/s<sup>2</sup>)
    
    In addition, if `makePlot=True`:
        
    * `fig`, figure object for plot
    
    """
    
    # Handle optional inputs
    if T_vals is None:
        T_vals = numpy.logspace(-2,1,100)
        
    T_vals = numpy.ravel(T_vals).tolist()
    
    # Print summary of key inputs
    if hasattr(accFunc,"__name__"):
        print("Ground motion function supplied: %s" % accFunc.__name__)
    print("Time-stepping analysis interval: [%.2f, %.2f]" % (0,tResponse))
    print("Number of SDOF oscillators to be analysed: %d" % len(T_vals))
    print("Damping ratio for SDOF oscillators: {:.2%}".format(eta))
    
    # Loop through all periods
    M = 1.0 # unit mass for all oscillators
    results_list = []
    
    print("Obtaining SDOF responses to ground acceleration...")
    
    for i, _T in enumerate(T_vals):

        period_str = "Period %.2fs" % _T
        
        if i % 10 == 0:
            print("    Period %d of %d" % (i,len(T_vals)))
        
        # Define SDOF oscillator
        SDOF_sys = msd_chain.MSD_Chain(name=period_str,
                                       M_vals = M,
                                       f_vals = 1/_T,
                                       eta_vals = eta,
                                       showMsgs=False)
        
        # Add output matrix to extract results
        SDOF_sys.AddOutputMtrx(output_mtrx=numpy.identity(3),
                               output_names=["RelDisp","RelVel","Acc"])
        
        # Define forcing function
        def forceFunc(t):
            return -M*accFunc(t)
        
        # Define time-stepping analysis
        tstep_obj = tstep.TStep(SDOF_sys,
                                tStart=0, tEnd=tResponse,
                                force_func_dict={SDOF_sys:forceFunc},
                                retainResponseTimeSeries=True)
        
        # Run time-stepping analysis and append results
        results_obj = tstep_obj.run(verbose=False)
        results_list.append(results_obj)
        
        # Obtain absolute acceleration by adding back in ground motion
        results_obj.responses_list[0][2,:] += accFunc(results_obj.t.T)
        
        # Recalculate statistics
        results_obj.CalcResponseStats(verbose=False)
        
        # Tidy up
        del SDOF_sys
    
    # Collate absmax statistics
    print("Retrieving maximum response statistics...")
    S_D = numpy.asarray([x.response_stats['absmax'][0] for x in results_list])
    S_V = numpy.asarray([x.response_stats['absmax'][1] for x in results_list])
    S_A = numpy.asarray([x.response_stats['absmax'][2] for x in results_list])
    
    # Evaluate psuedo-specta
    omega = numpy.divide(2*numpy.pi,T_vals)
    PSV = omega * S_D
    PSA = omega**2 * S_D
    
    if makePlot:
        
        fig, axarr = plt.subplots(3, sharex=True)
        
        fig.suptitle("Response spectra: {:.0%} damping".format(eta))
        
        ax = axarr[0]
        ax.plot(T_vals,S_D)
        ax.set_ylabel("SD (m)")
        ax.set_title("Relative displacement")
        
        
        ax = axarr[1]
        ax.plot(T_vals,S_V)
        ax.plot(T_vals,PSV)
        ax.legend((ax.lines),("$S_V$","Psuedo $S_V$",),loc='upper right')
        ax.set_ylabel("SV (m/s)")
        ax.set_title("Relative velocity")
        
        ax = axarr[2]
        ax.plot(T_vals,S_A)
        ax.plot(T_vals,PSA)
        ax.legend((ax.lines),("$S_A$","Psuedo $S_A$",),loc='upper right')
        ax.set_ylabel("SA (m/$s^2$)")
        ax.set_title("Absolute acceleration")
        
        ax.set_xlim([0,numpy.max(T_vals)])
        ax.set_xlabel("Oscillator natural period T (secs)")
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
    
    # Return values as dict
    return_dict = {}
    return_dict["T_vals"]=T_vals
    
    return_dict["S_D"]=S_D
    return_dict["S_V"]=S_V
    return_dict["S_A"]=S_A
    
    return_dict["PSV"]=PSV
    return_dict["PSA"]=PSA
    
    if makePlot:
        return_dict["fig"]=fig
    else:
        return_dict["fig"]=None
    
    return return_dict
    


def DesignResponseSpectrum_BSEN1998_1(T_vals=None,
                                      a_g=0.02,
                                      S=1.00,
                                      nu=1.00,
                                      T_BCD=[0.05,0.25,1.20]):
    """
    Returns horizontal elastic design response spectrum as given by 
    Cl. 3.2.2.2 of BS EN 1998-1:2004
    
    ![figure_3_1](../dynsys/img/design_response_spectrum_BSEN1998-1.png)
    ![equations](../dynsys/img/design_response_spectrum_BSEN1998-1_equations.png)
    
    Note response spectrum has units as given by `a_g` (units of 'g' typically)
    """
    
    if len(T_BCD)!=3:
        raise ValueError("`T_BCD` expected to be list of length 3")
        
    # Unpack list items for salient periods
    T_B = T_BCD[0]
    T_C = T_BCD[1]
    T_D = T_BCD[2]
    
    if T_vals is None:
        
        # Create default T_vals list to fully-illustrate key features
        T_vals = numpy.concatenate((numpy.linspace(0,T_B,10,endpoint=False),
                                    numpy.linspace(T_B,T_C,10,endpoint=False),
                                    numpy.linspace(T_C,T_D,10,endpoint=False),
                                    numpy.linspace(T_D,2*T_D,10,endpoint=True)))
    
    Se_vals = []
    
    for T in T_vals:
        
        if T < 0:
            raise ValueError("T_vals to be >0!")
        
        elif T < T_B:
            Se = 1 + (T / T_B)*(nu*2.5 - 1)
        
        elif T < T_C:
            Se = nu * 2.5
            
        elif T < T_D:
            Se = nu * 2.5 * (T_C / T)
            
        else:
            Se = nu * 2.5 * (T_C*T_D)/(T**2)
            
        Se = a_g * S * Se
        Se_vals.append(Se)
        
    return T_vals, Se_vals

  
    
# ********************** TEST ROUTINE ****************************************

if __name__ == "__main__":
    
    testRoutine=5
    
    if testRoutine==1:
        
        import modalsys
    
        modal_sys = modalsys.ModalSys(isSparse=False)
        modal_sys.AddOutputMtrx(fName="outputs.csv")
        
        def sine(t):
            return numpy.sin(5*t)
        
        loading_obj = loading.LoadTrain(loadX=0.0,loadVals=100.0,
                                        intensityFunc=sine,
                                        name="Sine test")
        
        #loading_obj = loading.LoadTrain()
        
        ML_analysis = MovingLoadAnalysis(modalsys_obj=modal_sys,
                                         dt=0.01,
                                         loadVel=20,
                                         loadtrain_obj=loading_obj,
                                         tEpilogue=5.0)
        ML_analysis.run()
        ML_analysis.PlotResults(dofs2Plot=[2,3,4])
        
    elif testRoutine==2:
        
        import modalsys
    
        modal_sys = modalsys.ModalSys(isSparse=False)
        modal_sys.AddOutputMtrx(fName="outputs.csv")
        
        loading_obj = loading.LoadTrain()
        
        ML_analysis = MovingLoadAnalysis(modalsys_obj=modal_sys,
                                         dt=0.01,
                                         loadVel=100,
                                         loadtrain_obj=loading_obj,
                                         tEpilogue=5.0)
        ML_analysis.run()
        ML_analysis.PlotResults(dofs2Plot=[2,3,4])
        
    
    elif testRoutine==3:
        
        import modalsys
    
        modal_sys = modalsys.ModalSys(isSparse=False)
        modal_sys.AddOutputMtrx(fName="outputs.csv")
        
        rslts = Multiple("MovingLoadAnalysis",
                         dynsys_obj=modal_sys,
                         loadVel=(numpy.array([380,390,400])*1000/3600).tolist(),
                         loadtrain_fName=["trainA2.csv","trainA5.csv"],
                         dt=0.01)
        
        [x.PlotResults() for x in rslts]
        
    
    elif testRoutine==4:    

        def test_func(t):
            return 0.02*numpy.sin(2.0*t)
        
        results = ResponseSpectrum(test_func,
                                   eta=0.005,
                                   T_vals=numpy.linspace(0.01,8.0,num=80))
        
    elif testRoutine==5:
        
        T_vals, Se_vals = DesignResponseSpectrum_BSEN1998_1()
        plt.plot(T_vals,Se_vals)

    else:
        raise ValueError("Test routine does not exist!")
