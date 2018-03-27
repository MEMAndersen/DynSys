# -*- coding: utf-8 -*-
"""
Classes and methods used to implement specific forms of dynamic analysis
"""

from __init__ import __version__ as currentVersion

import numpy
import timeit
import itertools
import matplotlib.pyplot as plt
import dill
import os
#import multiprocessing

import tstep
import loading
import msd_chain

def load(fName):
    """
    De-serialises `Dyn_Analysis` object from pickle fill
    """
    
    with open(fName,'rb') as fobj:
        print("\nLoading serialised `tstep_results` object from `{0}`".format(fName))
        obj = dill.load(fobj)
        
    print("De-serialisation successful!\n")
    
    return obj

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
            
        # Check class name of loadtrain_obj
        if loadtrain_obj.__class__.__name__ != "LoadTrain":
            raise ValueError("`loadtrain_obj`: instance of `LoadTrain` class expected!")
        
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
        
        forceFunc = modalsys_obj.CalcModalForces(loading_obj=loadtrain_obj,
                                                 loadVel=loadVel,
                                                 dt=dt_loads)
        
        
        self.tstep_obj = tstep.TStep(modalsys_obj,
                                     name=name,
                                     tStart=tStart,
                                     tEnd=tEnd,
                                     dt=dt,
                                     max_dt=max_dt,
                                     force_func=forceFunc,
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
        
        
        
    def run(self,
            saveResults=True,
            save_fName=None):
        """
        Runs moving load analysis, using `tstep.run()`
        
        _Refer documentation for that function for more details_
        """
        print("***** Running `%s`..." % self.__class__.__name__)
        print("Dynamic system: {0}".format(self.dynsys_obj.name))
        print("Load pattern: {0}".format(self.loading_obj.name))
        print("Load velocity: %.1f" % self.loadVel)
        tic=timeit.default_timer()
        self.tstep_obj.run()
        toc=timeit.default_timer()
        print("***** Analysis complete after %.3f seconds." % (toc-tic))
               
        if saveResults:
            self.save(fName=save_fName)
        
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
                 className:str,
                 dynsys_obj:object,
                 writeResults2File:bool=False,
                 retainDOFTimeSeries:bool=False,
                 retainResponseTimeSeries:bool=False,
                 **kwargs):
        """
        Initialisation function
        ****
        
        Required:
            
        * `className`, _string_ to denote `Dyn_Analysis` class (usually inherited) 
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
    
        print("Initialising multiple `{0}`".format(className))
    
        if className == "MovingLoadAnalysis":
            
            ReqdClass = MovingLoadAnalysis
            
            kwargs2permute = ["loadVel","loadtrain_obj"]
        
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
            
            # Initialisise analysis object
            analysis_list.append(ReqdClass(name="%04d"% i,
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
        
        print("Running multiple `{0}`".format(self.analysisType))
        tic=timeit.default_timer()
            
        # Run analyses using parallel processing (if possible)
        if solveInParallel:
            print("Parallel processing not yet implemented due to " + 
                  "outstanding bug associated with using "+
                  "`multiprocessing` module from within Spyder\n"+
                  "A single-process analysis will be carried out instead.")    

        # Run all pre-defined analyses
        for i, x in enumerate(self.analysis_list):
            
            print("Analysis #%04d of #%04d" % (i, len(self.analysis_list)))
            x.run(saveResults=False)
            print("")#clear line for emphasis
        
        toc=timeit.default_timer()
        print("Multiple `%s` complete after %.3f seconds!\n" % (self.analysisType,(toc-tic)))
            
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
    
    
    def plot_stats(self,stat_name='absmax',
                   key2plot='loadVel',
                   xConversionFactor:float=1.0,
                   xlabel=None,
                   figsize_inches=(14,8)):
        """
        Produces a plot of a given taken across multiple analyses
        
        Optional:
        
        * `stat_name`, name of statistic to be plotted
        * `key2plot`, name of key within `vals2permute` dict to be used
          as x-axis in plot
        """
        
        raise ValueError("UNFINISHED: DO NOT USE!")
        
        # Re-collate statistics as required
        if self.stats_dict is None:
            self.collate_stats()
            
        # Check requested stats is in dict
        if not stat_name in self.stats_dict:
            raise ValueError("Invalid statistic selected!")
            
        # Obtain responses names
        responseNames = self.dynsys_obj.output_names_list
        print(responseNames)
        
        # Get index and value to use along x-axis
        #kwargs2permute = list(self.vals2permute.keys())
        #key2plot_index=[i for i, x in enumerate(kwargs2permute) if x==key2plot][0]
        
        x_vals = xConversionFactor * numpy.array(self.vals2permute[key2plot])

        # Retrieve stats to plot
        stats_arr = self.stats_dict[stat_name]
        print("stats_arr.shape: {0}".format(stats_arr.shape))
        
        """
        Current manual workaround to produce load vs velocity plot!
        (Code should be tidied to be more generic)
        """
        
        # Create figure for each set of responses
        nFigures = len(responseNames)
        fig_list = []
        
        for fig_index in range(nFigures):
                
            nSubplots = len(responseNames[fig_index])
            
            fig, axarr = plt.subplots(nSubplots, sharex=True)
            fig.set_size_inches(figsize_inches)
            
            fig_list.append(fig)
            
            # Create subplots for each response
            for splt_index in range(nSubplots):
                
                r = splt_index
                ax = axarr[splt_index]
                vals2plot = stats_arr[:,:,r]
                
                ax.plot(x_vals,vals2plot)
                ax.set_title(responseNames[fig_index][r])
                
                if (xlabel is not None) and (splt_index==nSubplots-1):
                    ax.set_xlabel(xlabel)
                
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
        Collates computed statistics into a dict of ndarrays, to faciliate 
        efficient slicing operations, for example
        """
        
        print("Collating statistics...")
        
        if self.results_arr is None:
            raise ValueError("self.results_arr=None! No results to collate!")
        
        # Flatten array to list
        results_list = numpy.ravel(self.results_arr).tolist()
        
        # Collate ndarray of 'absmax' stats
        def collate_specified_stats(stats_name = 'max'):
            
            stats_list = []
            
            for i, results_obj in enumerate(results_list):
                
                stats_dict = results_obj.response_stats                    
                stats_vals = stats_dict[stats_name].tolist()
                if i==0: nResponses = len(stats_vals)
                stats_list.append(stats_vals)
            
            # Flatten nested list
            arr = numpy.ravel(stats_list)
            
            # Reshape as ndarray
            newshape = self.vals2permute_shape + (nResponses,)
            arr = numpy.reshape(arr,newshape)
            
            print("`{0}` stats saved as {1} ndarray".format(stats_name,arr.shape))
            
            return arr

        # Loop over all stats
        stats_keys = list(results_list[0].response_stats.keys())
        stats_dict={}

        for _key in stats_keys:
            
            stats_dict[_key] = collate_specified_stats(_key)
            
        # Save as attribute
        self.stats_dict = stats_dict
        
        return stats_dict
    
# ********************** FUNCTIONS   ****************************************
    
def ResponseSpectrum(accFunc,
                     T_vals=None,
                     tResponse=10.0,
                     eta = 0.05,
                     makePlot = True,
                     **kwargs):
    """
    Function to express ground acceleration time series as a seismic response 
    spectrum
    ***
    
    A _seismic response spectum_ summarise the peak acceleration response of a 
    SDOF oscillator in response to ground acceleration time series. Seismic 
    response spectra therefore represent a useful way of quantifying and 
    graphically illustrating the severity of a given ground acceleration time 
    series
    
    ***
    Required:
    
    * `accFunc`, function a(t) defining the ground acceleration time series 
      (usually this is most convenient to supply via an interpolation function)
      
    ***
    Optional:
        
    * `tResponse`, time interval over which to carry out time-stepping
      (set this to be at least the duration of the input acceleration time 
      series!)
      
    * `T_vals`, _list_, periods at which response spectrum to be evaluated
    
    * `eta`, damping ratio to which response spectrum obtained is applicable 
      (5% used by default as this is the default assumption in seismic design)
      
     `kwargs` may be used to pass additional arguments down to `TStep` object 
     that is used to implement time-stepping analysis. Refer `tstep` docs for 
     further details
    
    """
    
    # Handle optional inputs
    if T_vals is None:
        T_vals = numpy.arange(0.01,1.27,0.05)
        
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
    
    for _T in T_vals:

        period_str = "Period %.2fs" % _T
        
        # Define SDOF oscillator
        SDOF_sys = msd_chain.MSD_Chain(name=period_str,
                                       M_vals = M,
                                       f_vals = 1/_T,
                                       eta_vals = eta,
                                       showMsgs=False)
        
        # Add output matrix to extract results
        SDOF_sys.AddOutputMtrx(output_mtrx=numpy.identity(3),
                               output_names=["Disp","Vel","Acc"])
        
        # Define forcing function
        def forceFunc(t):
            return M*accFunc(t)
        
        # Define time-stepping analysis
        tstep_obj = tstep.TStep(SDOF_sys,
                                tStart=0, tEnd=tResponse,
                                force_func=forceFunc,
                                retainResponseTimeSeries=True)
        
        # Run time-stepping analysis and append results
        results_list.append(tstep_obj.run(showMsgs=False))
        
        # Obtain absolute acceleration by adding back in ground motion
        results_obj = tstep_obj.results_obj
        results_obj.responses[2,:] += accFunc(results_obj.t.T)
        
        # Recalculate statistics
        results_obj.CalcResponseStats(showMsgs=False)
        
        # Tidy up
        del SDOF_sys
    
    # Collate absmax statistics
    print("Retrieving maximum response statistics...")
    S_D = numpy.asarray([x.response_stats['absmax'][0] for x in results_list])
    S_V = numpy.asarray([x.response_stats['absmax'][1] for x in results_list])
    S_A = numpy.asarray([x.response_stats['absmax'][2] for x in results_list])
    
    if makePlot:
        
        fig, axarr = plt.subplots(3, sharex=True)
        
        fig.suptitle("Response spectra")
        
        ax = axarr[0]
        ax.plot(T_vals,S_D)
        ax.set_ylabel("SD (m)")
        ax.set_title("Relative displacement response spectrum")
        
        ax = axarr[1]
        ax.plot(T_vals,S_V)
        ax.set_ylabel("SV (m/s)")
        ax.set_title("Relative velocity response spectrum")
        
        ax = axarr[2]
        ax.plot(T_vals,S_A)
        ax.set_ylabel("SA (m/$s^2$)")
        ax.set_title("Absolute acceleration response spectrum")
        
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
    
    testRoutine=4
    
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
        
        rslts = run_multiple("MovingLoadAnalysis",
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