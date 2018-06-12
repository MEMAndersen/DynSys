# -*- coding: utf-8 -*-
"""
Classes and methods used to flexibly handle results from 
time-stepping analysis methods as implemented in `tstep` module
"""

from __init__ import __version__ as currentVersion

# ********************* IMPORTS **********************************************
import numpy as npy
import scipy.signal
import matplotlib.pyplot as plt
import timeit
import pandas
from datetime import datetime

class TStep_Results:
    """
    Class used to store, manipulate and plot results from time-stepping 
    analysis as carried out by `TStep`
    """
    # ********** CONSTRUCTOR / DESTRUCTOR ******************
    
    def __init__(self,tstep_obj,
                 calcDOFStats:bool=True,
                 calcResponseStats:bool=True,
                 retainDOFTimeSeries:bool=True,
                 retainResponseTimeSeries:bool=True):
        """
        Initialisation function
        ***
        
        Required:
        
        * `tstep_obj`, `TStep()` class instance to which results relate
        
        ***
        Optional
        
        * `calcDOFStats`, _boolean_, denotes whether statistics should be 
          computed to summarise DOF time series results
        
        * `calcResponseStats`, _boolean_, denotes whether statistics should be 
          computed to summarise response time series results
        
        * `retainResponseTimeSeries`, _boolean_, denotes whether detailed 
          _response_ time series results can be deleted once summary 
          statistics have been computed
          
        * `retainDOFTimeSeries`, _boolean_, denotes whether detailed _DOF_ 
          time series results can be deleted once summary statistics have been 
          computed
          
        Choosing _False_ for the above can be done to be more economical with 
        memory usage, for example when carrying out multiple analyses. Note 
        that providing DOF time series are retained, responses can always be 
        recomputed using `CalcResponses()` function
        """
        
        
        self.nDOF=None
        """
        Number of degrees of freedom for system being analysed
        """
        
        self.t=None
        """
        Time values
        """
        
        self.f=None
        """
        External applied forces
        """
        
        self.v=None
        """
        Displacements of analysis DOFs
        """
        
        self.vdot=None
        """
        Velocities of analysis DOFs
        """
        
        self.v2dot=None
        """
        Accelerations of analysis DOFs
        """
        
        self.f_constraint=None
        """
        Internal constraint forces
        """
        
        self.nResults=0
        """
        Number of time-steps at which results are recorded
        """
        
        self.tstep_obj=tstep_obj
        """
        Instance of `tstep` class, to which results relate
        """
        
        self.responses_list=[]
        """
        List of matrices to which computed responses are recorded 
        for each time step
        """
        
        self.response_names_list=[]
        """
        List of lists to described computed responses
        """
        
        self.calc_dof_stats=calcDOFStats
        """
        _Boolean_, denotes whether statistics should be computed to summarise 
        DOF time series results
        """
        
        self.calc_response_stats=calcResponseStats
        """
        _Boolean_, denotes whether statistics should be computed to summarise 
        response time series results
        """
        
        self.dof_stats={}
        """
        Dict of statistics, evaluated over all time steps, for each DOF
        
        _Set using `CalcDOFStats()`_
        """
        
        self.response_stats_dict={}
        """
        Dict of dict of statistics:
            
        * Primary dict: 
            
            * One entry for each subsystem
            
            * `DynSys` subsystem object pointer is used as key
        
        * Secondary dicts:
            
            * One entry for each statistic, with appropriate key string
            
            * Each entry is in general an array, obtained by evaluating 
              statistic over all time steps, for each defined response
        
        _Set using `CalcResponseStats()`_
        """

        self.retainDOFTimeSeries=retainDOFTimeSeries
        """
        _Boolean_, denotes whether DOF time series data should be 
        retained once responses have been computed. Note: if _False_ then full 
        re-analysis will be required to re-compute responses.
        """

        self.retainResponseTimeSeries=retainResponseTimeSeries
        """
        _Boolean_, denotes whether response time series data should be 
        retained once statistics have been computed
        """
        
        
    def ClearResults(self):
        """
        Clear all results from previous time-stepping analysis
        """
        
        self.t = None
        self.f = None
        self.v = None
        self.vdot = None
        self.v2dot = None
        self.f_constraint = None
        self.nResults = 0
        
        
    def RecordResults(self,t,f,v,vdot,v2dot,f_constraint):
        """
        Used to record results from time-stepping analysis as implemented 
        by `TStep().run()`
        """
        
        if self.nResults==0:
            self.nDOF=v.shape[0]
            self.t=npy.asmatrix(t)
            self.f=f.T
            self.v=v.T
            self.vdot=vdot.T
            self.v2dot=v2dot.T
            self.f_constraint=f_constraint.T
            
        else:
            self.t = npy.asmatrix(npy.append(self.t,npy.asmatrix(t),axis=0))
            self.f = npy.asmatrix(npy.append(self.f,f.T,axis=0))
            self.v = npy.asmatrix(npy.append(self.v,v.T,axis=0))
            self.vdot = npy.asmatrix(npy.append(self.vdot,vdot.T,axis=0))
            self.v2dot = npy.asmatrix(npy.append(self.v2dot,v2dot.T,axis=0))
            self.f_constraint = npy.asmatrix(npy.append(self.f_constraint,f_constraint.T,axis=0))
                        
        self.nResults+=1
        
        
    def GetResults(self,dynsys_obj,attr_list:list):
        """
        Retrieves results for a given DynSys object, which is assumed to be a 
        subsystem of the parent system (this is checked)
        
        ***
        Required:
            
        * `dynsys_obj`, instance of DynSys class; must be a subsystem of the 
          parent system
          
        * `attr_list`, list of strings, to denote the results attributes to be 
          returned
         
        """
        
        # Convert to list if single argument provided
        if not isinstance(attr_list,list):
            attr_list = list(attr_list)
        
        # Check that dynsys_obj is part of the system that has been analysed
        parent_sys = self.tstep_obj.dynsys_obj
        DynSys_list = parent_sys.DynSys_list
        
        if not dynsys_obj in DynSys_list:
            raise ValueError("`dynsys_obj` is not a subsystem " + 
                             "of the parent system that has been analysed!")
        
        # Determine splice indices to use to get results relating to requested system 
        sys_index = [i for i, j in enumerate(DynSys_list) if j == dynsys_obj][0]
        
        nDOF_list_cum = npy.cumsum([x.nDOF for x in DynSys_list]).tolist()
        
        if sys_index == 0:
            startIndex = 0
        else:
            startIndex = nDOF_list_cum[sys_index-1]
        
        endIndex = nDOF_list_cum[sys_index] 
        
        # Loop through to return requested attributes
        vals_list = []
        
        for attr in attr_list:
            
            if hasattr(self,attr):
                
                vals = getattr(self,attr)             # for full system
                vals = vals[:,startIndex:endIndex]    # for subsystem requested
                vals_list.append(vals)
                
            else:
                raise ValueError("Could not retrieve requested attribute '%s'" % attr)
        
        return vals_list
        
    
    def _TimePlot(self,ax,t_vals,data_vals,
                  titleStr=None,
                  xlabelStr=None,
                  ylabelStr=None,
                  xlim=None,
                  showxticks:bool=True,
                  data_labels=None):
        """
        Produces a time series results plot
        """

        lines = ax.plot(t_vals,data_vals)
        
        if titleStr is not None:
            ax.set_title(titleStr)
            
        if xlabelStr is not None:
            ax.set_xlabel(xlabelStr)
            
        if ylabelStr is not None:
            ax.set_ylabel(ylabelStr)
            
        if xlim is not None:
            ax.set_xlim(xlim)
            
        if not showxticks:
            ax.set_xticks([])
            
        return lines
        
    
    def PlotStateResults(self,
                         dynsys_obj=None,
                         verbose:bool=True,
                         dofs2Plot=None):
        """
        Produces time series plots of the following, as subplots in a single 
        figure:
            
        * Applied external forces
        
        * Displacements
        
        * Velocities
        
        * Accelerations
        
        * Constraint forces (only if constraint equations defined)
        
        Note all results plotted relate to the _analysis freedoms_.
        
        For systems composed of multiple subsystems, one figure will be 
        produced for each subsystem.
        
        ***
        **Required:**
        
        (No required arguments; results held as class attributes will be 
        plotted)
        
        ***
        **Optional:**
            
        * `dynsys_obj`, can be used to specify the subsystem for which results 
          which should be plotted. If _None_ then results for all freedoms 
          (i.e. all subsystems) will be plotted.
          
        * `dofs2plot`, _list_ or _array_ of indexs of freedoms for which 
          results should be plotted. If _None_ then results for all freedoms 
          will be plotted.
          
        * `verbose`, _boolean_, if True text will be written to console
          
        ***
        **Returns:**
        
        _List_ of figure objects
        
        """
        
        if verbose:
            print("Preparing state results plot...")
            tic=timeit.default_timer()
            
        # Get subsystems to iterate over
        DynSys_list = self.tstep_obj.dynsys_obj.DynSys_list
        if dynsys_obj is not None:
            
            if not (dynsys_obj in DynSys_list):
                raise ValueError("`dynsys_obj` not in DynSys_list!")
                
            DynSys_list = [dynsys_obj]
        
        # Iterate over all subsystems, producing figure for each
        fig_list = []
        
        for obj in DynSys_list:

            # Determine number of subplots required
            hasConstraints = obj.hasConstraints()
            if hasConstraints:
                nPltRows=5
            else:
                nPltRows=4
                
            # Create plot of results for analysis DOFs
            fig, axarr = plt.subplots(nPltRows, sharex=True)
            fig.set_size_inches((14,8))
            fig_list.append(fig)
        
            # Set xlim
            xlim = [self.tstep_obj.tStart,self.tstep_obj.tEnd]
            
            # Get system description string
            sysStr = ""
            DynSys_list = obj.DynSys_list
            
            if all([x.isModal for x in DynSys_list]):
                # All systems and subsystems are modal
                sysStr = "Modal "
                
            # Get data to plot            
            t,f,v,vdot,v2dot,f_constraint =self.GetResults(obj,
                                                           ['t','f','v',
                                                            'vdot','v2dot',
                                                            'f_constraint'])
            
            # Handle dofs2Plot in case of none
            if dofs2Plot is None:
                dofs2Plot = range(v.shape[1])
            
            # Create time series plots
            self._TimePlot(ax=axarr[0],
                           t_vals=t,
                           data_vals=f[:,dofs2Plot],
                           xlim=xlim,
                           titleStr="Applied %sForces (N)" % sysStr)
            
            self._TimePlot(ax=axarr[1],
                           t_vals=t,
                           data_vals=v[:,dofs2Plot],
                           titleStr="%sDisplacements (m)" % sysStr)
            
            self._TimePlot(ax=axarr[2],
                           t_vals=t,
                           data_vals=vdot[:,dofs2Plot],
                           titleStr="%sVelocities (m/s)" % sysStr)
            
            self._TimePlot(ax=axarr[3],
                           t_vals=t,
                           data_vals=v2dot[:,dofs2Plot],
                           titleStr="%sAccelerations ($m/s^2$)" % sysStr)
            
            if hasConstraints:
                self._TimePlot(ax=axarr[4],
                           t_vals=t,
                           data_vals=f_constraint,
                           titleStr="%sConstraint forces (N)" % sysStr)
                
            axarr[-1].set_xlabel("Time (secs)")
            fig.subplots_adjust(hspace=0.3)
            
            # Overall title for plot
            fig.suptitle("State variable results for '%s'" % obj.name)
        
        if verbose:
            toc=timeit.default_timer()
            print("Plot prepared after %.3f seconds." % (toc-tic))
            
        return fig_list
    
    
    def PlotResponseResults(self,
                            dynsys_obj=None,
                            responses2plot=None,
                            y_overlay:list=None,
                            y_overlay_color:str='darkorange',
                            verbose=True,
                            raiseErrors=True,
                            useCommonPlot:bool=False,
                            useCommonScale:bool=True):
        """
        Produces a new figure with linked time series subplots for 
        all responses/outputs
        
        ***
        **Required:**
        
        (No required arguments; results held as class attributes will be 
        plotted)
        
        ***
        **Optional:**
            
        * `dynsys_obj`, can be used to specify the subsystem for which results 
          which should be plotted. If _None_ then results for all freedoms 
          (i.e. all subsystems) will be plotted.
          
        * `responses2plot`, can be used to specify the indexs of responses to 
          plot, with the specified `dynsys_obj`.
          
        * `y_overlay`, _list_ of floats, can be used to overlay a horizontal 
          line onto plots e.g. to denote some criteria or limit to be satisfied
          
        * `verbose`, _boolean_, if True then output will be written to console
          
        * `raiseErrors`, _boolean, dictates whether `ValueError()` should be 
          raised in case of no results to plot.
        
        * `useCommonPlot`, _boolean_, if True all responses will be overlaid 
          onto a single axis. Otherwise multiple subplots will created, one for 
          each response
          
        * `useCommonScale`, _boolean_, if multiple subplots are created, 
          dictates whether a common vertical scale should be used
          
        ***
        **Returns:**
         
        _List_ of figure objects
          
        """
        
        if verbose:
            print("Preparing response results plots...")
            tic=timeit.default_timer()
            
        # Get subsystems to iterate over
        DynSys_list = self.tstep_obj.dynsys_obj.DynSys_list
        if dynsys_obj is not None:
            
            if not (dynsys_obj in DynSys_list):
                raise ValueError("`dynsys_obj` not in DynSys_list!")
                
            DynSys_list = [dynsys_obj]
    
        # Iterate over all responses for all specified subsystems
        
        responses_list = self.responses_list
        response_names_list = self.response_names_list
        
        fig_list = []
        
        for dynsys_obj, _responses, _response_names in zip(DynSys_list,
                                                           responses_list,
                                                           response_names_list):
            
            print("Plotting responses for '{0}'...".format(dynsys_obj.name))
            
            # Get responses to plot
            if responses2plot is not None:
                _responses = _responses[[r for r in responses2plot]]
                _response_names = [_response_names[r] for r in responses2plot]
        
            # Determine total number of responses to plot
            nResponses = _responses.shape[0]
            if verbose: print("# responses to plot: {0}".format(nResponses))
            
            if nResponses == 0:
                errorstr = "nResponses=0, nothing to plot!"
                if raiseErrors:
                    raise ValueError(errorstr)
                else:
                    print(errorstr)
                    return None
                    
            # Initialise figure 
            if not useCommonPlot:
                fig, axarr = plt.subplots(nResponses, sharex=True)
            else:
                fig, axarr = plt.subplots(1)
                
            fig.set_size_inches((14,8))
            
            fig.suptitle("Responses for '{0}'".format(dynsys_obj.name))

            fig_list.append(fig)
            
            # Determine common scale to use for plots
            if useCommonScale:    
                
                maxVal = npy.max(_responses)
                minVal = npy.min(_responses)
                absmaxVal = npy.max([maxVal,minVal])
                
                if y_overlay is not None:
                    
                    y_overlay_max = npy.max(npy.array(y_overlay))
                    absmaxVal = npy.max([absmaxVal,1.1*y_overlay_max])
            
            # Loop through plotting all responses
            tvals = self.t
            tInterval= [self.tstep_obj.tStart,self.tstep_obj.tEnd]
            
            for r in range(nResponses):
                
                # Get axis object
                if useCommonPlot:
                    ax = axarr
                else:
                    if hasattr(axarr, "__len__"):
                        ax = axarr[r]
                    else:
                        ax = axarr
                
                # Get values to plot
                vals = _responses[r,:].T
                label_str = _response_names[r]
                
                # Make time series plot
                ax.plot(tvals,vals,label=label_str)
                
                # Plot horizontal lines to overlay values provided
                # Only plot once in case of common plot
                if (useCommonPlot and r==0) or (not useCommonPlot):
    
                    if y_overlay is not None:
                        
                        # Convert to list in case of single float passed
                        if not isinstance(y_overlay,list):
                            y_overlay = [y_overlay]
                            
                        for y_val  in y_overlay:
                            ax.axhline(y_val,color=y_overlay_color)
                           
                # Set axis limits and labels
                ax.set_xlim(tInterval)
                
                if r == nResponses-1:
                    ax.set_xlabel("Time [s]")
                
                if useCommonScale and not useCommonPlot:
                    ax.set_ylim([-absmaxVal,+absmaxVal])
                
                # Create legend
                if _response_names is not None:
                    ax.legend(loc='right')
                
                
            
        if verbose:
            toc=timeit.default_timer()
            print("Plots prepared after %.3f seconds." % (toc-tic))
            
        return fig_list
        
    
    def PlotResults(self,
                    dynsys_obj=None,
                    verbose:bool=True,
                    dofs2Plot:list=None,
                    useCommonPlot:bool=False,
                    useCommonScale:bool=False,
                    y_overlay:float=None):
        """
        Presents the results of time-stepping analysis by producing the 
        following plots:
        
        * State results plot: refer `PlotStateResults()`
        
        * Response results plot: refer `PlotResponseResults()`
        
        ***
        **Required:**
        
        (No required arguments; results held as class attributes will be 
        plotted)
        
        ***
        **Optional:**
            
        * `dynsys_obj`, object specifying system whose results are to be 
          plotted. If _None_ plots will be produced for all systems. 
        
        * `verbose`, _boolean_, if True progress will be written to console
        
        * `dof2Plot`, _list_ of integers to denote DOFs to plot. Can be used to 
          produce a 'cleaner' plot, in case of large numbers of DOFs.
          
        * `useCommonPlot`, `useCommonScale`: _boolean_, 
          refer `PlotResponseResults()` docs for details
        
        ***
        **Returns:**
        
        _List_ of figure objects
        
        """
        
        fig_list1 = self.PlotStateResults(dynsys_obj=dynsys_obj,
                                          verbose=verbose,
                                          dofs2Plot=dofs2Plot)
        
        fig_list2 = self.PlotResponseResults(dynsys_obj=dynsys_obj,
                                             raiseErrors=False,
                                             verbose=verbose,
                                             useCommonPlot=useCommonPlot,
                                             useCommonScale=useCommonScale,
                                             y_overlay=y_overlay)
        
        return [fig_list1,fig_list2]
    
    
    def PlotResponsePSDs(self,nperseg=None):
        """
        Produces power spectral density plots of response time series
        
        ***
        Optional:
        
        * `nperseg`, length of each segment used in PSD calculation
        
        """
        
        print("Plotting PSD estimates of responses using periodograms...")
        
        if len(self.responses_list) == 0:
            raise ValueError("No response time series data avaliable!")
        
        # Get sampling frequency
        if self.tstep_obj.dt is None:
            raise ValueError("`dt` is None: cannot calculate PSDs from "
                             "variable timestep time series")
        else:
            fs = 1 / self.tstep_obj.dt
            print("Sampling frequency: fs = %.2f" % fs)
        
        # Loop through all responses
        fig_list = []
        
        for dynsys_obj, responses, response_names in zip(self.tstep_obj.dynsys_obj.DynSys_list,
                                                         self.responses_list,
                                                         self.response_names_list):
        
            # Create figure
            fig, axarr = plt.subplots(2,)
            fig.set_size_inches((14,8))
            
            # Time series plot
            ax = axarr[0]
            handles = ax.plot(self.t,responses.T)
            maxVal = npy.max(npy.abs(responses.T))
            ax.set_ylim([-maxVal,maxVal])
            ax.set_xlim(0,npy.max(self.t))
            ax.set_title("Response time series")
            ax.set_xlabel("Time (secs)")
            
            fig.legend(handles,response_names,loc='upper right')
            
            # PSD plot
            f, Pxx = scipy.signal.periodogram(responses,fs)
            ax = axarr[1]
            ax.plot(f,Pxx.T)
            ax.set_xlim([0,fs/2])
            ax.set_title("Periodograms of responses")
            ax.set_xlabel("Frequency (Hz)")
            
            fig.suptitle("System: '{0}'".format(dynsys_obj.name))
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)        # create space for suptitle
            fig.subplots_adjust(right=0.8)      # create space for figlegend
            
            fig_list.append(fig)
        
        return fig_list
    
        
    def AnimateResults(self):
        
        ValueError("Unfinished - do not use!")
        
        n = self.nDOF
        
        vmax = npy.max(npy.max(self.v))
        vmin = npy.min(npy.min(self.v))
        vabsmax = npy.max([vmax,vmin])
        Ns=self.t.shape[0]
        dt=self.dt
        
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, n-1), ylim=(-vabsmax,vabsmax))
        ax.grid()
        
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        
        def animate(i):
            #print("frame{0}".format(i))
            x = [npy.arange(n)]
            #print(x)
            y = [self.v[i,:]]
            #print(y)
            t_val = self.t[i,0]
            #print(t_val)
            
            line.set_data(x, y)
            time_text.set_text(time_template % t_val)
            
            return line, time_text
        
        ani = animation.FuncAnimation(fig, animate, npy.arange(0, Ns), interval=dt*1000,blit=False, init_func=init, repeat=False)
        
        plt.show()
        
        
    def CalcResponses(self,
                      write_results_to_file=False,
                      results_fName="ts_results.csv",
                      verbose=True):
        """
        Responses are obtained by pre-multiplying results by output matrices
        
        Output matrices, together with their names, must be pre-defined via 
        `DynSys` member function `AddOutputMtrx()` prior to running this 
        function.
        
        Where a system consists of multiple subsystems, it should be note that 
        output matrices relate to a given subsystem.
        """
        
        dynsys_obj=self.tstep_obj.dynsys_obj
        responses_list = []
        response_names_list = []
        
        # Calculate responses for all systems and subsystems
        for x in dynsys_obj.DynSys_list:
            
            # Get output matrix for subsystem
            output_mtrx = x.output_mtrx
            output_names = x.output_names
            
            # Retrieve state variables for subsystem
            v, vdot, v2dot = self.GetResults(x,['v', 'vdot', 'v2dot'])
            
            # Obtain new responses
            state_vector = npy.hstack((v,vdot,v2dot))
            
            responses_list.append(output_mtrx * state_vector.T)
            response_names_list.append(output_names)
            
        # Store as attributes
        self.responses_list = responses_list
        self.response_names_list = response_names_list
                
        # Calculate DOF statistics
        if self.calc_dof_stats:
            self.CalcDOFStats(verbose=verbose)
            
        # Calculate response statistics
        if self.calc_response_stats:
            self.CalcResponseStats(verbose=verbose)
            
        # Write time series results to file
        if write_results_to_file:
            self.WriteResults2File(output_fName=results_fName)
                
        # Delete DOF time series data (to free-up memory)
        if not self.retainDOFTimeSeries:
            if verbose: print("Clearing DOF time series data to save memory...")
            del self.v
            del self.vdot
            del self.v2dot
        
        # Delete response time series data (to free up memory)
        if not self.retainResponseTimeSeries:
            if verbose: print("Clearing response time series data to save memory...")
            del self.responses_list
            del self.response_names_list
        
        
        
    def CalcDOFStats(self,verbose=True):
        """
        Obtain basic statistics to describe DOF time series
        
        ***
        **Required:**
        
        No arguments are required
        
        ***
        **Optional:**
        
        * `verbose`, _boolean_, controls the amount of console output
        
        ***
        **Returns:**
        
        Stats are returned as a _list_ of _dicts_, with list indexs as follows:
            
        * 0 : DOF displacements
        
        * 1 : DOF velocities
        
        * 2 : DOF accelerations
        
        * 3 : Constraint forces (only if constraints defined)
        
        Dict entries will in general be _arrays_, giving stats results for each 
        degree of freedom
        
        """
        
        if self.calc_dof_stats:
            
            if verbose: print("Calculating DOF statistics...")
            
            stats=[]
            
            for _timeseries in [self.v,self.vdot,self.v2dot,self.f_constraint]:
                
                # Calculate stats for each response time series
                maxVals = npy.asarray(npy.max(_timeseries,axis=0).T)
                minVals = npy.asarray(npy.min(_timeseries,axis=0).T)
                stdVals = npy.asarray(npy.std(_timeseries,axis=0).T)
                absmaxVals = npy.asarray(npy.max(npy.abs(_timeseries),axis=0).T)
            
                # Record stats within a dict
                d={}
                d["max"]=maxVals
                d["min"]=minVals
                d["std"]=stdVals
                d["absmax"]=absmaxVals
                stats.append(d)
            
            # Store within object
            self.dof_stats=stats
            
        else:
            if verbose:
                print("calcDOFStats=False option set. " +
                      "DOF statistics will not be computed.")
        
        return stats
        
    
    def CalcResponseStats(self,verbose=True):
        """
        Obtain basic statistics to describe response time series
        """
        
        if self.calc_response_stats:
            
            # Get paired lists to loop over
            responses_list = self.responses_list
            dynsys_list = self.tstep_obj.dynsys_obj.DynSys_list
            
            # Loop over all systems and subsystems
            for dynsys_obj, responses in zip(dynsys_list,responses_list):
                
                if verbose:
                    print("Calculating response statistics " + 
                          "for '{0}'...".format(dynsys_obj.name))
                        
                # Calculate stats for each response time series
                maxVals = npy.ravel(npy.max(responses,axis=1))
                minVals = npy.ravel(npy.min(responses,axis=1))
                stdVals = npy.ravel(npy.std(responses,axis=1))
                absmaxVals = npy.ravel(npy.max(npy.abs(responses),axis=1))
            
                # Record stats within a dict
                stats_dict={}
                stats_dict["max"]=maxVals
                stats_dict["min"]=minVals
                stats_dict["std"]=stdVals
                stats_dict["absmax"]=absmaxVals
                
                # Store as new dict entry
                self.response_stats_dict[dynsys_obj] = stats_dict
            
        else:
            if verbose:
                print("calcResponseStats=False option set." + 
                      "Response statistics will not be computed.")
        
        return stats_dict
    
    
    def get_response_stats_df(self):
        """
        Returns response stats as list of Pandas dataframes
        """
        
        df_list = []
        
        for dynsys_obj, stats_dict in self.response_stats_dict.items():
            
            response_names = dynsys_obj.output_names

            df_list.append(pandas.DataFrame(data=stats_dict,
                                            index=response_names))
            
        return df_list
    
    
    def PrintResponseStats(self):
        """
        Prints response stats to text window
        """
        
        for dynsys_obj, stats_dict in self.response_stats_dict.items():
            
            print("System name: {0}".format(dynsys_obj.name))
            print("Response names: {0}".format(dynsys_obj.output_names))
            print("Response statistics :")
            
            for stats_str, stats_vals in stats_dict.items():
                
                print("'%s':" % stats_str)
                print(stats_vals)
            
            print("")
    
    
    def WriteResults2File(self,output_fName="timeseries_results.csv"):
        """
        Writes time-series results to file
        """
        
        header=""
        header+="Time series results export\n"
        header+="dynsys_obj:, {0}\n".format(self.tstep_obj.dynsys_obj.description)
        header+="nDOF:, {0}\n".format(self.nDOF)
        header+="nResults:, {0}\n".format(self.nResults)
        header+="DynSys version:, {0}\n".format(currentVersion)
        header+="Export date/time:, {0}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        header+="\n"
        
        array2write=None
        
        def writeArray(arrName):
            """
            Function to append array and create suitable headers
            """
            
            arr = getattr(self,arrName)

            nonlocal header, array2write
            
            if arr.shape[1]>1:
                for i in range(arr.shape[1]):
                    header += "%s%d," % (arrName,i+1)
            else:
                header += "%s," % arrName
                
            if array2write is None:
                array2write = arr
            else:
                array2write = npy.append(array2write,arr,axis=1)
            
        # Use the above function to prepare data to write to file
        writeArray('t')
        writeArray('f')
        writeArray('v')
        writeArray('vdot')
        writeArray('v2dot')
        writeArray('f_constraint')
        
        # Write to file
        npy.savetxt(output_fName,
                    X=array2write,
                    delimiter=',',
                    header=header)
        
        print("Time-series results written to `{0}`".format(output_fName))
