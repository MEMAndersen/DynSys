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
        
        self.responses=[]
        """
        List of matrices to which computed responses are recorded 
        for each time step
        """
        
        self.responseNames=[]
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
        
        self.response_stats={}
        """
        Dict of statistics, evaluated over all time steps, for each response
        
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
                         printProgress:bool=True,
                         dofs2Plot=None):
        """
        Produces time series plots of the following, all on one figure:
            
        * Applied external forces
        
        * Displacements
        
        * Velocities
        
        * Accelerations
        
        * Constraint forces (only if constraint equations defined)
        
        """
        
        if printProgress:
            print("Preparing state results plot...")
            tic=timeit.default_timer()
        
        # Determine number of subplots required
        constraints = self.tstep_obj.dynsys_obj.hasConstraints()
        if constraints:
            nPltRows=5
        else:
            nPltRows=4
            
        # Create plot of results for analysis DOFs
        fig, axarr = plt.subplots(nPltRows, sharex=True)
        fig.set_size_inches((18,9))
    
        # Handle dofs2Plot in case of none
        if dofs2Plot is None:
            dofs2Plot = range(self.v.shape[1])
        
        # Set xlim
        xlim = [self.tstep_obj.tStart,self.tstep_obj.tEnd]
        
        # Get system description string
        sysStr = ""
        if self.tstep_obj.dynsys_obj.__class__.__name__ == "ModalSys":
            sysStr = "Modal "
        
        # Create time series plots
        self._TimePlot(ax=axarr[0],
                       t_vals=self.t,
                       data_vals=self.f[:,dofs2Plot],
                       xlim=xlim,
                       titleStr="Applied {0} Forces (N)".format(sysStr))
        
        self._TimePlot(ax=axarr[1],
                       t_vals=self.t,
                       data_vals=self.v[:,dofs2Plot],
                       titleStr="{0}Displacements (m)".format(sysStr))
        
        self._TimePlot(ax=axarr[2],
                       t_vals=self.t,
                       data_vals=self.vdot[:,dofs2Plot],
                       titleStr="{0}Velocities (m/s)".format(sysStr))
        
        self._TimePlot(ax=axarr[3],
                       t_vals=self.t,
                       data_vals=self.v2dot[:,dofs2Plot],
                       titleStr="{0}Accelerations ($m/s^2$)".format(sysStr))
        
        if constraints:
            self._TimePlot(ax=axarr[4],
                           t_vals=self.t,
                           data_vals=self.f_constraint,
                           titleStr="{0}Constraint forces (N)".format(sysStr))
            
        axarr[-1].set_xlabel("Time (secs)")
        fig.subplots_adjust(hspace=0.3)
        
        if printProgress:
            toc=timeit.default_timer()
            print("Plot prepared after %.3f seconds." % (toc-tic))
            
        return fig
    
    def PlotResponseResults(self,
                            y_overlay:list=None,
                            useCommonPlot:bool=False,
                            useCommonScale:bool=True,
                            printProgress:bool=True):
        """
        Produces a new figure with linked time series subplots for 
        all responses/outputs
        """
        
        if printProgress:
            print("Preparing results plot...")
            tic=timeit.default_timer()
    
        # Determine total number of responses to plot
        nResponses=0
        for _responses in self.responses:
            nResponses+=_responses.shape[1]
            
        print("nResponses to plot: {0}".format(nResponses))
        
        if nResponses == 0:
            raise ValueError("nResponses=0, nothing to plot!")
    
        # Initialise figure 
        if not useCommonPlot:
            fig, axarr = plt.subplots(nResponses, sharex=True)
        else:
            fig, axarr = plt.subplots(1)
            
        fig.set_size_inches((14,8))
        
        # Determine common scale to use for plots
        if useCommonScale:    
            maxVal = npy.max(self.responses)
            minVal = npy.min(self.responses)
            absmaxVal = npy.max([maxVal,minVal])
        
        # Loop through all matrices in response list
        tvals = self.t
        
        for i, _responses in enumerate(self.responses):
            
            _responseNames = self.responseNames[i]
            
            for r in range(_responses.shape[1]):
                
                if useCommonPlot:
                    ax = axarr
                else:
                    ax = axarr[r]
                
                vals = _responses[:,r]
                print(vals.shape)
                label_str = _responseNames[r]
                print(label_str)
                
                ax.plot(tvals,vals,label=label_str)
                                
                ax.set_xlim([self.tstep_obj.tStart,self.tstep_obj.tEnd])
                ax.set_xlabel("Time [s]")
                
                if useCommonScale and not useCommonPlot:
                    ax.set_ylim([-absmaxVal,+absmaxVal])
                
                if self.responseNames is not None:
                    ax.legend(loc='best')
                
                # Plot horizontal lines to overlay values provided
                # Only plot once in case of common plot
                if (useCommonPlot and r==0) or (not useCommonPlot):
    
                    if y_overlay is not None:
                        for y_val  in y_overlay:
                            ax.axhline(y_val,color='r')
        
        if printProgress:
            toc=timeit.default_timer()
            print("Plot prepared after %.3f seconds." % (toc-tic))
        
        return fig
        
    def PlotResults(self,
                    printProgress:bool=True,
                    dofs2Plot:bool=None,
                    useCommonPlot:bool=False):
        """
        Produces the following standard plots to document the results of 
        time-stepping analysis:
            
        * State results plot: refer `PlotStateResults()`
        
        * Response results plot: refer `PlotResponseResults()`
        
        """
        
        if dofs2Plot is None:
            # Plot results for all DOFs
            dofs2Plot=range(self.nDOF)
            
        figs=[]
        
        figs.append(self.PlotStateResults(printProgress=printProgress,
                                          dofs2Plot=dofs2Plot))
        
        figs.append(self.PlotResponseResults(printProgress=printProgress,
                                             raiseExceptions=False,
                                             useCommonPlot=useCommonPlot))
        
        return figs
    
    def PlotResponsePSDs(self,nperseg=None):
        """
        Produces power spectral density plots of response time series
        
        ***
        Optional:
        
        * `nperseg`, length of each segment used in PSD calculation
        
        """
        
        print("Plotting PSD estimates of responses using Welch's method...")
        
        if self.responses is None:
            raise ValueError("No response time series data avaliable!")
        else:
            responses = self.responses
            
        # Get sampling frequency
        if self.tstep_obj.dt is None:
            raise ValueError("`dt` is None: cannot calculate PSDs from "
                             "variable timestep time series")
        else:
            fs = 1 / self.tstep_obj.dt
            print("Sampling frequency: fs = %.2f" % fs)
        
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
        
        fig.legend(handles,self.responseNames,loc='upper right')
        
        # PSD plot
        f, Pxx = scipy.signal.welch(responses,fs=fs,nperseg=nperseg)
        ax = axarr[1]
        ax.plot(f,Pxx.T)
        ax.set_xlim([0,fs/2])
        ax.set_title("PSDs of responses")
        ax.set_xlabel("Frequency (Hz)")
        
        if self.tstep_obj.name is not None:
            fig.suptitle("{0}".format(self.tstep_obj.name))
            fig.subplots_adjust(top=0.9)
        
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)      # create space for figlegend
            
    def AnimateResults(self):
        
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
                      results_fName="ts_results.csv"):
        """
        Responses are obtained by pre-multiplying results by output matrices
        
        Output matrices, together with their names, must be pre-defined via 
        `DynSys` member function `AddOutputMtrx()` prior to running this 
        function.
        """
        
        dynsys_obj=self.tstep_obj.dynsys_obj
        
        output_mtrx = dynsys_obj.output_mtrx
        output_names = dynsys_obj.output_names
            
        # Obtain new responses
        state_vector = npy.hstack((self.v,self.vdot,self.v2dot))
        self.responses = output_mtrx * state_vector.T
        self.responseNames = output_names
            
        # Calculate DOF statistics
        if self.calc_dof_stats:
            self.CalcDOFStats()
            
        # Calculate response statistics
        if self.calc_response_stats:
            self.CalcResponseStats()
            
        # Write time series results to file
        if write_results_to_file:
            self.WriteResults2File(output_fName=results_fName)
            
        # Delete DOF time series data (to free-up memory)
        if not self.retainDOFTimeSeries:
            print("Clearing DOF time series data to save memory...")
            del self.v
            del self.vdot
            del self.v2dot
        
        # Delete response time series data (to free up memory)
        if not self.retainResponseTimeSeries:
            print("Clearing response time series data to save memory...")
            del self.responses
            del self.responseNames
        
    def CalcDOFStats(self):
        """
        Obtain basic statistics to describe DOF time series
        """
        
        if self.calc_dof_stats:
            
            print("Calculating DOF statistics...")
            
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
            print("calcResponseStats=False option set. Response statistics "+
                  "will not be computed.")
        
        return stats
        
    def CalcResponseStats(self):
        """
        Obtain basic statistics to describe response time series
        """
        
        if self.calc_response_stats:
            
            print("Calculating response statistics...")
            
            stats=[]
            
            # Calculate stats for each response time series
            maxVals = npy.ravel(npy.max(self.responses,axis=1))
            minVals = npy.ravel(npy.min(self.responses,axis=1))
            stdVals = npy.ravel(npy.std(self.responses,axis=1))
            absmaxVals = npy.ravel(npy.max(npy.abs(self.responses),axis=1))
        
            # Record stats within a dict
            d={}
            d["max"]=maxVals
            d["min"]=minVals
            d["std"]=stdVals
            d["absmax"]=absmaxVals
            stats.append(d)
            
            # Store within object
            self.response_stats=stats
            
        else:
            print("calcResponseStats=False option set. Response statistics "+
                  "will not be computed.")
        
        return stats
    
    
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