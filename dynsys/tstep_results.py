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
from matplotlib.animation import FuncAnimation
import timeit
import pandas
from datetime import datetime
from collections import OrderedDict

from common import check_is_class, deprecation_warning



class TStep_Results:
    """
    Class used to store, manipulate and plot results from time-stepping 
    analysis as carried out by `TStep`
    """
    # ********** CONSTRUCTOR / DESTRUCTOR ******************
    
    def __init__(self,tstep_obj,
                 options={'calc_dof_stats' : True,
                          'calc_response_stats' : True,
                          'retain_dof_values' : True,
                          'retain_response_values' : True}
                 ):
        """
        Initialisation function
        ***
        
        Required:
        
        * `tstep_obj`, `TStep()` class instance to which results relate
        
        ***
        Optional
        
        * `options` _dict_, with the following as boolean entries:
        
            * `calc_DOF_stats`, if True statistics will be calculated, to 
              summarise time series results for DOF variables
            
            * `calc_response_stats`, if True statistics will be calculated, to 
              summarise time series results for response variables
              
            * `retain_DOF_values`, if False DOF time series results will be 
              deleted once summary statistics have been computed
                         
            * `retain_response_values`, if False response time series results 
              will be deleted once summary statistics have been computed
          
            Choosing _False_ for the above can be done to be more economical 
            with memory usage, for example when carrying out multiple analyses. 
            Note that providing DOF time series are retained, responses can 
            always be re-calculated using `calc_responses()` method
        """
                
        self.nResults=0
        
        self.tstep_obj=tstep_obj
        """
        Instance of `tstep` class, to which results relate
        """
        
        self.DynSys_list = tstep_obj.dynsys_obj.DynSys_list
        """
        List of DynSys objects to which results held by this object relate
        """
        
        self.response_results=[]
        """
        List of matrices to which computed responses are recorded 
        for each time step
        """
                
        self.options = options
        """
        Dict containing options - refer __init__() docstring for details
        """
        
        self.dof_stats=None
        """
        Dict of statistics, evaluated over all time steps, for each DOF
        
        _Set using `CalcDOFStats()`_
        """
        
    # -----------------GETTER / SETTER FUNCTIONS FOR PROPERTIES ---------------    
                
    @property
    def nDOF(self):
        """
        Number of degrees of freedom for system being analysed
        """
        return self._nDOF
    
    @nDOF.setter
    def nDOF(self,value):
        self._nDOF = value
        
    # ---------------
    @property
    def t(self):
        """
        Time values for each results time step
        """
        return self._t
        
    @t.setter
    def t(self,values):
        self._t = values
        
    # ---------------
    @property
    def f(self):
        """
        External applied forces
        
        Returns matrix of shape [nSteps,nDOF]
        """
        
        obj = self._force_results
        
        if obj is None:
            return None
        else:
            return obj.values
    
    @f.setter
    def f(self,values):
        
        if values is None:
            self._force_results = None
            
        else:
            nDOF = values.shape[1]
            
            obj = TimeSeries_Results(names = ["f%d" for n in range(nDOF)],
                                     values = values,
                                     tstep_results_obj = self,
                                     plot_options={'use_common_plot' :True})
            
            self._force_results = obj
        
    # ---------------
    @property
    def v(self):
        """
        Displacements of analysis DOFs
        
        Returns matrix of shape [nSteps,nDOF]
        """
        obj = self._disp_results
        
        if obj is None:
            return None
        else:
            return obj.values
    
    @v.setter
    def v(self,values):
        
        if values is None:
            self._disp_results = None
            
        else:
            nDOF = values.shape[1]
            
            obj = TimeSeries_Results(names = ["$y_{%d}$" for n in range(nDOF)],
                                     values = values,
                                     tstep_results_obj = self,
                                     plot_options={'use_common_plot' :True})
            
            self._disp_results = obj
        
    # ---------------
    @property
    def vdot(self):
        """
        Velocities of analysis DOFs
        """
        obj = self._velocity_results
        
        if obj is None:
            return None
        else:
            return obj.values
    
    @vdot.setter
    def vdot(self,values):
        
        if values is None:
            self._velocity_results = None
            
        else:
            nDOF = values.shape[1]
            
            obj = TimeSeries_Results(names = ["$\dot{v}_{%d}$"
                                              for n in range(nDOF)],
                                     values = values,
                                     tstep_results_obj = self,
                                     plot_options={'use_common_plot' :True})
            
            self._velocity_results = obj
        
    # ---------------
    @property
    def v2dot(self):
        """
        Accelerations of analysis DOFs
        """
        obj = self._accn_results
        
        if obj is None:
            return None
        else:
            return obj.values
    
    @v2dot.setter
    def v2dot(self,values):
        
        if values is None:
            self._accn_results = None
            
        else:
            nDOF = values.shape[1]
            
            obj = TimeSeries_Results(names = ["$\ddot{v}_{%d}$"
                                              for n in range(nDOF)],
                                     values = values,
                                     tstep_results_obj = self,
                                     plot_options={'use_common_plot' :True})
            
            self._accn_results = obj
        
    # ---------------
    @property
    def f_constraint(self):
        """
        Internal constraint forces
        """
        obj = self._constraint_forces
        
        if obj is None:
            return None
        else:
            return obj.values
    
    @f_constraint.setter
    def f_constraint(self,values):
        
        if values is None:
            self._constraint_forces = None
            
        else:
            nDOF = values.shape[1]
            
            obj = TimeSeries_Results(names = ["$f_{constraint,%d}$"
                                              for n in range(nDOF)],
                                     values = values,
                                     tstep_results_obj = self,
                                     plot_options={'use_common_plot' :True})
            
            self._constraint_forces = obj
    
    # ---------------     
    @property
    def nResults(self):
        """
        Number of time-steps at which results are recorded
        """
        return self._nResults
    
    @nResults.setter
    def nResults(self,value):
        self._nResults = value
     
    # -------------------------------------------------------------------------
    
    
    
    
        
        
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
        
        
    def RecordResults(self,t,f,v,vdot,v2dot,f_constraint,
                      verbose=True,append=True):
        """
        Used to record results from time-stepping analysis as implemented 
        by `TStep().run()`
        """
        
        if not hasattr(t,'__len__'):
            nResults = 1
        else:
            nResults = len(t)
            
        if self.nResults==0:
            append=False # no results currently save
                       
        if append:
            
            self.t = npy.asmatrix(npy.append(self.t,npy.asmatrix(t),axis=0))
            self.f = npy.asmatrix(npy.append(self.f,f.T,axis=0))
            self.v = npy.asmatrix(npy.append(self.v,v.T,axis=0))
            self.vdot = npy.asmatrix(npy.append(self.vdot,vdot.T,axis=0))
            self.v2dot = npy.asmatrix(npy.append(self.v2dot,v2dot.T,axis=0))
            self.f_constraint = npy.asmatrix(npy.append(self.f_constraint,
                                                        f_constraint.T,axis=0))
            
        else:
                                   
            self.nDOF=v.shape[0]
            self.t=npy.asmatrix(t)
            self.f=f.T
            self.v=v.T
            self.vdot=vdot.T
            self.v2dot=v2dot.T
            self.f_constraint=f_constraint.T
            
        self.nResults += nResults
        
        
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
                
                if attr != 't':
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
        
    
    def PlotStateResults(self,*args,**kwargs):
        deprecation_warning('PlotStateResults','plot_state_results')
        return self.plot_state_results(*args,**kwargs)
    
    
    def plot_state_results(self,
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
            
            print("   Plotting state results for '%s'..." % obj.name)

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
            t = self.t
            f,v,vdot,v2dot,f_constraint =self.GetResults(obj,['f','v',
                                                            'vdot','v2dot',
                                                            'f_constraint'])

            # Handle dofs2Plot if provided
            # Note in this case the system to which dofs relate must also
            # be defined!
            if dofs2Plot is not None and dynsys_obj is not None:

                f = f[:,dofs2Plot]
                v = v[:,dofs2Plot]
                vdot = vdot[:,dofs2Plot]
                v2dot = v2dot[:,dofs2Plot]
            
            # Create time series plots
            self._TimePlot(ax=axarr[0],
                           t_vals=t,
                           data_vals=f,
                           xlim=xlim,
                           titleStr="Applied %sForces (N)" % sysStr)
            
            self._TimePlot(ax=axarr[1],
                           t_vals=t,
                           data_vals=v,
                           titleStr="%sDisplacements (m)" % sysStr)
            
            self._TimePlot(ax=axarr[2],
                           t_vals=t,
                           data_vals=vdot,
                           titleStr="%sVelocities (m/s)" % sysStr)
            
            self._TimePlot(ax=axarr[3],
                           t_vals=t,
                           data_vals=v2dot,
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
    
    
    def PlotResponseResults(self,*args,**kwargs):
        deprecation_warning('PlotResponseResults','plot_response_results')
        return self.plot_response_results(*args,**kwargs)
    
    
    def plot_response_results(self,
                              dynsys_obj=None,
                              verbose=True):
        """
        Produces a new figure with linked time series subplots for 
        all responses/outputs
        
        ***
        **Optional:**
            
        * `dynsys_obj`, can be used to specify the subsystem for which results 
          which should be plotted. If _None_ then results for all freedoms 
          (i.e. all subsystems) will be plotted.
          
        ***
        **Returns:**
         
        Nested list of figure objects
          
        """
        
        if verbose:
            print("Preparing response results plots...")
            tic=timeit.default_timer()
            
        # Get subsystems to iterate over
        DynSys_list = self.DynSys_list
        if dynsys_obj is not None:
            
            if not (dynsys_obj in DynSys_list):
                raise ValueError("`dynsys_obj` not in DynSys_list!")
                
            DynSys_list = [dynsys_obj]
    
        # Iterate over all responses for all specified subsystems
        
        fig_list = []
        
        # Loop over all specified subsystems
        for dynsys_obj, response_obj_list in zip(DynSys_list,self.response_results):
            
            print("   Plotting responses results for '{0}'..."
                  .format(dynsys_obj.name))
            
            fig_list_inner = []
            
            # Iterate over all responses for this subsystem
            for response_results_obj in response_obj_list:
                
                fig_list_inner.append(response_results_obj.plot())
                
            fig_list.append(fig_list_inner)
                
        if verbose:
            toc=timeit.default_timer()
            print("Plots prepared after %.3f seconds." % (toc-tic))
            
        return fig_list
        
    
    def PlotResults(self,*args,**kwargs):
        
        deprecation_warning('PlotResults','plot_results')
    
        return self.plot_results(*args,**kwargs)
    
    
    def plot_results(self,
                     dynsys_obj=None,
                     verbose:bool=True,
                     dofs2Plot:list=None):
        """
        Presents the results of time-stepping analysis by producing the 
        following plots:
        
        * State results plot: refer `PlotStateResults()`
        
        * Response results plot: refer `PlotResponseResults()`
        
        ***
        **Optional:**
            
        * `dynsys_obj`, object specifying system whose results are to be 
          plotted. If _None_ plots will be produced for all systems. 
        
        * `verbose`, _boolean_, if True progress will be written to console
        
        * `dof2Plot`, _list_ of integers to denote DOFs to plot. Can be used to 
          produce a 'cleaner' plot, in case of large numbers of DOFs. Note will 
          only work in conjunction with `dynsys_obj`, i.e. subsystem must also 
          be selected
        
        ***
        **Returns:**
        
        _List_ of figure objects
        
        """
        
        state_fig_list = self.plot_state_results(dynsys_obj=dynsys_obj,
                                                 verbose=verbose,
                                                 dofs2Plot=dofs2Plot)
        
        response_fig_list = self.plot_response_results(dynsys_obj=dynsys_obj,
                                                       verbose=verbose)
        
        return state_fig_list, response_fig_list
    
    
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
    

    def PlotDeformed(self,timestep_index:int,dynsys_obj=None,ax=None,**kwargs):
        """
        Produce a plot of system in its deformed configuration, as per results 
        at a given time step
        
        Additional keyword arguments are passed-down to `PlotSystem()` method 
        of applicable system object
        """
        
        # Get system to plot
        if dynsys_obj is None:
            dynsys_obj = self.tstep_obj.dynsys_obj
        
        # Get results at applicable time step
        t_val = npy.ravel(self.t)[timestep_index]
        print(t_val)
        v_vals = self.GetResults(dynsys_obj,['v'])[0][timestep_index,:]
        
        # Create figure and axes onto which to draw system, if not provided
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
            
        # Plot system onto axes
        line = dynsys_obj.PlotSystem(ax=ax,v=v_vals,**kwargs)
    
        # Overlay caption to 
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        time_text.set_text('Time = %.2fs' % t_val)
        
        return fig, line, time_text
    
    
    def AnimateResults(self,
                       dynsys_obj=None,
                       ax=None,
                       SysPlot_kwargs={},
                       FuncAnimation_kwargs={}):
        """
        Produce animation of results, plotting deformed configuration of system 
        at each time step
        
        ***
        **Required:**
        
        (No arguments required)
        
        **Optional:**
        
        * `dynsys_obj`, instance of `DynSys()` class (or derived classes), 
          defining the system to be plotted. If None (per default argument) the 
          parent system will be plotted
          
        """
        
        # Define axes to plot to
        if ax is None:
            
            fig, ax = plt.subplots()
            fig.set_size_inches((10,6))
            
        else:
            
            fig = ax.get_figure()
        
        # Create class to faciliate animation production
        sys_plot_obj = SysPlot(ax,results_obj=self,**SysPlot_kwargs)
        
        # Get tstep data
        nResults = self.nResults
        
        # Get time step between frames
        dt = self.tstep_obj.dt
        
        if dt is None:
            # Compute average time step
            dt = (self.t[-1]-self.t[0])/nResults
        
        # Define default kwargs to pass to FuncAnimation
        FuncAnimation_kwargs_0 = {'frames':npy.arange(0,nResults),
                                  'interval':1000*dt,
                                  'repeat':False}
        
        # Merge with any passed kwargs: n.b passed kwargs override defaults
        FuncAnimation_kwargs = {**FuncAnimation_kwargs_0,
                                **FuncAnimation_kwargs}
        
        # Create animation
        anim = FuncAnimation(fig, sys_plot_obj.update,
                             **FuncAnimation_kwargs)
        
        return anim
        
        
        
        
    def calc_responses(self,
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
        
        # Calculate responses for all systems and subsystems
        for x in dynsys_obj.DynSys_list:
            
            responses_inner_list = []
            
            # Retrieve state variables for subsystem
            v, vdot, v2dot = self.GetResults(x,['v', 'vdot', 'v2dot'])
            state_vector = npy.hstack((v,vdot,v2dot))
            
            # Calculate responses for each set of outputs for this subsystem
            for _om, _names in zip(x.output_mtrx,x.output_names):
                
                values = _om * state_vector.T
                
                obj = TimeSeries_Results(names=_names,
                                         values=values,
                                         tstep_results_obj=self)
                
                responses_inner_list.append(obj)
                
            responses_list.append(responses_inner_list)
            
        # Store as attributes
        self.response_results = responses_list
                
        # Calculate DOF statistics
        if self.options['calc_dof_stats']:
            self.calc_dof_stats(verbose=verbose)
            
        # Calculate response statistics
        if self.options['calc_response_stats']:
            self.calc_response_stats(verbose=verbose)
            
        # Write time series results to file
        if write_results_to_file:
            self.WriteResults2File(output_fName=results_fName)
                
        # Delete DOF time series data (to free-up memory)
        if not self.options['retain_dof_values']:
            if verbose:
                print("Clearing DOF time series data to save memory...")
            del self.v
            del self.vdot
            del self.v2dot
        
        # Delete response time series data (to free up memory)
        if not self.options['retain_response_values']:
            if verbose:
                print("Clearing response time series data to save memory...")
            del self.responses_results
        
    
    def CalcKineticEnergy(self):
        """
        Calculates total kinetic energy of system at each time step:
        
        $$ T(t) = \dot{y}(t)^{T} M \dot{y}(t) $$
        
        """
        
        nResults = self.nResults
        
        M = self.tstep_obj.dynsys_obj.GetSystemMatrices()["M_mtrx"]
        vdot = self.vdot
        
        KE = npy.empty((nResults,))
        
        for i in range(nResults):
        
            KE[i] = vdot[i,:] @ M @ vdot[i,:].T
            
        self.KE = KE
        return KE
    
    
    def CalcPotentialEnergy(self):
        """
        Calculates total potential energy of system at each time step:
        
        $$ V(t) = y(t)^{T} K y(t) $$

        """
        
        nResults = self.nResults
        
        K = self.tstep_obj.dynsys_obj.GetSystemMatrices()["K_mtrx"]
        v = self.v
        
        PE = npy.empty((nResults,))
        
        for i in range(nResults):
        
            PE[i] = v[i,:] @ K @ v[i,:].T
            
        self.PE = PE
        return PE
   
    
    def CalcExternalWorkPower(self):
        """
        Calculates power of work done by external forces at each time step:
            
        $$ P(t) = f(t)^{T} \dot{v}(t) $$
        
        """
        
        nResults = self.nResults
        
        f = self.f
        vdot = self.vdot
        
        ExtWork_Power = npy.empty((nResults,))
        
        for i in range(nResults):
        
            ExtWork_Power[i] = f[i,:] @ vdot[i,:].T
            
        self.ExtWork_Power = ExtWork_Power
        return ExtWork_Power
    
    
    def CalcExternalWorkDone(self):
        """
        Calculates work done by external forces since t=0
        
        ***
        _Practically this is done by integrating the power of work done by 
        external forces as calculated by `CalcExternalWorkPower()` method. 
        Note integration is approximate; trapezium rule is used_
        """
        
        t = npy.ravel(self.t)
        ExtWork_power = self.CalcExternalWorkPower()
        
        # Integrate power of external work to get total external work from t=0
        ExtWork = scipy.integrate.cumtrapz(y=ExtWork_power, x=t, initial=0.0)
        
        self.ExtWork = ExtWork
        return ExtWork
    
    
    def PlotEnergyResults(self,recalculate=True):
        """
        Produce figure with subplots to show key energy results of analysis
        e.g. external work done, kinetic & potential energies
        """
        
        if recalculate:
            KE = self.CalcKineticEnergy()
            PE = self.CalcPotentialEnergy()
            ExtWork = self.CalcExternalWorkDone()
        else:
            KE = self.KE
            PE = self.PE
            ExtWork = self.ExtWork
            
        # Back-calculate energy dissipated
        E0 = PE[0] + KE[0]              # energy of system at t=0
        ED = ExtWork - (PE+KE) + E0     # energy dissipated since t=0
                        
        # Get time values for plots
        t = npy.ravel(self.t)
        
        tstep_obj = self.tstep_obj

        # Create new figure for plots
        fig,axarr = plt.subplots(3,1,sharex=True)
        fig.set_size_inches((10,7))
        fig.suptitle("Energy results\nAnalysis: '%s'" % tstep_obj.name)
        fontsize_titles = 10
        
        # Define individual subplots
        ax = axarr[0]
        ax.plot(t,ExtWork,label='External work done')
        ax.plot(t,(KE+PE),label='Conserved energy, $E=T+V$')
        ax.plot(t,ED,label='Energy dissipated (from energy balance)')
        ax.plot(t,(KE-PE),color='k',alpha=0.3,label='Lagrangian, $L=T-V$')
        ax.set_title("External/Internal work done",fontsize=fontsize_titles)
        ax.legend(fontsize=8)
        
        ax = axarr[1]
        ax.plot(t,KE)
        ax.set_title("Kinetic energy: T(t) = $\dot{y}(t)^{T}M\dot{y}(t)$",
                     fontsize=fontsize_titles)
        
        ax = axarr[2]
        ax.plot(t,PE)
        ax.set_title("Potential energy: V(t) = $y(t)^{T}Ky(t)$",
                     fontsize=fontsize_titles)
        
        # Set details for shared x-axis
        ax.set_xlim([tstep_obj.tStart,tstep_obj.tEnd])
        ax.set_xlabel("Time $t$ (secs)")
        
        return fig
        
    
        
    def calc_dof_stats(self,verbose=True):
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
        
        if self.options['calc_dof_stats']:
            
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
                print("'calc_dof_stats' option set to False\n" +
                      "DOF statistics will not be computed")
        
        return stats
        
    
    def calc_response_stats(self,verbose=True):
        """
        Calculate basic statistics to describe response time series
        """
        
        if self.options['calc_response_stats']:
            
            # Get paired lists to loop over
            response_obj_list = self.response_results
            dynsys_obj_list = self.tstep_obj.dynsys_obj.DynSys_list
            
            # Loop over all systems and subsystems
            for dynsys_obj, response_objs in zip(dynsys_obj_list,
                                                 response_obj_list):
                
                if verbose:
                    print("Calculating response statistics " + 
                          "for '{0}'...".format(dynsys_obj.name))

                for obj in response_objs:
                    obj.calc_stats()
                                
        else:
            if verbose:
                print("'calc_response_stats' option set to False\n" + 
                      "Response statistics will not be computed")
            
    
    def get_response_stats_df(self):
        """
        Returns response stats as list of Pandas dataframes
        """
        
        df_list = []
        
        for dynsys_obj, results_obj_list in zip(self.DynSys_list,
                                                self.responses_list):
                    
            for results_obj in results_obj_list:
            
                response_names = results_obj.names
                stats_dict = results_obj.stats

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
        
        

class SysPlot():
    """
    Class to faciliate animation of displacement results 
    as held in `tstep_results`
    """
    
    def __init__(self, ax, results_obj,
                 y_lim=None,
                 plot_loading=True,
                 load_scale=None,
                 load_scale_relative=0.3,
                 time_template = 'Time = %.2fs',
                 time_text_loc=(0.85, 0.95),):
        """
        Animation plot initialisation method
        
        ***
        Required:
            
        * `ax`, axes to plot to
        
        * `results_obj`, instance of `tstep_results` class
        
        ***
        Optional:
        
        * `time_text_loc`, tuple, defines location of time caption
        
        * `time_template`, string, text for time caption
        
        * `load_scale`, factor used to convert loads (N) to distances (m). If 
          _None_ (default) then `load_scale` will be set as proportion of 
           vertical scale according to `load_scale_relative`
        
        """
        
        # Get objects
        self.results_obj = results_obj
        
        # Get dynamic system object
        tstep_obj = results_obj.tstep_obj
        dynsys_obj = tstep_obj.dynsys_obj
        self.dynsys_obj = dynsys_obj
        
        # Call plot initialisation method of dynsys object
        dynsys_obj.PlotSystem_init_plot(ax)
        
        # Overlay plot of loading, if avaliable
        if plot_loading:
            
            attr = 'analysis_obj'
            if hasattr(results_obj,attr):
                
                analysis_obj = getattr(results_obj,attr)
                loading_obj = analysis_obj.loading_obj
                loading_obj.plot_init(ax=ax)
                
            else:
                loading_obj = None
                plot_loading = False # cannot plot loading
                print("Warning: could not plot loading")
        
        self.plot_loading = plot_loading
        """
        Boolean, denotes whether loading should be overlaid onto plots
        """
        
        self.loading_obj = loading_obj
        """
        Loading object
        """
        
        
        # Set y scale for plot
        if  y_lim is None:
            
            # Attempt to determine appropriate y limits
            v = results_obj.v
            y_absmax = npy.max(npy.abs(v))
            y_lim = (1.2*npy.array([-y_absmax,+y_absmax])).tolist()
            
        ax.set_ylim(y_lim)
        
        # Determine appropriate scale for loading in plot
        if plot_loading and load_scale is None:
            
            loadVals = loading_obj.loadVals
            
            max_load = npy.max(npy.abs(loadVals))
            load_scale = load_scale_relative * (y_lim[1] / max_load)
        
        self.load_scale = load_scale
        """
        Scale factor used to convert load units (N) to distance (m)
        """
        
        # ----------------------------------------------------------
        
        ax.set_title("Displacement results\n" + 
                     "Analysis: '%s'\n" % tstep_obj.name + 
                     "System: '%s'" % dynsys_obj.name)
        
        self.time_template = time_template
        self.time_text = ax.text(*time_text_loc, '',
                                 fontsize=8,transform=ax.transAxes)
        
        ax.legend(loc='lower right')
        

    def update(self, i):
        """
        Animation plot update method, for results time step (frame) `i`
        """
        
        results_obj = self.results_obj
        
        
        # Get results applicable to this time increment
        t = self.results_obj.t[i,0]
        v = self.results_obj.v[i,:]
        
        # Call plot update method of dynsys object
        lines = self.dynsys_obj.PlotSystem_update_plot(v=v)
        
        # Overlay plot of loading, if avaliable
        if self.plot_loading:
            
            analysis_obj = results_obj.analysis_obj
            loading_obj = self.loading_obj
            load_scale = self.load_scale
            
            load_velocity = analysis_obj.loadVel
            
            load_lines = loading_obj.plot_update(t=t,lead_x=load_velocity*t,
                                                 load_scale=load_scale)
            lines['load_lines'] = load_lines
                        
        # Update time caption
        self.time_text.set_text(self.time_template % (t))
        
        return lines
    
    
    
class TimeSeries_Results():
    """
    Class to act as container for time series results, with methods to compute 
    statistics etc
    """
    
    def __init__(self,names,values,tstep_results_obj,
                 location_obj=None,
                 plot_options={'use_common_scale' :True}
                 ):
        
        self.names = names
        self.values = values
        self.tstep_results_obj = tstep_results_obj
        self.location_obj = location_obj
        self.plot_options = plot_options
        
        self.dynsys_obj = tstep_results_obj.tstep_obj.dynsys_obj
        
        
    # ----------------
    @property
    def names(self):
        """
        List of names of defined responses
        """
        return self._names
    
    @names.setter
    def names(self,value):
        self._names = value
    
    # ----------------
    @property
    def values(self):
        """
        Time series results
        """
        return self._values
    
    @values.setter
    def values(self,value):
        self._values = value
        
    # ----------------
    @property
    def location_obj(self):
        """
        Location object to which results relate
        """
        return self._location_obj
    
    @location_obj.setter
    def location_obj(self,obj):
        if obj is not None:
            check_is_class(obj)
        self._location_obj = obj
        
    # ----------------
    @property
    def stats(self):
        """
        Dict containing statistics dervied from time series results
        """
        return self._stats_dict
    
    # Note no setter method - cannot set stats directly!
    # ----------------
        
    def calc_stats(self):
        """
        Calculate basis statistics to summarise time series results
        """
        
        vals = self.values
        
        # Record stats within a dict
        stats_dict={}
        stats_dict["max"] = npy.ravel(npy.max(vals,axis=1))
        stats_dict["min"] = npy.ravel(npy.min(vals,axis=1))
        stats_dict["std"] = npy.ravel(npy.std(vals,axis=1))
        stats_dict["absmax"] = npy.ravel(npy.max(npy.abs(vals),axis=1))
            
        self._stats_dict = stats_dict
        
        return stats_dict
    
    
   
    
    
    def plot(self,axarr=None):
        """
        Plots time series results along with related stats
        """
        
        tstep_results_obj = self.tstep_results_obj
        tstep_obj = tstep_results_obj.tstep_obj
        
        nResponses = len(self.names)
        
        if axarr is None:
            fig, axarr = plt.subplots(nResponses,sharex=True)
        
        # Get options
        use_common_scale = self.plot_options['use_common_scale']
        
        # Get time values and time interval
        t = tstep_results_obj.t
        tInterval= [tstep_obj.tStart,tstep_obj.tEnd]
        
        # Determine common scale to use for plots
        if use_common_scale:
            
            maxVal = npy.max(self.values)
            minVal = npy.min(self.values)
            absmaxVal = npy.max([maxVal,minVal])
            
        # Loop through plotting all responses
        
        for r, (vals, names) in enumerate(zip(self.values,self.names)):
            
            # Get axis object
            if hasattr(axarr, "__len__"):
                ax = axarr[r]
            else:
                ax = axarr
                        
            # Make time series plot
            ax.plot(t,vals.T,label=names)
                                   
            # Set axis limits and labels
            ax.set_xlim(tInterval)
            
            if r == nResponses-1:
                ax.set_xlabel("Time [s]")
            
            if use_common_scale:
                ax.set_ylim([-absmaxVal,+absmaxVal])
            
            # Create legend
            if names is not None:
                ax.legend(loc='right')
                
        return fig
        
        
        

