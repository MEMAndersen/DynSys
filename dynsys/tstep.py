# -*- coding: utf-8 -*-
"""
Classes and methods used to facilate time-stepping analysis involving dynamic 
systems
"""

# ********************* IMPORTS **********************************************
import numpy as npy
import timeit
import inspect
from scipy.integrate import solve_ivp

import tstep_results
    
# ********************** CLASSES *********************************************

class TStep:
    """
    Class used to implement time-stepping analysis, i.e. to determine the 
    time-varying response of a dynamic system, given known initial conditions 
    and external loading
    
    The ODE solution algorithm used is provided by Scipy (v1.0 and above).
    Refer 
    [Scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp) 
    for further details: 
    
    """
    
    def __init__(self,
                 dynsys_obj,
                 name=None,
                 tStart=0, tEnd=30.0,
                 dt=None, max_dt=0.1,
                 retainDOFTimeSeries=True,
                 retainResponseTimeSeries=True,
                 writeResults2File=False,
                 results_fName="results.csv",
                 plotResponseResults=True,
                 responsePlot_kwargs={},
                 x0=None,
                 force_func:callable=None,
                 event_funcs:callable=None,
                 post_event_funcs:callable=None,
                 ):
        """
        Initialises time-stepping analysis
        
        ***
        Required:
            
        * `dynsys_obj`, instance of `dynsys` class (or derived classes), used 
          to define the dynamic system to which the analysis relates
        
        ***
        Optional:
        
        * `tStart`, start time (secs)
            
        * `tEnd`, end time (secs)
        
        * `dt`, constant time-step to use. If `None` then results will only be 
          returned at time steps chosen by `scipy.integrate.solve_ivp()`.
        
        * `max_dt`, maximum time-step to use. Only applies if `dt=None`.
        
        * `x0`, _array-like_ defining initial conditions of freedoms. If `None` 
          then zeros will be assumed.
          
        * `force_func`, _callable_, used to define applied external forces. 
          If `None` then zero external forces will be assumed.
        
        * `event_funcs`, _callable_ or _list of callables_, 
          events to track. Events are defined by functions which take a zero 
          value at the time of an event. Functions should have attribute 
          `terminal` and `direction` assigned to them to describe the required 
          behaviour. Refer [Scipy documentation] for further details.
          
        * `post_event_funcs`, _callable_ or _list of callables_, functions to 
          execute immediately after `event_funcs` have resolved. If list, 
          length must correspond to length of `event_funcs`.
          
          [Scipy documentation]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        
        Note one of `x0` or `force_func` must be provided (this is checked) 
        otherwise the dynamic system in question will not do anything!
        
        """
  
        # Write basic details to object   
        
        self.name = name
        """
        String identifier for object
        """
        
        self.tStart = tStart
        """
        Time denoting start of analysis
        """
        
        self.tEnd = tEnd
        """
        Time denoting end of analysis
        """
        
        self.dt = dt
        """
        Constant time step to evaluate results at.
        """
        
        self.max_dt = max_dt
        """
        Maximum time step, as used by `solve_ivp` to control ODE solution
        """
        
        self.dynsys_obj = dynsys_obj
        """
        `DynSys` class instance: defines dynamic system to which time-stepping 
        analysis relates
        """
        
        # Check either initial conditions set or force - otherwise nothing will happen!
        if x0 is None and force_func is None:
            raise ValueError("Either `x0` or `force_func` required, " + 
                             "otherwise nothing will happen!")
        
        # Set initial conditions
        if x0 is None:
            
            # By default set initial conditions to be zeros
            x0 = npy.zeros((2*dynsys_obj.nDOF,))
            
        else:
            
            # Flatten array
            x0 = npy.ravel(npy.asarray(x0))
        
            # Check shape of initial conditions vector is consistent with dynsys
            if x0.shape[0] != 2*dynsys_obj.nDOF:
            
                raise ValueError("Error: `x0` of unexpected shape!\n" + 
                                 "dynsys_obj.nDOF: {0}".format(dynsys_obj.nDOF) + 
                                 "x0.shape: {0}".format(x0.shape)) 
                
        self.x0 = x0
        """
        Initial conditions vector
        """
        
        # Set applied forces
        self.force_func = self._check_force_func(force_func)
        """
        Function to define external loading vector at time t
        """

        # Set events
        self.event_funcs = self._check_event_funcs(event_funcs)
        """
        List of functions which define _events_. 
        Refer [solve_ivp] documentation for requirements for how event 
        functions should be defined.
        
        [solve_ivp]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        """
        
        # Set post-e
        self.post_event_funcs = self._check_post_event_funcs(post_event_funcs)
        """
        List of functions to execute directly after the occurence of _events_
        """
        
        self.writeResults2File = writeResults2File
        """
        _Boolean_, controls whether time series results will be written to 
        file at the end of time-stepping analysis
        """
        
        self.plotResponseResults = plotResponseResults
        """
        _Boolean_, controls whether response results plot should made 
        at the end of time-stepping analysis
        """
        
        self.responsePlot_kwargs = responsePlot_kwargs
        """
        Dict containing option arguments for response plotting function
        """
        
        self.results_fName = results_fName
        """
        File to write time-series results to
        """
        
        # Create object to write results to
        self.results_obj=tstep_results.TStep_Results(self,
                                                     retainDOFTimeSeries=retainDOFTimeSeries,
                                                     retainResponseTimeSeries=retainResponseTimeSeries)
        """
        Results (displacements, velocities, constraint forces etc.) are stored 
        in this object, which provides useful functionality for computing 
        statistics, plotting etc.
        """
        
    
    def _check_force_func(self,force_func):
        """
        Function checks that `force_func` as supplied in TStep __init__() function 
        is appropriate
        """
    
        # Handle `None` case
        if force_func is None:
                
            # Define null force function
            def null_force(t):
                return npy.zeros((expected_nDOF,))
            
            force_func = null_force
        
        # Check force_func is a function
        if not inspect.isfunction(force_func):
            raise ValueError("`force_func` is not a function!")
            
        # Check force_func has `t` as first argument
        sig = inspect.signature(force_func)
        if not 't' in sig.parameters:
            raise ValueError("1st argument of `force_func` must be `t`\n" + 
                             "i.e. `force_func` must take the form " +
                             "force_func(t,*args,**kwargs)")
            
        # Check dimension of vector returned by force_func is of the correct shape
        expected_nDOF = self.dynsys_obj.nDOF
        
        t0 = self.tStart
        force0 = force_func(t0)
        
        if isinstance(force0,list):
            if force0.shape[0]!=expected_nDOF:
                raise ValueError("`force_func` returns vector of unexpected shape!\n" + 
                                 "Shape expected: ({0},)\n".format(expected_nDOF) + 
                                 "Shape received: {0}".format(force0.shape))
            
        return force_func
    
        
    def _check_event_funcs(self,event_funcs):
        """
        Function checks that `event_funcs` as supplied in TStep __init__() function 
        are appropriate
        """
        
        # Handle None case
        if event_funcs is None:
            return event_funcs    
            
        # Convert to list
        if type(event_funcs) is not list:
            event_funcs = [event_funcs]
            
        i = 0
        for _event_func in event_funcs:
            
            # Check _event_func is a function
            if not inspect.isfunction(_event_func):
                raise ValueError("events_funcs[{0}] is not a function!".format(i))
            
            # Check _event_func has the right form: f(t,y) is required
            sig = inspect.signature(_event_func)
            
            if str(sig.parameters[0])!='t':
                raise ValueError("1st argument of events_funcs[{0}] " + 
                                 "must be `t`\n".format(i) + 
                                 "Event functions must take the form " +
                                 "f(t,y)")
                
            if str(sig.parameters[0])!='y':
                raise ValueError("2nd argument of events_funcs[{0}] " + 
                                 "must be `y`\n".format(i) + 
                                 "Event functions must take the form " +
                                 "f(t,y)")
                
            # Check dimension of vector returned by force_func is float
            val  = _event_func(0.0,npy.zeros())
            if type(val) is not float:
                raise ValueError("events_funcs[{0}] must return float".format(i) + 
                                 "Events are defined at the time t* when event function " + 
                                 "f(t*,y)=0")
            
            i += 1
            
        return event_funcs
        
            
    def _check_post_event_funcs(self,post_event_funcs):
        """
        Function checks that `post_event_funcs` as supplied in TStep's 
        __init__() function are appropriate
        """
        
        # Handle None case
        if post_event_funcs is None:
            return post_event_funcs    
            
        # Convert to list
        if type(post_event_funcs) is not list:
            post_event_funcs = [post_event_funcs]
            
        # Check list length
        if len(post_event_funcs)!=len(self.event_funcs):
            raise ValueError("Length of `post_event_funcs` list must " + 
                             "equal to length of `event_funcs`!")
            
        i = 0
        for _func in post_event_funcs:
            
            # Check _event_func is a function
            if not inspect.isfunction(_func):
                raise ValueError("post_events_funcs[{0}] is not a function!".format(i))
        
            i += 1
        

    def run(self,method='RK45',showMsgs=True):
        """
        Runs time-stepping analysis
        ***
        
        Solution is obtained using Scipy's ODE solver for initial value 
        problems [solve_ivp].
        
        Optional argument `method` can be used to specify the particular 
        solver type to use. Refer Scipy docs for details of the 
        options avaliable.
        
        `RK45` is the default solver. As described in the documentation for 
        [solve_ivp], this is an explicit Runge-Kutta method of order 5(4). 
        This should be appropriate for most applications.
        
        [solve_ivp]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        """
        
        if showMsgs: 
            if self.name is None:
                print("Running time-stepping analysis...")
            else:
                print("Running time-stepping analysis: %s" % self.name)
        
        # Retrieve solver params from class attributes
        tmin = self.tStart
        tmax = self.tEnd
        y0 = self.x0
        results_obj = self.results_obj
        
        # Define keyword arguments for solve_ivp
        kwargs = {}
        kwargs["method"]=method                
        kwargs["events"]=self.event_funcs
        
        # Print to denote parameters used
        if showMsgs: print("Analysis time interval: [%.2f, %.2f] seconds" % (tmin,tmax))
        
        if self.dt is not None:
            if showMsgs: print("Fixed time step specified: %.2e seconds" % self.dt)
            kwargs["t_eval"]=npy.arange(tmin,tmax,self.dt)
        else:
            if self.max_dt is not None:
                if showMsgs: print("Maximum time step specified: %.3f seconds" % self.max_dt)
                kwargs["max_step"]=self.max_dt
        
        # Define ODE function in the expected form dy/dt = f(t,y)
        eqnOfMotion_func = self.dynsys_obj.EqnOfMotion
        
        def ODE_func(t,y):
            
            results = eqnOfMotion_func(t=t,x=y,forceFunc=self.force_func)
            
            # Return xdot as flattened array
            ydot = results["ydot"]
            y2dot = results["y2dot"]
            xdot = npy.ravel(npy.vstack((ydot,y2dot)))

            return xdot
        
        # Run solver
        terminateSolver = False
        solvecount = 0
        eventcount = 0
        
        solve_time=0
        resultsproc_time=0
        
        sol_list=[]
        
        while not terminateSolver:
            
            tic=timeit.default_timer()
            
            # Run solution
            sol = solve_ivp(fun=ODE_func, t_span=[tmin,tmax], y0=y0, **kwargs)
            sol_list.append(sol)
            solvecount += 1
            
            toc=timeit.default_timer()
            solve_time += toc-tic
            
            # Post-process results
            for n in range(len(sol.t)):
                
                # Solve equation of motion
                results = eqnOfMotion_func(t=sol.t[n],
                                           x=sol.y[:,n],
                                           forceFunc=self.force_func)
            
                # Record results
                tic=timeit.default_timer()
                
                results_obj.RecordResults(t=results["t"],
                                          f=results["f"],
                                          v=results["y"],
                                          vdot=results["ydot"],
                                          v2dot=results["y2dot"],
                                          f_constraint=results["f_constraint"])
                
                toc=timeit.default_timer()
                resultsproc_time += toc-tic
            
            # Handle solver status 
            if sol.status == 1:
                # termination event occurred
                
                # Register new event
                eventcount += 1
                if eventcount == 1:
                    t_events = sol.t_events[0]
                else:
                    t_events = npy.append(t_events,sol.t_events[0])
                
                # Set new initial conditions
                # Run post-event function
                t_current = t_events[-1]
                y_current= sol.y[:,-1]
                t, y = self.post_event_funcs[0](t_current,y_current)
            
            elif sol.status == 0:
                # The solver successfully reached the interval end
                
                terminateSolver = True
                if showMsgs: print("Analysis complete!")
                if showMsgs: print(sol.message) 
                
            else:
                
                raise ValueError("Integration failed.")
                
        # Calculate responses
        results_obj.CalcResponses(write_results_to_file=self.writeResults2File,
                                  results_fName=self.results_fName,
                                  showMsgs=showMsgs)
        
        if showMsgs: print("Total time steps: {0}".format(results_obj.nResults))        
        if showMsgs: print("Overall solution time: %.3f seconds" % solve_time)
        if showMsgs: print("Overall post-processing time: %.3f seconds" % resultsproc_time)
        
        return self.results_obj
    
        
        
# ********************** FUNCTIONS *******************************************



    

# ********************** TEST ROUTINE ****************************************

if __name__ == "__main__":

    import msd_chain
    
    # Define dynamic system
    mySys = msd_chain.MSD_Chain([100,50,100,50],
                                [1.2,1.8,2.0,4.5],
                                [0.03,0.02,0.01,0.1],
                                isSparse=False)
    
    mySys.AddConstraintEqns(Jnew=[[1,-1,0,0],[0,0,1,-1]])
    mySys.PrintSystemMatrices(printShapes=True,printValues=True)
    
    # Define applied forces
    def sine_force(t,F0,f):
        
        F0 = npy.asarray(F0)
        f = npy.asarray(f)
        
        return F0*npy.sin(2*npy.pi*f*t)
    
    F0_vals = [200,0,0,0]
    f_vals=[1.0,0,0,0]

    # Run time-stepping and plot results
    myTStep = TStep(mySys,
                    force_func=lambda t: sine_force(t,F0_vals,f_vals),
                    tEnd=10.0,
                    max_dt=0.01)
    myTStep.run()
    res = myTStep.results_obj
    res.PlotStateResults()
    
    # Define output matrix to return relative displacements
    outputMtrx = npy.asmatrix([[1,-1,0,0,0,0,0,0,0,0,0,0],[0,1,-1,0,0,0,0,0,0,0,0,0]])
    res.CalcResponses(outputMtrx,["Rel disp 12","Rel disp 23"])
    res.PlotResponseResults()