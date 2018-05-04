# -*- coding: utf-8 -*-
"""
Classes and methods used to define loading to apply to dynamic systems
"""

from __init__ import __version__ as currentVersion

import numpy
import pandas as pd
import inspect
import scipy
import matplotlib.pyplot as plt

from numpy import sin, pi

class Loading():
    """
    Generic loading class
    
    _Inheritance expected_
    """
    
    def __init__(self,name):
        """
        Initialisation function
        """
        self.name = name
    
    
    def PrintDetails(self):
        """
        Prints details of loading definition to std output
        """
        print("Loading type: \t\t'{0}'".format(self.__class__.__name__))
        print("Loading object name: \t'%s'" % self.name)
    

class LoadTrain(Loading):
    """s
    Defines a series of point loads at fixed relative positions
    """
    
    def __init__(self,
                 loadX=None,
                 loadVals=None,
                 name=None,
                 intensityFunc=None,
                 fName="loadDefs.csv"):
        """
        Define a pattern of point loads at fixed relative positions
        ***
        
        Input may be defined by either `loadX` and `loadVals` or by providing 
        filename of a tabular comma-delimited file format with columns | Xpos | Load |.
        
        Details read-in from file are written to object as instance variables 
        (refer descriptions below)
        """
        
        if loadX is None or loadVals is None:
            
            if name is None:
                name=fName
            
            # Read data from text file
            loadData = numpy.genfromtxt(fName,delimiter=',',skip_header=1)
        
            # Get load position and value from txt file
            loadX = loadData[:,0]
            loadVals = loadData[:,1]
            
            print("Load train defined via definitions from '{0}'".format(fName))
        
        else:
            
            loadX = numpy.array(loadX)
            loadVals = numpy.array(loadVals)
            
            if loadX.shape != loadVals.shape:
                raise ValueError("Shapes of `loadX` and `loadVals` differ!\n"+
                                 "loadX:\t {0}\n".format(loadX.shape) +
                                 "loadVals:\t {0}\n".format(loadVals.shape))
        
        # Get length of load pattern
        loadLength = numpy.max(loadX) - numpy.min(loadX)
        
        # Set such that lead axle at X = 0
        loadX = loadX - numpy.max(loadX)
        
        # Save as attributes
        super().__init__(name=name)
        
        self.loadX = loadX
        """
        Relative position of loads. Lead load is defined to be at X=0, 
        subsequent loads are at X<0 (_adjustment is made by this function_)
        """
        
        self._loadVals = loadVals
        """
        Intensity of individual loads (in N)
        """
        
        self.loadLength = loadLength
        """
        Defines distance between lead and trailing loads
        """
        
        if intensityFunc is None:
            def unity(t):
                return 1.0
            intensityFunc=unity
        
        if not inspect.isfunction(intensityFunc):
            raise ValueError("`intensityFunc` invalid: function required!")
        
        sig = inspect.signature(intensityFunc)
        if list(sig.parameters)[0]!='t':
            raise ValueError("1st argument of `intensityFunc` must be `t`")
        
        self.intensityFunc = intensityFunc
        """
        Function of the form f(t), used as a time-varying multiplier on 
        otherwise constant point loads defined by this class
        """
        
        
        
        
    def loadVals(self,t):
        """
        Returns intensity of point loads at time t
        """
        return self.intensityFunc(t)*self._loadVals
        
    
    def PrintDetails(self):
        """
        Prints details of loading definition to std output
        """

        # Run parent function
        super().PrintDetails()
        
        # Assemble pandas dataframe
        print("Load pattern length: \t {0}".format(self.loadLength))
        print("X positions of loads:\t {0}".format(self.loadX))
        print("Load intensities:\t {0}".format(self._loadVals))
        print("Intensity function:\t {0}".format(self.intensityFunc))
        
        
class UKNA_BSEN1991_2_walkers_joggers_loading(LoadTrain):
    """
    Defines moving point load to represent the action of walkers / joggers
    per NA.2.44.4 of BS EN 1991-2
    
    ![NA.2.44.4](../dynsys/img/UKNA_BSEN1991_2_NA2_44_4.PNG)
    """
    
    def __init__(self,
                 fv:float,
                 gamma:float=1.0,
                 N:int=2,
                 analysis_type:str="walkers"):
        """
        Defines fluctuating point load to represent either walkers or joggers 
        according to NA.2.44.4, UK NA to BS EN 1991-2
        
        ***
        Required:
        
        * `fv`, natural frequency (Hz) of the mode under consideration
        
        ***
        Optional:
        
        * `gamma`, reduction factor to allow for unsynchronised actions in a 
          pedestrian group. Value of 1.0 used by default (conservative)
          
        * `N`, _integer_ number of pedestrians in group. Default value 
          corresponds to bridge class A; but actual value should generally be 
          provided
          
        * `analysis_type`, _string_, either 'walkers' or 'joggers' required, 
          to denote the case under consideration
        
        """
        
        # Determine F0 from Table NA.8
        if analysis_type=="walkers":
            F0 = 280
        elif analysis_type=="joggers":
            F0 = 910
        else:
            raise ValueError("Invalid 'analysis_type'!" + 
                             "'walkers' or 'joggers' expected")
            
        # Get k(fv) factor
        k = UKNA_BSEN1991_2_Figure_NA_8(fv=fv,analysis_type=analysis_type)
            
        # Calculate amplitude of sinusoidal forcing function per NA.2.44.4(1)
        F_amplitude = F0 * k * (1 + gamma*(N-1))**0.5
        
        # Define sinusoidal function of unit amplitude
        def sine_func(t):
            return sin(2*pi*fv*t)
        
        # Run init for parent 'LoadTrain' class
        super().__init__(loadX=[0.0],
                         loadVals=F_amplitude,
                         intensityFunc=sine_func,
                         name=analysis_type)
        
        # Save other attributes
        self.F0 = F0
        """
        Reference amplitude of applied fluctuating force (N)
        """
        
        self.N = N
        """
        Number of pedestrians in the group
        """
        
        self.gamma = gamma
        """
        Reduction factor, in the range [0,1.0], to allow for unsynchronised 
        actions in a pedestrian group
        """
        
        self.fv = fv
        """
        Natural frequency (Hz) of the mode for which loading has been derived
        """
        
        self.k = k
        """
        Factor to account for:
            
        * The effects of a more realistic pedestrian population
        * Harmonic responses
        * Relative weighting of pedestrian sensitivity to response
        """
        
        self.F_amplitude = F_amplitude
        """
        Amplitude of sinusoidal moving load (N)
        calculated according to NA.2.44.4(1)
        """
        
        
# ********************** FUNCTIONS ****************************************
        
def UKNA_BSEN1991_2_Figure_NA_8(fv,
                                analysis_type="walkers",
                                kind='cubic',
                                makePlot=False):
    """
    Returns $k_{v}(f)$ from Figure NA.8 in BS EN 1992-1:2003
    
    ***
    Required:
        
    * `fv`, mode frequency in Hz to evaluate kv at
    
    ***
    Optional:
    
    * `analysis_type`, _string_, either 'walkers' or 'joggers'
    
    * `kind`, keyword argument used by scipy.interpolate.interp1d function. 
      Defines method of interpolation, e.g. 'linear' or 'cubic'
      
    * `makePlot`, _boolean_, if True plot will be made akin to Figure NA.8
    
    """
    
    # Arrays to digitise Figure NA.8
    walkersData = [[0.000,0.000], [0.200,0.000], [0.400,0.010], [0.600,0.030],
                   [0.800,0.080], [1.000,0.240], [1.200,0.440], [1.400,0.720],
                   [1.600,0.930], [1.700,0.980], [1.800,1.000], [2.000,0.997],
                   [2.100,0.970], [2.200,0.900], [2.400,0.650], [2.600,0.400],
                   [2.800,0.250], [3.000,0.280], [3.200,0.320], [3.400,0.340],
                   [3.600,0.360], [3.800,0.360], [4.000,0.350], [4.500,0.280],
                   [5.000,0.180], [5.500,0.130], [6.000,0.115], [6.500,0.098],
                   [7.000,0.080], [8.000,0.020], [9.000,0.000]]
    
    walkersData = numpy.array(walkersData)
    
    walkersFunc = scipy.interpolate.interp1d(x=walkersData[:,0],
                                             y=walkersData[:,1],
                                             kind=kind,
                                             bounds_error=False,
                                             fill_value=0.0)
    
    joggersData = [[0.000,0.000], [0.200,0.000], [0.400,0.000], [0.600,0.000],
                   [0.800,0.000], [1.000,0.010], [1.200,0.040], [1.400,0.150],
                   [1.600,0.300], [1.700,0.450], [1.800,0.550], [2.000,0.870],
                   [2.100,1.010], [2.200,1.110], [2.400,1.160], [2.600,1.120],
                   [2.800,0.930], [3.000,0.640], [3.200,0.360], [3.400,0.160],
                   [3.600,0.100], [3.800,0.130], [4.000,0.160], [4.500,0.210],
                   [5.000,0.220], [5.500,0.180], [6.000,0.110], [6.500,0.040],
                   [7.000,0.030], [8.000,0.020], [9.000,0.000]]
    
    joggersData = numpy.array(joggersData)
    
    joggersFunc = scipy.interpolate.interp1d(x=joggersData[:,0],
                                             y=joggersData[:,1],
                                             kind=kind,
                                             bounds_error=False,
                                             fill_value=0.0)
    
    # Get applicable array to use
    if analysis_type=="walkers":
        k_fv_func = walkersFunc
    elif analysis_type=="joggers":
        k_fv_func = joggersFunc
    else:
        raise ValueError("Invalid 'analysis_type' specified!")
    
    # Use interpolation function to read off value at fv
    k_fv = k_fv_func(fv)
    
    # Make plot (to show digitised curves)
    if makePlot:
        
        fVals = numpy.arange(0.0,8.2,0.05)
        caseA = walkersFunc(fVals)
        caseB = joggersFunc(fVals)
        
        fig, ax = plt.subplots()
        
        ax.plot(fVals,caseA,label='A')
        ax.plot(fVals,caseB,label='B')
        
        ax.axvline(fv,color='r',alpha=0.3)
        ax.axhline(k_fv,color='r',alpha=0.3)
        
        ax.legend()
        
        ax.set_xlim([0,8.0]) # per Fig.NA.8
        ax.set_ylim([0,1.4]) # per Fig.NA.8
        
        ax.set_title("Combined population and harmonic factor k($f_{v}$)\n" + 
                     "per Figure NA.8, UK NA to BS EN 1992-1:2003")
        ax.set_xlabel("Mode frequency $f_{v}$, Hz")
        ax.set_ylabel("k($f_{v}$)")
    
    return k_fv
        
        
# ********************** TEST ROUTINE ****************************************

if __name__ == "__main__":
    
    testRoutine = 2
    
    if testRoutine ==1:
        
        kv = UKNA_BSEN1991_2_Figure_NA_8(fv=2.5,
                                         analysis_type="joggers",
                                         makePlot=True)
        
    if testRoutine ==2:
        
        fn = 2.3
        Tn = 1/fn
        
        loading_obj = UKNA_BSEN1991_2_walkers_joggers_loading(fv=fn,
                                                      analysis_type="joggers")
        loading_obj.PrintDetails()
        
        tVals = numpy.arange(0,5,0.01)
        loadVals=[]
        for t in tVals:
            loadVals.append(loading_obj.loadVals(t))
            
        fig,ax = plt.subplots()
        
        ax.plot(tVals,loadVals)
        
        for T in [Tn,2*Tn,3*Tn,10*Tn]:
            ax.axvline(T,color='r',alpha=0.3)
        
