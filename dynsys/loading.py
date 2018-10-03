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
        
        
    def plot(self):
        """
        Plots diagram to illustrate load definition
        """
        pass
    

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
        
        
    def plot(self,ax=None, t=0, lead_x=0):
        """
        Plots diagram to illustrate load definition
        
        * `t`, time to plot loads at
        
        * `xpos`, defines position of lead axle
        
        """
        
        ax = self.plot_init(ax=ax)
        self.plot_update(t=t, lead_x=lead_x, ax=ax, update_xlim=True)
        
        return ax
    
    
    def plot_init(self,ax=None):
        """
        Method to initialise plot
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Produce vector plot to visualise loads
        self.plot_artists = ax.quiver([],[],[],[],
                                      pivot='tip',color='b')
        
        return ax
        
    
    def plot_update(self, t=0, lead_x=0, ax=None, update_xlim=False):
        """
        Method to update plot for given time and lead axle position
        """
        
        # Get axle positions
        x = lead_x + self.loadX
        
        # Get load values at time t
        vals = self.loadVals(t)
        
        z = numpy.zeros_like(x)
        
        # Update plot artists
        self.plot_artists.set_offsets(numpy.vstack((x,z)).T)
        self.plot_artists.set_UVC(0,vals)
              
        if update_xlim:
            ax.set_xlim([numpy.min(x),numpy.max(x)])
                
        return self.plot_artists
        
        
# ********************** FUNCTIONS ****************************************
        

        
        
# ********************** TEST ROUTINE ****************************************

if __name__ == "__main__":
    
    pass
