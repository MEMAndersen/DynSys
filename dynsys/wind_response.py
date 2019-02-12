# -*- coding: utf-8 -*-
"""
Module to provide classes and methods for calculation response of systems 
to wind-induced loading

@author: RIHY
"""

from common import check_class
from modalsys import ModalSys
from wind_env import WindEnv

#%%

class Buffeting():
    """
    Class to implement gust-buffeting response analysis
    """
    
    def __init__(self,sys,wind_env):
        """
        Defines analysis
        
        ***
        Required:
            
        * `sys`, instance of `ModalSys` class. Note: a modal system with 
          multiple sub-systems (e.g. TMDs) appended is permitted
          
        * `wind_env`, instance of `WindEnv` class (or derived classes)
        """
        
        print("**** UNDER DEVELOPMENT ****")
        
        self.sys = sys
        self.wind_env = wind_env
        
        # Check modal system has a mesh associated with it
        if not sys.has_mesh():
            raise ValueError("`sys` must have an associated mesh\n" + 
                             "(This is used to implement integration, " + 
                             "store and handle results etc.)")
            

        
    # -------
    @property
    def sys(self):
        """
        Instance of `ModalSys` class used to define system to be analysed
        """
        return self._sys
    
    @sys.setter
    def sys(self,obj):
        check_class(obj,ModalSys)
        self._sys = obj
        
    # -------
    @property
    def wind_env(self):
        """
        Instance of `WindEnv` class, used to define wind environment
        """
        return self._wind_env
    
    @wind_env.setter
    def wind_env(self,obj):
        check_class(obj,WindEnv)
        self._wind_env = obj
    
    # -------
            
    def plot(self,verbose=True,custom_settings={}):
        """
        Produces plots required to document buffeting analysis
        """
        
        if verbose: print("Producing summary plots to document analysis...")
        
        # Define default plot settings
        settings = {}
        settings['plot_wind_profiles'] = True
        
        # Overide with any settings passed in
        settings = {**settings, **custom_settings}
        
        fig_list = []
        if settings['plot_wind_profiles']:
            fig_list.append(self.wind_env.plot_profiles())
            
        if len(fig_list)==0:
            if verbose: print("(No plots produced)")
            
        return fig_list
        
        
    def run(self):
        """
        Runs analysis
        """
        pass

#%%
        
class VIV():
    """
    Class to implement vortex-induced vibrations response analysis
    """
    pass