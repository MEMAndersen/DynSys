# -*- coding: utf-8 -*-
"""
Classes used to define wind cross-sections
(e.g. to calculate drag, lift loading)

@author: RIHY
"""

import inspect


class WindSection():
    """
    Generic cross-section, where (drag, lift, moment) resistances are 
    defined explicitly
    """
    
    def __init__(self,R_D, R_L=0.0, R_M=0.0):
        """
        Defines general cross-section via functions that define variation in
        wind resistances with angle of attack
        
        **Important note 1**: Resistances are defined in a _wind-aligned_ 
        axis system, which rotates with angle of attack. This is shown in the 
        following figure:
        
        ![axes](../dynsys/img/wind_loading.png)
        
        In this context:
            
        $$ R_{D} = h C_{D} $$
        $$ R_{L} = b C_{L} $$
        $$ R_{M} = b^{2} C_{M} $$
        
        **Important note 2**: Angle of attack to be specified in _radians_. 
        This is important to ensure accurate calculation of deriviatives.
        
        ***
        Required:
            
        * `R_D`, wind resistance in the direction of the wind. Can be _float_ 
          or function in the form `R_D(alpha)`, where `alpha` is the angle of 
          attack
          
        Optional:
            
        * `R_L`, function to define wind resistance in the lift direction, 
          perpendicular to the wind. Function in the form `R_L(alpha)` required
          
        * `R_M`, function to define wind resistance in terms of moment loading.
          Function in the form `R_M(alpha)` required
        
        """
        
        # Evaluate drag at zero angle of attack
        if not isinstance(R_D,float):
            
            # Check function supplied as expected
            if not inspect.isfunction(R_D):
                raise ValueError("`R_D` must be either a float or a function")
            
            if R_D.signature[0]!="alpha":
                raise ValueError("`R_D` must have first argument `alpha`" + 
                                 " denoting the angle of attack (radians)")
            
        
        self.R_D = R_D
        """
        Wind resistance for calculation of drag load, i.e. load in the 
        direction of the mean wind vector, projected into the section plane
        """
        
        if not isinstance(R_L,float):
            
            if not inspect.isfunction(R_L):
                raise ValueError("`R_L` must be either a float or a function")
            
            if R_L.signature[0]!="alpha":
                raise ValueError("`R_L` must have first argument `alpha`" + 
                                 " denoting the angle of attack (radians)")
        
        self.R_L = R_L
        """
        Wind resistance for calculation of lift load, i.e. load perpendicular 
        to the mean wind vector, but in the section plane
        """
        
        if not isinstance(R_M,float):
            
            if not inspect.isfunction(R_M):
                raise ValueError("`R_M` must be either a float or a function")
                
            if R_M.signature[0]!="alpha":
                raise ValueError("`R_M` must have first argument `alpha`" + 
                                 " denoting the angle of attack (radians)")
        
        self.R_M = R_M
        """
        Wind resistance for calculation of moment load. Note: when expressed 
        as a vector moment is about the (-x) section axis. Refer image included 
        in `__init__()` doctring
        """

