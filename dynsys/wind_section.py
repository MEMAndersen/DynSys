# -*- coding: utf-8 -*-
"""
Classes used to define wind cross-sections
(e.g. to calculate drag, lift loading)

@author: RIHY
"""

import inspect
import numpy
from numpy import log10
import matplotlib.pyplot as plt

v_air = 15e-6
"""
Kinematic viscocity of air [m2/s]
"""

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
            
        
        self._R_D = R_D
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
        
        self._R_L = R_L
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
        
        self._R_M = R_M
        """
        Wind resistance for calculation of moment load. Note: when expressed 
        as a vector moment is about the (-x) section axis. Refer image included 
        in `__init__()` doctring
        """


    def calc_R_D(self,alpha,U,**kwargs):
        """
        Calculates wind resistance for drag loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        if isinstance(self._R_D, float):
            return self._R_D
        
        else:
            return self._R_D(alpha,U)
        
        
    def calc_R_L(self,alpha,U,**kwargs):
        """
        Calculates wind resistance for lift loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        if isinstance(self._R_L, float):
            return self._R_L
        
        else:
            return self._R_L(alpha,U)
        
    
    def calc_R_M(self,alpha,U,**kwargs):
        """
        Calculates wind resistance for moment loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        if isinstance(self._R_M, float):
            return self._R_M
        
        else:
            return self._R_M(alpha,U)



class WindSection_Circular(WindSection):
    """
    Derived class to implement WindSection for circular section
    """
    
    def __init__(self,d,C_d=None,k=None):
        """
        Defines wind resistances for a circular section
        
        ***
        Required:
            
        * `d`, outer diameter of section [m]
        
        ***
        Optional:
            
        * `C_d`, drag coefficient to be used in conjunction with `d`. If None 
          appropriate drag coefficient will be determined based on mean wind 
          speed and surface roughness `k` (see below)
        
        * `k`, surface roughness [m]. Only required if `C_d`=None        
                
        """
        
        self.d = d
        self.C_d = C_d
        
        if C_d is None and k is None:
            raise ValueError("Given `C_d` is None, `k` must be provided")
            
        self.k = k
        
    # ------------------ PROPERTIES -------------------------------------------
    # ----------
    @property
    def d(self):
        """
        Outer diameter of section [m]
        """
        return self._d
    
    @d.setter
    def d(self,val):
        self._d = val
        
    # ----------
    @property
    def C_d(self):
        """
        Drag coefficient
        """
        return self._C_d
    
    @C_d.setter
    def C_d(self,val):
        
        if (not isinstance(val,float)) and (val is not None):
            raise ValueError("`C_d` must be either float or None\n" + 
                             "C_d : {0}".format(val))
            
        self._C_d = val
        
    # ---------
    @property
    def k(self):
        """
        Surface roughness [m]
        """
        return self._k
    
    @k.setter
    def k(self,val):
        
        if not isinstance(val,float):
            raise ValueError("`k` must be a float")
            
        self._k = val
        
    # ----------------- CLASS METHODS -----------------------------------------
    
    def calc_R_D(self,alpha,U,**kwargs):
        """
        Calculates wind resistance for drag loading, given angle of attack
        `alpha` and mean wind speed `U`
        
        If `C_d` has not been defined for section explicitly, appropriate drag 
        coefficient will be calculated given `U` and surface roughness `k` for 
        section
        """
        C_d = self.C_d
        d = self.d
        
        if C_d is None:
            # Calculate C_d given Reynold number and surface roughness k
            C_d = self.calc_C_d(U)
            
        return d*C_d
    
        
    def calc_C_d(self,U):
        """
        Calculates drag coefficient in accordance with Fig 7.28 BS EN 
        1991-1-4:2005
        """
        
        if not isinstance(U,float):
            raise ValueError("`U` must be a float")
        
        k_b = max(self.k/self.d, 1e-5)
        
        Re = calc_Re(U,b)
        
        # Curve 1
        c_f0_1 = 0.11 / (Re/1e6)**1.4
        c_f0_1 = min(max(c_f0_1, 0.4),1.2)
        
        # Curve 2
        c_f0_2 = 1.2 + (0.18 * log10(10*k_b)) / (1 + 0.4 * log10(Re/1e6))
    
        # Return maximum of curve 1 and 2
        return max(c_f0_1,c_f0_2)
        
        
        
def calc_Re(U,d,v=v_air):
    """
    Calculates Reynold number given wind speed `U` and characterstic 
    diameter `d`.
    
    `v` can be used to specify kinematic viscocity. Default value relates 
    to air
    """
    return U*d/v



# ---------------- TEST ROUTINES ----------------------------------------------

if __name__ == "__main__":
    
    plt.close('all')
    
    testroutine = 1
    
    if testroutine == 1:
        
        print("*** Test routine to re-create Fig 7.28, BS EN 1991-1-4 ***")
        
        b = 0.1
        
        k_b_vals = numpy.array([1e-2,1e-3,1e-4,1e-5,1e-6])
        k_vals =  k_b_vals * b
        Re_vals = numpy.logspace(5,7,num=100)
        U_vals = Re_vals * v_air / b
        
        outer_list = []
        
        for k in k_vals:
            
            inner_list = []
            
            for U in U_vals:
                
                obj = WindSection_Circular(d=b,k=k)
                c_f0 = obj.calc_C_d(U)
                inner_list.append(c_f0)
                
            outer_list.append(inner_list)
            
        c_f0 = numpy.array(outer_list)
        
        # Re-create Fig. 7.28 to test calc_C_d() method
        fig, ax = plt.subplots()
        
        h = ax.plot(Re_vals,c_f0.T)
        h[-1].set_linestyle('--')
        
        ax.set_xscale('log')
        
        ax.legend(h,["%.0e" % x for x in k_b_vals],title="$k/b$")
        
        ax.set_ylim([0,1.4]) # per Fig 7.28
        ax.set_xlim([Re_vals[0],Re_vals[-1]]) # per Fig 7.28
        
        ax.set_xlabel("$Re$")
        ax.set_ylabel("$c_{f0}$")
        ax.set_title("Drag coefficients for circles\n" + 
                     "according to Fig 7.28, BS EN 1991-1-4:2005")
        
    else:
        
        raise ValueError("Invalid `testroutine` specified")
        pass