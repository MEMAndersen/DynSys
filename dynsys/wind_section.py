# -*- coding: utf-8 -*-
"""
Classes used to define wind cross-sections
(e.g. to calculate drag, lift loading)

@author: RIHY
"""

import numpy
import scipy
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from common import check_class, check_float_or_callable, func_or_float
from common import check_class_name

from numpy import pi

import pandas as pd

#%%
# ------------------- MODULE-LEVEL VARIABLES ----------------------------------

v_air = 15e-6
"""
Kinematic viscocity of air [m2/s]
"""


#%%
# ------------------- CLASSES ------------------------------------------------

class WindSection_Resistances():
    """
    Generic cross-section, where (drag, lift, moment) _resistances_ are 
    defined explicitly
    
    _Note: this is intended to be an abstract class. Objects cannot be 
    instantiated. Use derived classes instead._
    """
    
    def __init__(self,R_D, R_L=0.0, R_M=0.0, name=None):
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
        
        # Prevent direct instatiation of this class
        if type(self) == WindSection_Resistances:
            raise Exception("<WindSection_Resistances> must be subclassed.")
        
        # Define functions for (drag, lift, moment) resistance as attributes
        check_float_or_callable(R_D,'R_D')
        check_float_or_callable(R_L,'R_L')
        check_float_or_callable(R_M,'R_M')
        
        self.R_D = R_D
        self.R_L = R_L
        self.R_M = R_M
        
        self.name = name
        
        
        
    @property
    def R_D(self):
        """
        Wind resistance for calculation of drag load, i.e. load in the 
        direction of the mean wind vector, projected into the section plane
        """
        return self.calc_R_D
    
    @R_D.setter
    def R_D(self,val):
        self._R_D = val
    
    @property
    def R_L(self):
        """
        Wind resistance for calculation of lift load, i.e. load perpendicular 
        to the mean wind vector, but in the section plane
        """
        return self.calc_R_L
    
    @R_L.setter
    def R_L(self,val):
        self._R_L = val
    
    @property
    def R_M(self):
        """
        Wind resistance for calculation of moment load. Note: when expressed 
        as a vector moment is about the (-x) section axis. Refer image included 
        in `__init__()` doctring
        """
        return self.calc_R_M
    
    @R_M.setter
    def R_M(self,val):
        self._R_M = val


    def calc_R_D(self,alpha,**kwargs):
        """
        Calculates wind resistance for drag loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        return func_or_float(self._R_D,alpha)
        
        
    def calc_R_D_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluate derivative of drag resistance with respect to angle of attack 
        at the specified angle `alpha`
        """
        return _derivative_func(self._R_D,alpha,order=order)
        
        
    def calc_R_L(self,alpha,**kwargs):
        """
        Calculates wind resistance for lift loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        return func_or_float(self._R_L,alpha)
        
        
    def calc_R_L_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluate derivative of lift resistance with respect to angle of attack 
        at the specified angle `alpha`
        """
        return _derivative_func(self._R_L,alpha,order=order)
        
    
    def calc_R_M(self,alpha,**kwargs):
        """
        Calculates wind resistance for moment loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        return func_or_float(self._R_M,alpha)
        
        
    def calc_R_M_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluate derivative of lift resistance with respect to angle of attack 
        at the specified angle `alpha`
        """
        return _derivative_func(self._R_M,alpha,order=order)
    
    
    def calc_denHertog(self,alpha,find_roots:bool=True,**kwargs):
        r"""
        Evaluates the 'den Hertog stability criterion' at angles of attack 
        `alpha`, i.e. the following function: 
        
        $$ H(\alpha) = dR_L/d\alpha|_{\alpha} + R_D(\alpha) $$
        
        ***
        **Required:**
            
        * `alpha`, array of angle of attack values at which to evaluate H
        
        ***
        **Optional**
        
        * `find_roots`: if True, method will seek all roots (where H=0) and
          return list to define regions of instablity (where H<0 between roots)

        ***
        **Returns:**
            
        If `find_roots` is False, returns array of H(alpha) values
        
        Otherwise returns tuple consisting of the following:
            
        * Array of H(alpha) values
        
        * List of roots, i.e. alpha values where H(alpha)=0
        
        * List of 2-item lists, defining bounding alpha values for regions 
          where H(alpha) < 0
          
        """
        
        def H_func(alpha):
            
            RL_derivative = self.calc_R_L_derivative(alpha,order=1,**kwargs)
            RD = self.calc_R_D(alpha,**kwargs)
            return RL_derivative + RD
        
        H = H_func(alpha)
        
        if find_roots:
            
            # Detect zero-crossings first
            brackets = []
            for i in range(len(H)-1):
                
                H1 = H[i+1]
                H0 = H[i]
                
                if numpy.sign(H1)!=numpy.sign(H0):
                    # Zero-crossing detected
                    brackets.append([alpha[i],alpha[i+1]])
                    
            # Use bisection method to obtain precise zero-crossings
            roots = []
            for (a,b) in brackets:
                roots.append(scipy.optimize.bisect(H_func,a,b))
                
            # Identify instability regions (with H<0)
            regions = []
            
            if len(roots)>0:
                
                # Wrap list around
                roots = [roots[-1]-2*pi] + roots + [roots[0]+2*pi]
                
                # Identify regions where H<0
                for i in range(len(roots)-1):
                    
                    H_mid = H_func(0.5*(roots[i]+roots[i+1]))
                    
                    if H_mid < 0:
                        regions.append([roots[i],roots[i+1]])
            
            return H, roots, regions
            
        else:
            return H
    
    
    def plot_denHertog(self,ax=None,**kwargs):
        """
        Plots the den Hertog parameter H for all angles of attack
        
        Where unstable regions (H<0) exist, these are annotated via shading
        """
        
        alpha = numpy.linspace(0,2*pi,61)
        H, roots, regions = self.calc_denHertog(alpha,find_roots=True,**kwargs)
        
        alpha = numpy.rad2deg(alpha)
        roots = numpy.rad2deg(roots)
        regions = numpy.rad2deg(regions)
        
        if ax is None:
            fig,ax = plt.subplots()
            fig.set_size_inches((8,4))
                        
        else:
            fig = ax.get_figure()
            
        ax.plot(alpha,H)
        ax.set_xlim([0,360])
        ax.axhline(0.0,color='k',alpha=0.5)
        
        [ax.axvline(r,color='r') for r in roots]
        [ax.axvspan(a,b,color='r', alpha=0.1) for (a,b) in regions]
         
        ax.set_title("Den Hertog parameter $H$\n" + 
                     r"$ H = dR_L/d\alpha + R_D $")
            
        return fig
    
    
    def plot_resistances(self,U=20.0, alpha=None):
        """
        Plots variation in wind resistances with angle of attack 
        
        ***
        Optional:
            
        * `U`, mean wind speed used for resistances calculation (note: for some 
          sections, e.g. circles, wind resistance is Re-dependent)
        
        * `alpha`, specific angle of attack values to evaluate resistance at. 
          By default `alpha` will be plotted in range [0, 2*pi]
          
        """
        
        if alpha is None: alpha = numpy.linspace(0,2*pi,61)
        alpha_deg = numpy.rad2deg(alpha)
        
        R_D = [self.calc_R_D(a,U=U) for a in alpha]
        R_D_deriv = [self.calc_R_D_derivative(a,U=U) for a in alpha]
        
        R_L = [self.calc_R_L(a,U=U) for a in alpha]
        R_L_deriv = [self.calc_R_L_derivative(a,U=U) for a in alpha]
        
        R_M = [self.calc_R_M(a,U=U) for a in alpha]
        R_M_deriv = [self.calc_R_M_derivative(a,U=U) for a in alpha]
        
        fig, axarr = plt.subplots(3,2,sharex=True)
        fig.set_size_inches((12,8))
        fig.subplots_adjust(wspace=0.3)
        
        ax = axarr[0,0]
        ax.plot(alpha_deg,R_D)
        ax.set_ylabel("$R_D$")
        
        ax = axarr[0,1]
        ax.plot(alpha_deg,R_D_deriv)
        ax.set_ylabel(r"$dR_D/d\alpha}$")
        
        ax = axarr[1,0]
        ax.plot(alpha_deg,R_L)
        ax.set_ylabel("$R_L$")
        
        ax = axarr[1,1]
        ax.plot(alpha_deg,R_L_deriv)
        ax.set_ylabel(r"$dR_L/d\alpha$")
        
        ax = axarr[2,0]
        ax.plot(alpha_deg,R_M)
        ax.set_ylabel("$R_M$")
        
        ax = axarr[2,1]
        ax.plot(alpha_deg,R_M_deriv)
        ax.set_ylabel(r"$dR_M/d\alpha$")
        
        ax = axarr[2,0]
        ax.set_xlim([alpha_deg[0],alpha_deg[-1]])
        ax.set_xlabel(r"Angle of attack $\alpha$ (degrees)")
        ax.set_ylabel("$R_M$")
        
        if ax.get_xlim()==(0,360):
            ax.set_xticks(numpy.arange(0,361,60))
        
        fig.suptitle("Variation of wind-aligned wind resistances " + 
                     "with angle of attack")


#%%        
        
class WindSection_Circular(WindSection_Resistances):
    """
    Derived class to implement WindSection for circular section
    """
    
    def __init__(self,d,C_D=None,k=None,**kwargs):
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
                
        Any additional keyword arguments are passed to `__init__()` method of 
        parent class
        """
        
        self.d = d
        self.C_D = C_D
        
        if C_D is None and k is None:
            raise ValueError("Given `C_d` is None, `k` must be provided")
            
        self.k = k
        
        super().__init__(R_D=self.calc_R_D,
                         R_L=self.calc_R_L,
                         R_M=self.calc_R_M,
                         **kwargs)
        
        
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
    def C_D(self):
        """
        Drag coefficient
        """
        return self._C_D
    
    @C_D.setter
    def C_D(self,val):
        
        if (not isinstance(val,float)) and (val is not None):
            raise ValueError("`C_d` must be either float or None\n" + 
                             "C_d : {0}".format(val))
            
        self._C_D = val
        
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
    
    def calc_R_D(self,alpha,U=20.0,**kwargs):
        """
        Calculates wind resistance for drag loading, given angle of attack
        `alpha` and mean wind speed `U`
        
        If `C_d` has not been defined for section explicitly, appropriate drag 
        coefficient will be calculated given `U` and surface roughness `k` for 
        section
        """
        C_D = self.C_D
        d = self.d
        
        if C_D is None:
            # Calculate C_d given Reynold number and surface roughness k
            C_D = self.calc_C_D(U)
            
        return d*C_D
    
    
    def calc_R_D_derivative(self,alpha,**kwargs):
        return numpy.zeros_like(alpha)
    
    
    def calc_R_L(self,alpha,**kwargs):
        """
        Calculates wind resistance for lift loading
        
        _Always equals 0.0 for circles, due to symmetry_
        """
        return numpy.zeros_like(alpha)
    
    
    def calc_R_L_derivative(self,alpha,**kwargs):
        return numpy.zeros_like(alpha)
    
    
    def calc_R_M(self,alpha,**kwargs):
        """
        Calculates wind resistance for moment loading
        
        _Always equals 0.0 for circles, due to symmetry_
        """
        return numpy.zeros_like(alpha)
    
    
    def calc_R_M_derivative(self,alpha,**kwargs):
        return numpy.zeros_like(alpha)
    
        
    def calc_C_D(self,U):
        """
        Calculates drag coefficient in accordance with Fig 7.28 BS EN 
        1991-1-4:2005
        """
        
        if not isinstance(U,float):
            raise ValueError("`U` must be a float")
        
        k = self.k
        b = self.d
        k_b = max(k/b, 1e-5)
        
        Re = calc_Re(U,b)
        
        # Curve 1
        c_f0_1 = 0.11 / (Re/1e6)**1.4
        c_f0_1 = min(max(c_f0_1, 0.4),1.2)
        
        # Curve 2
        c_f0_2 = 1.2 + (0.18*numpy.log10(10*k_b)) / (1+0.4*numpy.log10(Re/1e6))
    
        # Return maximum of curve 1 and 2
        return max(c_f0_1,c_f0_2)
    
#%%
        
class WindSection_Custom(WindSection_Resistances):
    """
    Class to define custom wind section by explicitly defining variation of 
    (drag, lift, moment) resistances with angle of attack
    """
    
    def __init__(self,alpha:list,R_D:list,R_L:list=None,R_M:list=None):
        """
        Initialise custom wind section 
        
        ***
        Required:
        
        * `alpha`, list-like, gives specific values of angle of attack 
          values [radians]
          
        * `R_D`, list-like, gives specific values of drag wind resistance [m] 
          at angles of attack corresponding to `alpha`
          
        ***
        Optional:
            
        * `R_L`, list-like, gives specific values of lift wind resistance [m]
          at angles of attack corresponding to `alpha`. If None (default) then 
          zero values assumed for all `alpha`
          
        * `R_M`, list-like, gives specific values of moment wind resistance 
          [m2] at angles of attack corresponding to `alpha`. If None (default) 
          then zero values assumed for all `alpha`
    
        """
        
        splines = _convert_to_cubic_splines(alpha,[R_D,R_L,R_M])
            
        # WindSection init method
        super().__init__(R_D=splines[0], R_L=splines[1], R_M=splines[2])
        
    
#%%
        
class WindSection_Coeffs(WindSection_Custom):
    """
    Class to define custom wind section by explicitly defining variation of 
    (drag, lift, moment) **coefficients** with angle of attack
    """
    
    def __init__(self,alpha:list,C_D:list,C_L:list=None,C_M:list=None):
        """
        Initialise custom wind section , defined in terms of dimensionless 
        (drag, lift, moment) **coefficients**
        
        ***
        Required:
        
        * `alpha`, list-like, gives specific values of angle of attack 
          values [radians]
          
        * `C_D`, list-like, gives specific values of (dimensionless) drag 
          coefficient at angles of attack corresponding to `alpha`
          
        ***
        Optional:
            
        * `C_L`, list-like, gives specific values of (dimensionless) lift 
          coefficient at angles of attack corresponding to `alpha`. 
          If None (default) then zero values assumed for all `alpha`
          
        * `C_M`, list-like, gives specific values of (dimensionless) moment 
          coefficient at angles of attack corresponding to `alpha`. 
          If None (default) then zero values assumed for all `alpha`
    
        """
        
        splines = _convert_to_cubic_splines(alpha,[C_D,C_L,C_M])
        self._C_D, self._C_L, self._C_M = splines

    # ------- PROPERTIES ---------
    
    @property
    def C_D(self):
        """
        Drag coefficient function
        """
        return self._C_D
    
    @property
    def C_L(self):
        """
        Lift coefficient function
        """
        return self._C_L
    
    @property
    def C_M(self):
        """
        Moment coefficient function
        """
        return self._C_M
    
    # ---------------------
    
    def calc_C_D(self,alpha,**kwargs):
        """
        Evaluates drag coefficient at angles `alpha`
        """
        return self.C_D(alpha)
    
    
    def calc_C_D_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluates derivative of drag coefficient with regards to angle of 
        attack, at angles `alpha`
        """
        return _derivative_func(self.C_D,alpha,order=order)
    
    
    def calc_C_L(self,alpha,**kwargs):
        """
        Evaluates lift coefficient at angles `alpha`
        """
        return self.C_L(alpha)
    
    
    def calc_C_L_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluates derivative of lift coefficient with regards to angle of 
        attack, at angles `alpha`
        """
        return _derivative_func(self.C_L,alpha,order=order)
    
    
    def calc_C_M(self,alpha,**kwargs):
        """
        Evaluates moment coefficient at angles `alpha`
        """
        return self.C_M(alpha)
    
    
    def calc_C_M_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluates derivative of moment coefficient with regards to angle of 
        attack, at angles `alpha`
        """
        return _derivative_func(self.C_M,alpha,order=order)
    
    
    def calc_force_coeffs(self,alpha,alignment='wind'):
        """
        Evaluates (drag, lift) force coefficients at angles `alpha`
        
        If `alignment='wind'` (default), force coefficients are returned in 
        wind-aligned axes. Otherwise if 'body', force coefficients are returned 
        with regards to body axes
        
        _Refer diagram in `WindSection_Resistances.__init__()` method for a 
        reminder of how (drag, lift) coefficients and angle of attack is 
        defined with regards to body axes 'y' and 'z'_
        """
        
        C_D = self.calc_C_D(alpha)
        C_L = self.calc_C_L(alpha)
        
        if alignment == 'wind':
            
            return C_D, C_L
            
        elif alignment == 'body':
            
            C_y = C_D * numpy.cos(alpha) - C_L *numpy.sin(alpha)
            C_z = C_L * numpy.cos(alpha) + C_D *numpy.sin(alpha)
            return C_y, C_z
            
            
        else:
            raise ValueError("Unexpected `alignment` parameter")
            
    
        
    # --------------------
    
    def plot_coefficients(self,alpha=None):
        """
        Plots variation in (drag,list,moment) coefficients with angle of attack 
        
        ***
        Optional:
            
        * `alpha`, specific angle of attack values to evaluate resistance at. 
          By default `alpha` will be plotted in range [0, 2*pi]
          
        """
        
        if alpha is None: alpha = numpy.linspace(0,2*pi,61)
        alpha_deg = numpy.rad2deg(alpha)
        
        C_D = [self.calc_C_D(a) for a in alpha]
        C_D_deriv = [self.calc_C_D_derivative(a) for a in alpha]
        
        C_L = [self.calc_C_L(a) for a in alpha]
        C_L_deriv = [self.calc_C_L_derivative(a) for a in alpha]
        
        C_M = [self.calc_C_M(a) for a in alpha]
        C_M_deriv = [self.calc_C_M_derivative(a) for a in alpha]
        
        fig, axarr = plt.subplots(3,2,sharex=True)
        fig.set_size_inches((12,8))
        fig.subplots_adjust(wspace=0.3)
        
        ax = axarr[0,0]
        ax.plot(alpha_deg,C_D)
        ax.set_ylabel("$C_D$")
        
        ax = axarr[0,1]
        ax.plot(alpha_deg,C_D_deriv)
        ax.set_ylabel(r"$dC_D/d\alpha}$")
        
        ax = axarr[1,0]
        ax.plot(alpha_deg,C_L)
        ax.set_ylabel("$C_L$")
        
        ax = axarr[1,1]
        ax.plot(alpha_deg,C_L_deriv)
        ax.set_ylabel(r"$dC_L/d\alpha$")
        
        ax = axarr[2,0]
        ax.plot(alpha_deg,C_M)
        ax.set_ylabel("$C_M$")
        
        ax = axarr[2,1]
        ax.plot(alpha_deg,C_M_deriv)
        ax.set_ylabel(r"$dC_M/d\alpha$")
        
        ax = axarr[2,0]
        ax.set_xlim([alpha_deg[0],alpha_deg[-1]])
        ax.set_xlabel(r"Angle of attack $\alpha$ (degrees)")
        
        if ax.get_xlim()==(0,360):
            ax.set_xticks(numpy.arange(0,361,60))
        
        fig.suptitle("Variation of (drag, lift, moment) coefficients " + 
                     "with angle of attack")
    
    
#%%
    
class WindSection_Similar(WindSection_Resistances):
    """
    Defines wind cross-section via section dimensions and instance of 
    `WindSection_Coeffs()` class, which is used to define how (drag, lift, 
    moment) force coefficients vary with angle of attack
    """
    
    def __init__(self,b:float,h:float,wind_section_coeffs:WindSection_Coeffs):
        """
        Defines wind cross-section via section dimensions and instance of 
        `WindSection_Coeffs()` class, which is used to define how (drag, lift, 
        moment) force coefficients vary with angle of attack
        
        ***
        Required:
        
        * `b`, float defining breadth of section in local y direction
        
        * `h`, float defining thickness of section in local z direction
        
        * `wind_section_coeffs`, instance of `WindSection_Coeffs` class
        
        Refer docstrings for instance variables for more complete definition of 
        `b` and `h`, in terms of their usage in calculating wind resistances.
        """
        
        check_class(wind_section_coeffs,WindSection_Coeffs)
        check_class(b,float)
        check_class(h,float)
        
        self.wind_section_coeffs = wind_section_coeffs
        """
        Instance of `WindSection_Coeffs()` class, which defines how (drag, 
        lift, moment) coefficients vary with angle of attack
        """
        
        self.b = b
        """
        Characteristic breadth of section in local y direction, such that 
        (lift,moment) wind resistances relate to (lift,moment) coefficients as 
        follows:
        $$ R_L = b C_L  $$
        $$ R_M = b^2 C_M $$
        """
        
        self.h = h
        """
        Characteristic height of section in local z direction, such that wind 
        drag resistance relates to the drag coefficient as follows:
        $$ R_D = h C_D $$
        """
        
        super().__init__(R_D=self.calc_R_D,
                         R_L=self.calc_R_L,
                         R_M=self.calc_R_M)
    
    
    def calc_R_D(self,alpha,**kwargs):
        h = self.h
        ws = self.wind_section_coeffs
        return h * ws.calc_C_D(alpha,**kwargs)
    
    
    def calc_R_D_derivative(self,alpha,order=1,**kwargs):
        h = self.h
        ws = self.wind_section_coeffs
        return h * ws.calc_C_D_derivative(alpha,order=order,**kwargs)
    
    
    def calc_R_L(self,alpha,**kwargs):
        b = self.b
        ws = self.wind_section_coeffs
        return b * ws.calc_C_L(alpha,**kwargs)
    
    
    def calc_R_L_derivative(self,alpha,order=1,**kwargs):
        b = self.b
        ws = self.wind_section_coeffs
        return b * ws.calc_C_D_derivative(alpha,order=order,**kwargs)
    
        
    def calc_R_M(self,alpha,**kwargs):
        b = self.b
        ws = self.wind_section_coeffs
        return b**2 * ws.calc_C_M(alpha,**kwargs)
    
    
    def calc_R_M_derivative(self,alpha,order=1,**kwargs):
        b = self.b
        ws = self.wind_section_coeffs
        return b**2 * ws.calc_C_M_derivative(alpha,order=order,**kwargs)

    
    
        
#%% ------------------ PUBLIC FUNCTIONS -------------------
        
def calc_Re(U,d,v=v_air):
    """
    Calculates Reynold number given wind speed `U` and characterstic 
    diameter `d`.
    
    `v` can be used to specify kinematic viscocity. Default value relates 
    to air
    """
    return U*d/v


def calc_U_from_Re(d,Re,v=v_air):
    """
    Calculates mean wind speed, given characteristic diameter 'd' and Reynolds 
    number `Re`
    
    `v` can be used to specify kinematic viscocity. Default value relates 
    to air
    """
    return Re * v / d


def read_wind_sections(fname,verbose=True):
    """
    Reads section definitions from .csv file `fname`
    
    Expected input file to have the following columns:
        
    Name | Type | Dim1 | Dim2 |
    ---------------------------
         |      |      |      |
         
    Usage of columns 'Dim1', 'Dim2' depends on 'Type', i.e. the type of wind 
    section being defined:
        
    * `Type=Cylinder`:
        
        * `Dim1` defines outer diameter [m]
        
        * `Dim2` defines surface roughness [m]
        
    Other `Type` values not (yet) accepted.

    ***
    Returns:
        
    Dict of WindSection objects
    """
    
    if verbose:
        print("Reading wind section definitions from file...\n" + 
              "\tDefinitions file:\t'%s'" % fname)
    
    df = pd.read_csv(fname,index_col=0)
    
    ws_dict = {}
    
    for section_name, section_data in df.iterrows():
        
        section_type = section_data['Type']
        
        if section_type=='Circular':
            
            d = section_data['Dim1'] # Dim1 specifies diameter of cylinder
            k = section_data['Dim2'] # Dim2 specifies surface roughness of cylinder
            ws_obj = WindSection_Circular(name=section_name, d=d, k=k)
            
        
        else:
            print("Section '%s' is of unsupported 'Type': %s" 
                  % (section_name, section_type))
            break
            
        ws_dict[section_name] = ws_obj
    
    if verbose:
        print("\tNo sections defined:\t%d" % len(ws_dict))
    
    return ws_dict


def assign_wind_sections(fname:str, mesh_obj:object, ws_dict:dict, 
                         verbose=True):
    """
    Assign wind sections, as supplied via `ws_dict` argument, to line elements 
    with mesh `mesh_obj`, using assignment data as supplied via `fname` .csv 
    file
    """
    
    check_class_name(mesh_obj,'Mesh')
    
    if verbose:
        print("Assigning wind sections to mesh...\n" + 
              "\tMesh:\t\t\t'%s'\n" % mesh_obj.name + 
              "\tDefinitions file:\t'%s'" % fname)
    
    # Read data from .csv file
    df = pd.read_csv(fname,index_col=0)
    
    for element_name, row in df.iterrows():
        
        # Get element from mesh
        element_obj = mesh_obj.get_object('Element',element_name)
        
        # Get names of wind sections
        ws_name_end1 = row['Section 1']
        ws_name_end2 = row['Section 2']
        
        # Get wind section objects from provided dict
        ws_list = []
        
        # Define function for handling key errors in a nice way
        def get_wind_section(name):
            try:
                obj = ws_dict[name]
            except KeyError as e:
                raise KeyError("Wind section '%s' has not been defined" % e)
                
            return obj
                
        # Append wind section for end 1 / uniform wind section for element
        ws_list.append(get_wind_section(ws_name_end1))
        
        # Append wind section for end 2, if defined
        if not numpy.isnan(ws_name_end2):
            ws_list.append(get_wind_section(ws_name_end2))
        
        # Associate wind sections with element object
        element_obj.define_wind_sections(ws_list)
        

#%% ------------------ PRIVATE FUNCTIONS -------------------

def _convert_to_cubic_splines(x:list,y:list):
    """
    Converts interpolated functions as described by `x`, `y` into a series of 
    cubic spline classes
    
    ***
    Required:
        
    * `x`, list of values at which `y` evaluated
    
    * `y`, list of lists, giving output values at `x`
    """
    
    settings = {'bc_type':'periodic'}
    spline_class = CubicSpline

    # Check array dimensions
    N = len(alpha)
    
    func_list = []
    for i, yi in enumerate(y):
           
        if yi is not None:
            
            if len(yi)!= N:
                raise ValueError("`len(yi)` does not agree with `len(x)`\n" + 
                                 "i=%d, len(yi)=%d, N=%d" % (i, len(yi), N) )
        
            # Define cubic splines through passed-in list data
            yi_func = spline_class(x, yi,**settings)
            
        else:
            yi_func = lambda x : 0.0

        func_list.append(yi_func)
        
    return func_list


def _derivative_func(val,alpha,order=1):
    """
    Private function, used to evaluate derivatives
    (avoids code repetition)
    """
    
    if isinstance(val, float):
        return 0.0 # constant R_L thus derivative = 0.0 for all alpha
    
    elif isinstance(val,CubicSpline):
        
        if order > 2:
            raise ValueError("Cubic spline cannot be used to evaluate 3rd " + 
                             "derivatives or greater")
            
        return val(alpha,order)
    
    else:
        raise ValueError("Unexpected type. Cannot evaluate derivative\n" + 
                         "{0}".format(type(val)))


#%%
# ---------------- TEST ROUTINES ----------------------------------------------

if __name__ == "__main__":
    
    plt.close('all')
    
    test_routine = 2

    if test_routine == 1:
        
        d = 1.5
        circle = WindSection_Circular(d=d,k=1e-4)
        circle.plot_resistances(U=10.0)
        circle.plot_denHertog()
        
    elif test_routine == 2:
        
        # Define angles of attack to define resistances at
        alpha = numpy.linspace(0,2*pi,100)
        
        # Define some arbitrary resistances
        R_D = numpy.sin(alpha)
        R_L = numpy.cos(alpha)**2
        R_M = 0.1*alpha*(2*pi - alpha)*(pi - alpha)
        
        # Define wind section and plot resistances and their derivatives
        section = WindSection_Custom(alpha,R_D,R_L,R_M)
        section.plot_resistances()
        
        section.plot_denHertog()
        
    elif test_routine == 3:
        
        # Define angles of attack to define resistances at
        alpha = numpy.linspace(0,2*pi,100)
        
        # Define some arbitrary coefficients
        CD = numpy.sin(alpha)
        CL = numpy.cos(alpha)**2
        CM = 0.1*alpha*(2*pi - alpha)*(pi - alpha)
        
        # Define wind section and plot resistances and their derivatives
        coeffs_section = WindSection_Coeffs(alpha,CD,CL,CM)
        coeffs_section.plot_coefficients()
        
        # Define multiple similar sections
        nSections = 1000 # to test how fast implementation is!
        b_vals = numpy.linspace(5.0,20.0,nSections)
        h_vals = numpy.linspace(2.0,1.2,nSections)
        
        wind_sections = []
        for (_b, _h) in zip(b_vals,h_vals):
            wind_sections.append(WindSection_Similar(_b,_h,coeffs_section))
            
        # Get wind resistances
        angles = numpy.linspace(-pi,+pi,61)
        RL_vals = numpy.array([x.R_L(angles) for x in wind_sections])
        
        wind_sections = wind_sections[:4] # keep front of list only
        
        # Show how coeffs_section is used by all wind_sections
        print("`wind_sections`:\n{0}\n".format(wind_sections))
        print("`coeffs_section`:\n{0}\n".format(coeffs_section))
        print("`wind_sections.wind_section_coeffs`:\n{0}\n".format(
                [x.wind_section_coeffs for x in wind_sections]))
        print("Note: same object is being used by all sections")
        
        # Show how all the usual methods still work!
        [ws.plot_resistances() for ws in wind_sections]
        
    else:
        
        raise ValueError("Invalid `testroutine` specified")
        pass