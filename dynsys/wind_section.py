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

#%%
# ------------------- MODULE-LEVEL VARIABLES ----------------------------------

v_air = 15e-6
"""
Kinematic viscocity of air [m2/s]
"""


#%%
# ------------------- CLASSES ------------------------------------------------

class WindSection():
    """
    Generic cross-section, where (drag, lift, moment) resistances are 
    defined explicitly
    
    _Note: this is intended to be an abstract class. Objects cannot be 
    instantiated. Use derived classes instead._
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
        
        # Prevent direct instatiation of this class
        if type(self) == WindSection:
            raise Exception("<WindSection> must be subclassed.")
        
        # Evaluate drag at zero angle of attack
        if not isinstance(R_D,float) and not callable(R_D):
                raise ValueError("`R_D` must be either a float or a function")
            
        
        self.R_D = R_D
        """
        Wind resistance for calculation of drag load, i.e. load in the 
        direction of the mean wind vector, projected into the section plane
        """
        
        if not isinstance(R_L,float) and not callable(R_L):
            raise ValueError("`R_L` must be either a float or a callable")
        
        self.R_L = R_L
        """
        Wind resistance for calculation of lift load, i.e. load perpendicular 
        to the mean wind vector, but in the section plane
        """
        
        if not isinstance(R_M,float) and not callable(R_M):
            raise ValueError("`R_M` must be either a float or a callable")
        
        self.R_M = R_M
        """
        Wind resistance for calculation of moment load. Note: when expressed 
        as a vector moment is about the (-x) section axis. Refer image included 
        in `__init__()` doctring
        """


    def calc_R_D(self,alpha,**kwargs):
        """
        Calculates wind resistance for drag loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        return _func_or_float(self.R_D,alpha)
        
        
    def calc_R_D_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluate derivative of drag resistance with respect to angle of attack 
        at the specified angle `alpha`
        """
        return _derivative_func(self.R_D,alpha,order=order)
        
        
    def calc_R_L(self,alpha,**kwargs):
        """
        Calculates wind resistance for lift loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        return _func_or_float(self.R_L,alpha)
        
        
    def calc_R_L_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluate derivative of lift resistance with respect to angle of attack 
        at the specified angle `alpha`
        """
        return _derivative_func(self.R_L,alpha,order=order)
        
    
    def calc_R_M(self,alpha,**kwargs):
        """
        Calculates wind resistance for moment loading, given angle of attack
        `alpha` and mean wind speed `U`
        """
        return _func_or_float(self.R_M,alpha)
        
        
    def calc_R_M_derivative(self,alpha,order=1,**kwargs):
        """
        Evaluate derivative of lift resistance with respect to angle of attack 
        at the specified angle `alpha`
        """
        return _derivative_func(self.R_M,alpha,order=order)
    
    
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
                roots = [roots[-1]-2*numpy.pi] + roots + [roots[0]+2*numpy.pi]
                
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
        
        alpha = numpy.linspace(0,2*numpy.pi,61)
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
        
        if alpha is None: alpha = numpy.linspace(0,2*numpy.pi,61)
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
        
    # --------- RE-DIRECT GETTER METHODS --------------------------------------
    @property
    def R_D(self):
        return self.calc_R_D
    
    @property
    def R_L(self):
        return self.calc_R_L
    
    @property
    def R_M(self):
        return self.calc_R_M
        
    # ----------------- CLASS METHODS -----------------------------------------
    
    def calc_R_D(self,alpha,U=20.0,**kwargs):
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
    
        
    def calc_C_d(self,U):
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
        
class WindSection_Custom(WindSection):
    """
    Class to define custom wind section by explicitly defining variation of 
    drag, lift and moment resistance with angle of attack
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
        
        
        # Check array dimensions
        N = len(alpha)
        
        if len(R_D)!= N:
            raise ValueError("`len(R_D)` must agree with `len(alpha)`")
            
        if R_L is not None and len(R_L)!= N:
            raise ValueError("`len(R_L)` must agree with `len(alpha)`")
        
        if R_M is not None and len(R_M)!= N:
            raise ValueError("`len(R_M)` must agree with `len(alpha)`")
            
        # Check limits
        if alpha[-1] > 2*numpy.pi or alpha[0]<0:
            raise ValueError("`alpha` to be in the range [0,2*pi]")
        
        
        # Define cubic splines through passed-in list data
        
        settings = {'bc_type':'periodic'}   # n.b. must be periodic in alpha
        
        interpolation_class = CubicSpline   # class to use for interpolation
        # note use of cubic spline is advantageous as allows derivatives calc
        
        R_D_func = interpolation_class(alpha,R_D,**settings)
        
        if R_L is not None:
            R_L_func = interpolation_class(alpha,R_L,**settings)
        else:
            R_L_func = lambda x : 0.0
            
        if R_M is not None:
            R_M_func = interpolation_class(alpha,R_M,**settings)
        else:
            R_M_func = lambda x : 0.0
            
        # WindSection init method
        super().__init__(R_D=R_D_func, R_L=R_L_func, R_M=R_M_func)
    
    
        
#%% ------------------ FUNCTIONS -------------------
        
def calc_Re(U,d,v=v_air):
    """
    Calculates Reynold number given wind speed `U` and characterstic 
    diameter `d`.
    
    `v` can be used to specify kinematic viscocity. Default value relates 
    to air
    """
    return U*d/v


def _func_or_float(f,x):
    """
    Evaluates `f` at x. If `f` is float, value is returned
    """
    if isinstance(f, float):
        return f
    else:
        return f(x)


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
        raise ValueError("Unexpected type. Cannot evaluate derivative")


def test_calc_C_d():
    """
    Test routine to demonstrate accuracy of `calc_C_d()` method for circles, 
    by re-creating Fig 7.28, BS EN 1991-1-4
    """
    
    b=0.1 #arbitrary
    
    # List k/b, Re values per figure
    k_b_vals = numpy.array([1e-2,1e-3,1e-4,1e-5,1e-6])
    Re_vals = numpy.logspace(5,7,num=100)
    
    # Convert to inputs required
    k_vals =  k_b_vals * b
    U_vals = Re_vals * v_air / b
    
    # Evaluate c_f0 for each k,U pair
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


def define_cylinders(fname,mesh_obj=None):
    """
    Read file containing definitions of cylindrical wind sections
    
    ***
    Required:
        
    * `fname`, filename string
    
    ***
    Optional:
        
    * `mesh_obj`, instance of `Mesh()` class. If provided, wind sections will 
      be associated with distinct mesh locations
    
    """
    raise NotImplementedError("Not yet implemented!")
    


# ---------------- TEST ROUTINES ----------------------------------------------

if __name__ == "__main__":
    
    plt.close('all')
    
    test_routine = 3
    
    if test_routine == 1:
        
        print("Test routine to check drag factor calculation for circles")
        test_calc_C_d()
        
    elif test_routine == 2:
        
        d = 1.5
        circle = WindSection_Circular(d=d,k=1e-4)
        circle.plot_resistances(U=10.0)
        circle.plot_denHertog()
        
    elif test_routine == 3:
        
        # Define angles of attack to define resistances at
        alpha = numpy.linspace(0,2*numpy.pi,100)
        
        # Define some arbitrary resistances
        R_D = numpy.sin(alpha)
        R_L = numpy.cos(alpha)**2
        R_M = 0.1*alpha*(2*numpy.pi - alpha)*(numpy.pi - alpha)
        
        # Define wind section and plot resistances and their derivatives
        section = WindSection_Custom(alpha,R_D,R_L,R_M)
        section.plot_resistances()#alpha=numpy.linspace(-1.5,7.5,100))
        
        section.plot_denHertog()
        
    else:
        
        raise ValueError("Invalid `testroutine` specified")
        pass