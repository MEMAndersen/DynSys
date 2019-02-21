# -*- coding: utf-8 -*-
"""
Classes and functions used to define 'wind environments'
i.e. the spatial variation of mean and turbulence components of wind speed

@author: RIHY
"""

import scipy
import numpy
import matplotlib.pyplot as plt

from numpy import log as ln

from scipy import spatial
from scipy import interpolate
from scipy import optimize

from mesh import Point
from common import rotate_about_axis, check_class

#%%

fontsize_labels = 8

#%%
class WindEnv():
    """
    Base class used to defines those attributes and methods required to 
    define a 'wind environment' for a given site
    
    _Note: implemented as an abstract class. Objects cannot be instantiated_
    """
    
    def __init__(self,
                 points_list=None,
                 z_min:float=10.0, z_max:float=300.0, nz=100
                 ):
        """
        Base class initialisation function for defining wind enviroments
        
        ***
        Optional:
        
        * `points_list` : list of `Point` instances, defining the points (in 
          3D space) to evaluate wind properties at
            
        * `z_min`, `z_max`; height limits used for basis wind profiles plot. 
          Only used if `points_list` is None
        
        * `nz`, number of height increments, used in conjunction with `z_min` 
          and `z_max`
        
        """
        
        # Prevent direct instatiation of this class
        if type(self) == WindEnv:
            raise Exception("<WindEnv> must be subclassed.")
            
        # Define points to evaluate wind properties at
        if points_list is None:
            z_vals = numpy.linspace(z_min,z_max,nz)
            points_list = [Point(xyz=[0.0,0.0,z]) for z in z_vals]
            
        self._points_list = points_list
        
            
    # -------------------- PROPERTIES ----------------------------------------
    
    @property
    def points(self):
        """
        List of `Point` instances, defining locations (in 3D space) at which 
        wind properties are to be evaluated
        """
        return self._points_list
    
    @points.setter
    def points(self,a):
        
        check_class(a,list)
        self._points_list = a
        
    # ---------
    
    @property
    def mean_speed(self)->list:
        """
        Mean wind speed (m/s), evaluated at each point
        """
        return self._U
    
    @property
    def U(self):
        """
        _Psuedonym for `mean_speed`_
        """
        return self.mean_speed
    
    # --------
    
    @property
    def i_u(self)->list:
        """
        Along-wind turbulence intensity, evaluated at each point
        """
        return self._i_u
    
    @property
    def i_v(self)->list:
        """
        Horizontal across-wind turbulence intensity
        """
        return self._i_v
    
    @property
    def i_w(self)->list:
        """
        Vertical across-wind turbulence intensity
        """
        return self._i_w
    
    # ------------------------
    
    @property
    def sigma_u(self)->list:
        """
        Along-wind RMS turbulence (m/s)
        """
        return self._sigma_u
    
    @property
    def sigma_v(self)->list:
        """
        Horizontal across-wind RMS turbulence (m/s)
        """
        return self._sigma_v
    
    @property
    def sigma_w(self)->list:
        """
        Vertical across-wind RMS turbulence (m/s)
        """
        return self._sigma_w
    
    # ------------------------
    
    @property
    def xLu(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Along-wind component (u)
        
        * Along-wind direction (x)
        """        
        return self._xLu
    
    @property
    def yLu(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Along-wind component (u)
        
        * Horizontal across-wind direction (y)
        """
        return self._yLu
    
    @property
    def zLu(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Along-wind component (u)
        
        * Vertical across-wind direction (z)
        """ 
        return self._zLu
    
    # ----
    
    @property
    def xLv(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Horizontal across-wind component (v)
        
        * Along-wind direction (x)
        """        
        return self._xLv
    
    @property
    def yLv(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Horizontal across-wind component (v)
        
        * Horizontal across-wind direction (y)
        """ 
        return self._yLv
    
    @property
    def zLv(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Horizontal across-wind component (v)
        
        * Vertical across-wind direction (z)
        """ 
        return self._zLv
    
    # ----
    
    @property
    def xLw(self)->list:
        """
        Integral turbulence length scale (m):
        
        * Vertical across-wind component (w)
        
        * Along-wind direction (x)
        """        
        return self._xLw
    
    @property
    def yLw(self)->list:
        """
        Integral turbulence length scale (m):
            
        * Vertical across-wind component (w)
        
        * Horizontal across-wind direction (y)
        """ 
        return self._yLw
    
    @property
    def zLw(self)->list:
        """
        Integral turbulence length scale (m):
            
        * Vertical across-wind component (w)
        
        * Vertical across-wind direction (z)
        """ 
        return self._zLw
    
    
    # --------------- METHODS TO BE IMPLEMENTED BY DERIVED CLASSES ------------
    @property
    def calc_mean_speed(self):
        raise NotImplementedError()
        
    @property
    def calc_iu(self):
        raise NotImplementedError()
        
    # -------------- CLASS METHODS --------------------------------------------
    
    def get_x(self):
        """
        Returns 1d array giving x-coordinates of points at which wind 
        properties are evaluated
        """
        return numpy.array([p.x for p in self.points])

    def get_y(self):
        """
        Returns 1d array giving y-coordinates of points at which wind 
        properties are evaluated
        """
        return numpy.array([p.y for p in self.points])

    def get_z(self):
        """
        Returns 1d array giving z-coordinates of points at which wind 
        properties are evaluated
        """
        return numpy.array([p.z for p in self.points])
        
    def get_xyz(self):
        """
        Returns 2d array giving xyz-coordinates of points at which wind 
        properties are evaluated
        """
        return numpy.array([p.xyz for p in self.points])
    
    

#%%
class WindEnv_equilibrium(WindEnv):
    """
    Class to implement _equilibrium_ wind environment
    """
    
    def __init__(self,
                 V_ref:float,
                 z0:float,
                 phi:float,
                 direction:float=0.0,
                 d=None,
                 z_ref:float=10.0,
                 A=-1.0, B=6.0,
                 u_star=None,
                 
                 calc_wind_params=True,
                 **kwargs):
        """
        Defines equilibrium wind environment
        
        ***
        Required:
            
        * `V_ref`  : wind speed (m/s) at height `z_ref`
        
        * `z0`, ground roughness (m). Default value (0.03) corresponds to 
          'Country' terrain
          
        * `phi`, site latitude (degrees)
        
        ***
        Optional:
            
        * `direction`, bearing angle in range [0,360] degrees, defines 
          the direction that wind is coming _from_. Default = 0.0, i.e. wind 
          from North
          
        * `z_ref`, reference height to which `V_ref` relates. Default = 10m
        
        * `d`, displacement height (m). If None (default) value consistent with 
          `z0` is determined
          
        _Additional keyword arguments will be passed to parent `__init__()` 
        method._
        
        ***
        Optional configuration parameters (which will not normally be 
        provided):
            
        * `A`, `B`, parameters required for use in Deaves & Harris log-law 
          formula
          
        * `zg`, initial guess at gradient height (m). _This is determined 
        iteratively_
          
        * `u_star`, friction velocity (m/s), which in essence defines the mean 
          wind profile. If None (default) then `u_star` will be determined 
          based on gradient height (from iterative calculation)
        
        """
        
        # Run parent init method using any additional keyword arguments
        super().__init__(**kwargs)
        
        self.zg = None
        """
        Gradient height (m)
        """
                
        self.z0 = z0
        """
        Ground roughness (m)
        """
        
        if d is None:
            d = self._calc_d()
        self.d = d
        """
        Displacement height (m)
        """
        
        self.z_ref = z_ref
        """
        Reference height (m) at which `V_ref` is quoted
        """
        
        self.V_ref = V_ref
        """
        Mean wind speed (m/s) at height `z_ref`
        """
        
        self.phi = phi
        """
        Site latitude in degrees
        """
        
        self.coriolis_f = None
        """
        Coriolis parameter
        """
    
        self.A = A
        """
        Parameter as used in Deaves and Harris log-law mean wind speed equation
        """
        
        self.B = B
        """
        Parameter as used in Deaves and Harris log-law mean wind speed equation
        """
        
        self.A1 = 2*(ln(B)-A) - (1/6)
        """
        Parameter as used in Deaves and Harris log-law mean wind speed equation
        """
        
        if u_star is None:
            
            # Iteratively calculate gradient height consistent with defined terrain
            zg = self.calc_gradient_height()
            
            # Recalculate u_star now that zg determined
            u_star = self._calc_u_star()
            
        else:
            
            # Calculate zg that is consistent with u_star provided
            zg = self._calc_gradient_height(u_star=u_star)
            
        self.zg = zg
        """
        Gradient height (m)
        """
        
        self.u_star = u_star
        """
        Friction velocity (m/s)
        """
        
        self.mean_direction = direction        
        self.calc_exposure_factor()
        
        # Evaluate wind params at point set currently defined
        if calc_wind_params:
            self.calculate_wind_params()
      
    
    
    # -----
    
    @property
    def mean_direction(self):
        """
        Mean wind direction, expressed as an angle in plan, in degrees 
        clockwise from North
        """
        return self._direction
    
    @property
    def direction(self):
        """
        _Psuedonym for `mean_direction`_
        """
        return self.mean_direction
    
    @mean_direction.setter
    def mean_direction(self,value):
        value = numpy.mod(value,360.0) # force to be in range [0,360]
        self._direction  = value
        
    # ------- GETTER METHODS OVERRIDING PARENT CLASS METHODS ------------------
    
    @property
    def sigma_u(self)->list:
        """
        Along-wind RMS turbulence (m/s)
        
        _Note: defined indirectly via mean speed and turbulence intensity_
        """
        return self.calc_sigma_u()
    
    @property
    def i_v(self)->list:
        """
        Horizonal across-wind turbulence intensity
        
        _Note: defined indirectly via mean speed and RMS turbulence_
        """
        return self.sigma_v / self.U
    
    @property
    def i_w(self)->list:
        """
        Vertical across-wind turbulence intensity
        
        _Note: defined indirectly via mean speed and RMS turbulence_
        """
        return self.sigma_w / self.U
    
    # -------- PROPERTIES -----------------------------------------------------
    
    @property
    def S_exposure(self)->float:
        """
        Ratio of the local mean speed at 10m to the basic mean wind speed in 
        standard terrain
        """
        return self._S_exposure
    
    
    # -------------------- PUBLIC CLASS METHODS -------------------------------
        
    def calculate_wind_params(self):
        """
        Evaluates all wind parameters at current list of points          
        """
        self.calc_mean_speed()
        self.calc_mean_vector()
        self.calc_iu()
        self.calc_RMS_turbulence()
        self.calc_turbulence_length_scales()
        
        self._recalculate = False
               
    
    def print_details(self):
        print("Vref = %.1f\t[m/s]" % self.V_ref)
        print("zref = %.0f\t[m]" % self.z_ref)
        print("z0 = %.3f\t[m]" % self.z0)
        print("zg = %.0f\t[m]" % self.zg)
        print("d = %.1f\t\t[m]" % self.d)
        print("phi = %.1f\t[deg]" % self.phi)
        print("u* = %.2f\t[m/s]" % self.u_star)
        print("f = %.2e" % self.coriolis_f)
        
        
    def plot_profiles(self,
                      params2plot = ['U',
                                     ['i_u','i_v','i_w'],
                                     ['sigma_u','sigma_v','sigma_w'],
                                     ['xLu','yLu','zLu']
                                     ],
                      xlabels = ["Mean wind speed (m/s)",
                                 "Turbulence intensities",
                                 "RMS turbulence (m/s)",
                                 "Along-wind turbulence\nlength scales (m)"]
                      ):
                                                       
        nSubplots = len(params2plot)
        fig,axarr = plt.subplots(1,nSubplots,sharey=True)
        fig.set_size_inches((10,8))
        fig.suptitle("Profiles of wind environment parameters")
        
        y_label = False
               
        for i, (ax,param,xlabel) in enumerate(zip(axarr,params2plot,xlabels)):
                                                
            if i==0:
                y_label=True
            else:
                y_label=False
                
            self.plot_profile(param,ax=ax,x_label=xlabel,y_label=y_label)
            
        return fig
        
        
    def plot_profile(self,param_list:list,
                     ax=None,
                     x_label:str=None,
                     y_label=False,title=False):
        """
        Plots profile / variation of specified parameter with height
        """
        
        print("Plotting profile of '%s' versus height..." % param_list)
        
        # Convert to list
        if not isinstance(param_list,list):
            param_list = [param_list]
            
        # Get z values at which parameter defined
        z = self.get_z()
        
        # Prepare plot window
        if ax is None:
            fig,ax = plt.subplots()
            y_label=True
            title=True
        else:
            fig = ax.get_figure()
            
        # Loop over all listed parameters
        for param in param_list:
            
            if not hasattr(self,param):
                raise ValueError("Parameter '%s' does not exist!\n" % param +
                                 "Avaliable parameters:\n" + 
                                 "{0}".format(self.__dict__.keys()))
                
            else:
                vals = getattr(self,param)
                                        
                ax.plot(vals,z,label=param)
        
        if len(param_list)>1:
            ax.legend(fontsize=fontsize_labels)
        
        # Define and assign x label
        if x_label is None:
            x_label = "%s" % param
        
        ax.set_xlabel(x_label, fontsize=fontsize_labels)
        
        if y_label:
            ax.set_ylabel("z (m)",fontsize=fontsize_labels)
             
        ax.set_xlim([0,ax.get_xlim()[1]])
        ax.set_ylim([0,ax.get_ylim()[1]])
        
        if title:
            ax.set_title("Variation of '{0}' with height".format(param_list))
            
        return ax
            
            
    def plot_param_3d(self,param,fig_size_inches=(10,8)):
        """
        Produces 3D plot, with specified wind environment parameter set 
        evaluated at all points in current point set
        """
        
        print("Visualising 3D variation of '%s' across point set..." % param)
        
        accepted_params = ['U','i','sigma','iLu','iLv','iLw']
        
        if param not in accepted_params:
            raise ValueError("Unexpected param '%s'" % param + "\n" + 
                             "Allowed: {0}".format(accepted_params))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(fig_size_inches)
        
        x, y, z = self.get_xyz()
        
        if param == 'U':
            
            u,v,w = self.mean_vector.T.tolist()
            
            ax.quiver(x,y,z,u,v,w)
            ax.set_xlim([0,100])
            
        else:
            
            raise ValueError("Not yet implemented!")
            
        # Set axis titles        
        ax.set_xlabel("Along-wind direction, x (m)")
        ax.set_ylabel("Horizontal across-wind direction, y (m)")
        ax.set_zlabel("Vertical direction, z (m)")
        
        return ax
        
            
        
    def calc_mean_speed(self,z=None,**kwargs):
        """
        Calculate mean wind speed at height `z`, given wind environment 
        parameters already defined
        """
        
        if z is None:
            z = self.get_z()
        
        u_star = self.u_star
        K_z = self._calc_K_z(z=z)
        U_z = K_z * u_star
        
        self._U = U_z
        
        return U_z
    
    
    def calc_mean_direction(self,**kwargs):
        """
        Calculates mean wind direction as vector    
        """
        
        angle_rad = numpy.deg2rad(self.direction)
        wind_direction = numpy.array([0.0,-1.0,0.0]) # wind from North
        axis = numpy.array([0.0,0.0,-1.0]) # downward vertical direction
        
        # Rotate clockwise by angle
        wind_direction = rotate_about_axis(wind_direction,axis,angle_rad)
        
        return wind_direction
        
    
    def calc_mean_vector(self,**kwargs):
        """
        Calculates mean wind vector, in a generalised sense where mean speed 
        and direction may vary with spatial coordinates
        """
        speed_vals = numpy.asmatrix(self.calc_mean_speed(**kwargs))
        direction_vectors = numpy.asmatrix(self.calc_mean_direction(**kwargs))
        return speed_vals.T * direction_vectors # matrix of shape [Npoints,3]
    
    
    def calc_iu(self,apply_cook_correction=True):
        """
        Calculate along-wind turbulence intensity per Deaves and Harris
        """
        
        zg = self.zg
        z0 = self.z0
        d = self.d
        z = self.get_z()
        
        # Define non-dimensional heights used in expression
        z_rel_g = (z-d)/zg
        z_rel_0 = (z-d)/z0
        
        num = 3*(1-z_rel_g)*((0.538+0.09*ln(z_rel_0))**((1-z_rel_g)**(16)))
        denom2 = 1 + 0.156*ln(6*zg/z0)
        denom1 = ln(z_rel_0)
        
        if apply_cook_correction:
            # Augment with additional terms per numerator
            # f eqn (9.12) in Cook Part 1
            denom1 += 5.75 * z_rel_g - 1.875 * z_rel_g**2 -(4/3) * z_rel_g**3 + (1/4) * z_rel_g**4
        
        i_u = num / (denom1 * denom2)
        
        self._i_u = i_u
        
        return i_u
    
    
    def calc_RMS_turbulence(self):
        """
        Calculates RMS turbulence components
        """
        
        sigma_u = self.calc_sigma_u()
        sigma_v = self.calc_sigma_v()
        sigma_w = self.calc_sigma_w()
        
        return [sigma_u,sigma_v,sigma_w]
    
    
    def calc_sigma_u(self):
        """
        Calculate RMS along-wind turbulence component
        """

        U = self.U
        i_u = self.i_u
            
        sigma_u = i_u * U        
        
        return sigma_u
    
    
    def calc_sigma_v(self):
        """
        Calculates RMS cross-wind turbulence component 
        per eqn (6.5), ESDU 86010
        """
        
        z = self.get_z()
        sigma_u = self.sigma_u
            
        # eqn (6.5), ESDU 86010
        h = self.zg
        sigma_v = sigma_u * (1 - 0.22 * numpy.cos(numpy.pi/2 * z/h)**4)
    
        self._sigma_v = sigma_v
        
        return sigma_v
    
    
    def calc_sigma_w(self):
        """
        Calculates RMS vertical turbulence component 
        per eqn (6.6), ESDU 86010
        """
        
        z = self.get_z()
        sigma_u = self.sigma_u
            
        # eqn (6.5), ESDU 86010
        h = self.zg
        sigma_w = sigma_u * (1 - 0.45 * numpy.cos(numpy.pi/2 * z/h)**4)
    
        self._sigma_w = sigma_w
        
        return sigma_w
    
    

    
    def calc_exposure_factor(self):
        """
        Calculate exposure factor, i.e. the ratio of the local mean speed 
        at 10m to the basic mean wind speed in standard terrain
        """
        
        # Values for this terrain
        zg = self.zg
        z0 = self.z0
        
        # Standard terrain values
        zg_std = 2550
        z0_std = 0.03
        
        # equation (9.12) from Cook part 1
        term1 = (ln(zg_std/z0_std) + 2.79) / (ln(zg/z0) + 2.79)
        term2 = ln(10.0/z0) / ln(10.0/z0_std)
        SE = term1 * term2
        
        self._S_exposure = SE
        return SE
        
        
    
    def calc_gradient_height(self,zg_assumed=2500):
        """
        Iteratively determine gradient height:
            
        * Initial guess will be made as to zg
        * u_star will be calculated for this zg
        * Determine zg implies by u_star
        * Repeat until convergence
        """
                
        self.zg = zg_assumed # save assumed value for use in class methods
        
        def zg_error(zg):
            
            u_star = self._calc_u_star()
            zg_est = self._calc_gradient_height(u_star=u_star)
            return zg_est - zg

        zg = scipy.optimize.newton(zg_error,zg_assumed)
        
        return zg
            
    
    def _calc_d(self):
        """
        Seeks to calculate zero-plane displacement height consistent with z0
        
        _Refer 9.2.1.2 in Cook Part 1 for basis_
        """
        
        z0 = self.z0
        
        if z0 <= 0.03:
            return 0.0
        
        elif z0 == 0.1:
            return 2.0
    
        elif z0 == 0.3:
            return 10.0
        
        elif z0 == 0.8:
            return 25.0
        
        else:
            raise ValueError("Zero-plane displacement height `d` to be " + 
                             "provided\n(Could not infer from `z0`)")
            
        
    def _calc_R_z(self,z=None):
        """
        Calculates R per Deaves and Harris model
        """
        if z is None:
            z = self.get_z()
            
        zg = self.zg            
        d = self.d

        z_rel = (z-d)/zg
        
        return numpy.min([numpy.ones_like(z_rel),z_rel],axis=0)
    
        
    def _calc_K_z(self,z=None):
        """
        K = U / $u^*$
        """
        
        if z is None:
            z = self.get_z()
            
        z0 = self.z0
        d = self.d
        A1 = self.A1
        
        R = self._calc_R_z(z=z)
        
        K = 2.5*(ln((z-d)/z0) + A1*R
                 + (1-A1/2)*(R**2)
                 - (4/3)*(R**3)
                 + (1/4)*(R**4))
        
        return K
            
        
    def _calc_u_star(self):
        
        z_ref = self.z_ref
        V_ref = self.V_ref
        
        K_ref = self._calc_K_z(z=z_ref)
        
        return V_ref / K_ref
    
        
    def _calc_coriolis_f(self,phi=None,omega=0.0000727):
        """
        Calculates Coriolis parameter

        phi    : site latitude (degrees)

        omega  : Earth's rotation speed (rad/s)
        
        """
        
        if phi is None:
            phi = self.phi
        
        phi = numpy.deg2rad(phi)
        
        f = 2*omega*numpy.sin(phi)
        self.coriolis_f = f
        return f
    
    
    def _calc_gradient_height(self,u_star=None):
        """
        Calculates gradient height, given friction velocity u_star
        """
        
        if u_star is None:
            u_star = self.u_star
            
        B = self.B
        f = self._calc_coriolis_f()
        
        return u_star / (B*f)
    
    
    def calc_Su(self,n):
        """
        Evaluates along-wind power spectrum at specified frequency `n`
        
        
        """
        
        pass
    
    
    def calc_Sv_Sw(self,z,n,make_plot=False,ax_list=None):
        """
        Evaluates horizontal and vertical across-wind power spectrum at 
        specified frequencies `n` and heights 'z'
        
        _Refer eqns (3.5.22) and (3.5.23) from "Wind Loading on Structures", 
        Hansen & Dyrbye. Equations are attributes to Simiu and Scanlan (1986)._
        
        ***
        Required:
            
        * `z`, float or 1d-array of shape (Nz,), defining heights above 
          ground level [m] at which to evaluate power spectrum
        
        * `n`, float or 1d-array of shape (Nn,), defining frequencies [Hz] 
          at which to evaluate power spectrum
            
        ***
        Returns:
            
        Pair of 2d-arrays of shape (Nz,Nn), giving (Sv, Sw) at (z,n) pairs
            
        """
        
        Z, N = numpy.meshgrid(z,n)
        
        U = self.calc_mean_speed(z=z)   
        f_z = self.calc_fz(Z,N,U)
        
        u_star = self.u_star
        
        Sv = (15*f_z) / (1 + 9.5*f_z)**(5/3) * (u_star**2/N)
        Sw = (3.36*f_z) / (1 + 10.0*f_z)**(5/3) * (u_star**2/N)
        
        Sv, Sw = Sv.T, Sw.T  # return arrays of shape [Nz, Nn]
        
        if make_plot:
            
            if ax_list is None:
                fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
            else:
                ax1 = ax_list[0]
                ax2 = ax_list[1]
                fig = ax1.get_figure()
            
            h1 = ax1.plot(Sv,z_vals)
            ax2.plot(Sw,z_vals)
            
            fig.legend(h1,n_vals,title='n [Hz]',fontsize='x-small')
            
            ax1.set_xlabel("$S_v(z,n)$")
            ax1.set_ylabel("z [m]")
            ax2.set_xlabel("$S_w(z,n)$")
            
            ax1.set_ylim([0,ax1.get_ylim()[1]])
            
            fig.set_size_inches((8,8))
            
            fig.suptitle("Horizontal and vertical across-wind power spectra\n"+ 
                         "(Simiu and Scanlan, 1986)")
            
            return Sv, Sw, fig
            
        else:
            return Sv, Sw
        
    
    def calc_fz(self,z,n,U):
        """
        Evaluates Monin similarity coordinate given the following:
        
        * `n`, frequency [Hz]
        
        * `U`, mean wind speed [m/s]
        
        * `z`, height [m]
        
        """
        return n*z/U
       
        
    def calc_turbulence_length_scales(self):
        """
        Calculates turbulence length scales xLu, yLu etc using EDSU 86010
        """
        
        z = self.get_z()
        
        # Calculate xLu, to which all other length scales relate
        xLu = self.calc_xLu()
        
        # Calculate other related length scales for along-wind component
        h = self.zg
        
        zLu = xLu * (0.5 - 0.34 * numpy.exp(-35 * (z/h)**1.7))      # eqn (6.3)
        yLu = xLu * (0.16 + 0.68 * (zLu/zLu))                       # eqn (6.4)       
        self._yLu = yLu
        self._zLu = zLu
        
        # Evaluate turbulence RMS
        sigma_u = self.calc_sigma_u()
        sigma_v = self.calc_sigma_v()
        sigma_w = self.calc_sigma_w()
        
        # Evaluate cross-wind component length scales
        xLv = xLu * (0.5 * (sigma_v/sigma_u)**3)                    #eqn (6.9)
        yLv = xLu * (1.0 * (2 * yLu / xLu) * (sigma_v/sigma_u)**3)  #eqn (6.10)
        zLv = xLu * (0.5 * (2 * zLu / xLu) * (sigma_v/sigma_u)**3)  #eqn (6.11)
        
        self._xLv = xLv
        self._yLv = yLv
        self._zLv = zLv
        
        # Evaluate vertical component length scales
        xLw = xLu * (0.5 * (sigma_w/sigma_u)**3)                    #eqn (6.12)
        yLw = xLu * (0.5 * (2 * yLu / xLu) * (sigma_w/sigma_u)**3)  #eqn (6.13)
        zLw = xLu * (1.0 * (2 * zLu / xLu) * (sigma_w/sigma_u)**3)  #eqn (6.14)
        
        self._xLv = xLv
        self._yLv = yLv
        self._zLv = zLv
                     
        # Return as ndarray
        Nz = len(z)
        length_scales = numpy.zeros((Nz,3,3))
        for i in range(Nz):
            length_scales[i,:,:] = numpy.array([[xLu[i], yLu[i], zLu[i]],
                                                [xLv[i], yLv[i], zLv[i]],
                                                [xLw[i], yLw[i], zLw[i]]])
        return length_scales
    
    
    def calc_xLu(self):
        """
        Calculates along-wind turbulence length scale according to ESDU 85020
        """
        
        z = self.get_z()        
        A = self.A
        u_star = self.u_star
        zg = self.zg
        z0 = self.z0
        f = self.coriolis_f
        sigma_u = self.sigma_u
        
        # Calculate R0 (Rossby number)
        R0 = u_star / (f*z0)
                
        # Calculate other parameters dependent on Rossby number
        B0 = 24 * R0**0.155
        K0 = 0.39 / R0**0.11
        N = 1.24 * R0**0.008
        
        # Calculate A, K parameters as function of height
        K = 0.19 - (0.19-K0) * numpy.exp(-B0 * (z/zg)**N)        
        A = 0.115 * (1 + 0.315 * (1 - z/zg)**6)**(2/3)
        
        # Calculate xLu using the above parameters
        num = A**(3/2) * (sigma_u/u_star)**3 * z
        denom = 2.5 * K**(3/2) * (1 - z/zg)**2 * (1 + 5.75*z/zg)
        xLu = num / denom
        
        self._xLu = xLu
        
        return xLu
        
    
    def calc_rLu(self):
        """
        Evaluates rLu using eqn (6.15), ESDU 86010
        """
        
        yLu = self.yLu
        zLu = self.zLu
        dy = self.dy
        dz = self.dz
        
        # Evaluate rLu per eqn (6.15)
        rLu = ((yLu * dy)**2 + (zLu * dz)**2)**0.5 / (dy**2 + dz**2)**0.5
        self.rLu = rLu
        
        return rLu
    
    
    def calc_rLv(self):
        """
        Evaluates rLv using eqn (6.16), ESDU 86010
        """
        
        xLv = self.xLv
        zLv = self.zLv
        dx = self.dx
        dz = self.dz
        
        # Evaluate rLu per eqn (6.15)
        rLv = ((xLv * dx)**2 + (zLv * dz)**2)**0.5 / (dx**2 + dz**2)**0.5
        self.rLv = rLv
        
        return rLv
    
    
    def calc_rLw(self):
        """
        Evaluates rLw using eqn (6.17), ESDU 86010
        """
        
        xLw = self.xLw
        yLw = self.zLw
        dx = self.dx
        dy = self.dy
        
        # Evaluate rLu per eqn (6.15)
        rLw = ((xLw * dx)**2 + (yLw * dy)**2)**0.5 / (dx**2 + dy**2)**0.5
        self.rLw = rLw
        
        return rLw
   
    
#%%
class WindEnv_single_fetch(WindEnv_equilibrium):
    """
    Implements adjustments required to cater for single upwind fetch change
    
    Refer Section G.1.3 of TOWER manual (r2) for details of method
    """
    
    def __init__(self,X,z0_X,**kwargs):
        """
        Required:
            
        * `X` upwind fetch to roughness change (m)
        
        * `z0_X`, ground roughness at distance `X` upwind (m)
        
        Other keyword arguments are passed to `WindEnv_equilibrium.__init__()`
        
        """
        
        print("***** WARNING: REQUIRES VALIDATION / FURTHER WORK ******")
                
        self.X = X
        """
        Upwind distance (fetch) **in meters** to ground roughness change
        """
        
        self.z0_X = z0_X
        """
        Ground roughness at upwind distance X (m)
        """
        
        # Define equilibrium wind model for site terrain, given parameters
        # passed to this function defining conditions at the site
        kwargs['calc_wind_params']=False
        super().__init__(**kwargs)
        
        # Define equilibrium wind model for upwind terrain
        upwind_kwargs = kwargs.copy()
        upwind_kwargs['z0']=z0_X # overrides value at site
        upwind_kwargs['points_list'] = self.points
        upwind_we = WindEnv_equilibrium(**upwind_kwargs)
        
        self.upwind_we = upwind_we
        """
        Object defining upwind wind environment
        """
                
        self.S_exposure_X = upwind_we.S_exposure
        """
        Exposure factor at upwind terrain
        """
                
        SX = self.calc_fetch_factor()
        self.S_fetch = SX
        """
        Fetch factor, defined as ratio between local friction 
        velocity and equilibrium friction velocity
        """
        
        # Calculate local friction velocity
        u_star = upwind_we.u_star            # equilibrium value at site
        u_star_local = u_star * SX      # by definition of fetch factor
        self.u_star_local = u_star_local
        """
        Local value of friction velocity 'u_star', to account for 
        upwind roughness change
        """
        
        # Define equilibrium wind model for site terrain
        kwargs['u_star']=u_star_local
        super().__init__(**kwargs)
        
        # Calculate heights for roughness changes
        z_bottom, z_top = self.calc_transition_z()
        
        self.z_bottom = z_bottom
        """
        Height (m) denoting lower extent of transition region; 
        refer Fig 9.8, Cook part 1
        """
        
        self.z_top = z_top
        """
        Height (m) denoting upper extent of transition region; 
        refer Fig 9.8, Cook part 1
        """
        
    
    def calc_fetch_factor(self):
        """
        Calculate fetch factor, defined as ratio between local friction 
        velocity and equilibrium friction velocity
        """
        
        X = self.X
        
        # Parameters relating to upwind site
        z0_X = self.z0_X
        SE_X = self.S_exposure_X
        
        # Parameters relating to site
        z0 = self.z0
        SE = self.S_exposure

        if not all([X,z0,z0_X,SE,SE_X]):
            print([X,z0,z0_X,SE,SE_X])
            raise ValueError("Not all parameters initialised!")
        
        # Determine m0
        m0 = calc_m0(X=X,z0=z0)
                        
        # Determine fetch factor
        term1 = 1 - ln(z0_X/z0) / (0.42 + ln(m0))
        term2 = SE_X / ln(10/z0_X)
        term3 = ln(10/z0) / SE
        SX_X = term1 * term2 * term3
                
        return SX_X
    
    
    def calc_transition_z(self):
        """
        Calculates z_bottom and z_top, i.e. heights at which transitions 
        occur between upwind and site flow character
        
        Refer eqns (9.17) to (9.19) in Cook part 1
        """
        
        X = self.X
        d = self.d        # displacement height
        z0 = self.z0      # roughness at sit 
        z0_X = self.z0_X  # upwind roughness
        
        z0_rough = max(z0,z0_X)
        z0_smooth = min(z0,z0_X)
        
        # Calculate z_bottom, i.e. height of bottom of transition region
        
        if z0_X == z0_rough:
            # Rough->Smooth : eqn (9.17) applies
            z_bottom = d + 0.36 * z0_rough**0.25 * X**0.75
            
        else:
            # Smooth->Rough : eqn (9.18) applies
            z_bottom = d + 0.07 * X * (z0_smooth/z0_rough)**0.5
            
        # Calculate z_top, i.e. height of top of transition region
        # eqn (9.19) applies
        z_top = d + 10 * z0_rough**0.4 * X**0.6
        
        return z_bottom, z_top
    
    
    def calc_mean_speed(self,make_plot=True, verbose=True):
        """
        Calculate mean wind speed at height `z`, given wind environment 
        parameters already defined
        """
        
        upwind_wind_env = self.upwind_we
        
        z = self.get_z()
        d = self.d
        z_eff = z - d
        
        SE_site = self.S_exposure
        SE_upwind = self.S_exposure_X
        SE_upwind = 1.37
                
        S_fetch = self.S_fetch
        
        z_bottom = self.z_bottom
        z_top = self.z_top
                
        # Calculate wind speeds for site assuming this is at equilibrium
        # with local terrain
        U_site_equ = super(WindEnv_single_fetch,self).calc_mean_speed()
        
        # Calculate wind speeds for upwind site assuming this is at equilibrium
        # with local terrain
        U_upwind_equ = upwind_wind_env.calc_mean_speed()
        
        # Define local internal wind profile
        U1 = S_fetch * U_site_equ
        
        # Define wind profile for heights in range 0 < z < z_bottom
        U3 = U_upwind_equ
        
        # Locate intersection point z_l as described in Cook 9.4.1.6.1
        U1_func = scipy.interpolate.interp1d(z,U1)
        U3_func = scipy.interpolate.interp1d(z,U3)
        
        def U_diff(z):
            return U3_func(z) - U1_func(z)
        
        try:
            z_l = scipy.optimize.bisect(U_diff,z_bottom,z_top)
        except:
            z_l = z_bottom # conservative
            print("*** Error in calculation of z_l! ***")
        
        # Define transition profile to be used
        U_transition = numpy.where(z_eff < z_l, U1, U3)
        
        if verbose:
            
            print("S_fetch:\t%.3f" % S_fetch)
            print("SE_site\t\t%.3f" % SE_site)
            print("SE_upwind\t%.3f" % SE_upwind)
            print("z_bottom:\t%.1f\t[m]" % z_bottom)
            print("z_l:\t\t%.1f\t[m]" % z_l)
            print("z_top:\t\t%.0f\t[m]" % z_top)
        
        
        if make_plot:
            
            fig,ax = plt.subplots()
            fig.set_size_inches((6,8))
            
            ax.plot(U_site_equ,z_eff,label='New equilibrium profile')
            ax.plot(U_upwind_equ,z_eff,label='Old equilibrium profile')
            
            ax.plot(U1,z_eff,label='U1')
            ax.plot(U3,z_eff,label='U3')
            
            ax.plot(U_transition,z_eff,'k-',label='Composite design profile')
            
            ax.axhline(y=z_bottom,color='k',alpha=0.3)            
            ax.axhline(y=z_l,color='r',alpha=0.3)
            ax.axhline(y=z_top,color='k',alpha=0.3)
                                                
            ax.set_xlim([0,ax.get_xlim()[1]])
            ax.set_ylim([0.001,500]) # per Figure 9.10 in Cook part 1
            
            ax.set_xlabel("Mean wind speed (m/s)")
            ax.set_ylabel("Effective height (z-d) (m)")
            
            ax.set_yscale("log")
            
            ax.set_title("Transition wind profile for single fetch change\n"+
                         "z0 = %.3f; X = %.0f; z0(X) = %.3f" 
                         % (self.z0,self.X,self.z0_X))
            
            ax.legend()
            
        self.U = U_transition
            
        return U_transition
    
    
    def print_details(self):
        
        super().print_details()

        print("z0(X) = %.3f\t[m]" % self.z0_X)
        print("SE(X) = %.3f" % self.S_exposure_X)
        print("SX(X) = %.3f" % self.S_fetch)
    
    
    
#%% --------------- FUNCTIONS -----------------
    
def calc_m0(X,z0,tol=0.000001,max_iter=100,verbose=False):
    """
    Iteratively calculates m0 as required for calculation of fetch factor
    
    m0 is calculated using eqn (9.21) in Cook part 1
    """
   
    def m0_func(m0_val):
        return (0.32 * X) / (z0 * (ln(m0_val)-1))
    
    m0_last = X / (10*z0)   # Initial guess at m0
    iter_count = 0
    
    converged = False
    while not converged and iter_count < max_iter:
        
        iter_count += 1
        
        m0 = m0_func(m0_last)
        
        if abs(m0/m0_last - 1) < tol:
            converged = True
            
        m0_last = m0
        
    if not converged:
        raise ValueError("Could not converge on 'm0'!")
    else:
        if verbose: print("Converged on m0 after %d iterations" % iter_count)
        
    return m0


#%% ------------------- TEST ROUTINES -----------------------
    
def plot_fetch_factor(phi):
    """
    Tests implementation of single roughness change wind model by deriving
    fetch factor for various roughness transitions
    
    Format of figure is intended to replicate Figure 9.9 in Cook part 1
    
    ***
    Required:
        
    * `phi`, latitude of site (degrees)
    """
    
    print("Calculating to produce fetch plot...")
    
    V_ref = 10 # arbitrary
        
    # Practical range of fetch values in m
    fetch_vals = numpy.geomspace(0.1,10000) * 1000
    
    # z0 values for terrain categories - refer Fig 9.7 Cook part 1
    z0_vals = [0.003,0.01,0.03,0.1,0.3,0.8]
    
    # Append to additionally consider reverse roughness transitions
    z0_vals += z0_vals[::-1][1:]

    # Loop over all transitions
    SX_vals = []
    for i in range(len(z0_vals)-1):
        
        z0_0 = z0_vals[i]
        z0_1 = z0_vals[i+1]
        
        SX_vals_inner = []
        
        for X in fetch_vals:        
        
            we = WindEnv_single_fetch(V_ref=V_ref,
                                      z0=z0_1,
                                      X=X,
                                      z0_X=z0_0,
                                      phi=phi,
                                      direction=0.0 #arbitary)
                                      )
            
            SX_vals_inner.append(we.S_fetch)
            
        SX_vals.append(SX_vals_inner)
        
    SX_vals = numpy.array(SX_vals)
        
    # Make plot and scale per Cook's to faciliate comparison
    fig,ax = plt.subplots()
    
    fetch_vals = fetch_vals/1000 # convert to km for plot
    ax.semilogx(fetch_vals,SX_vals.T)
    
    ax.set_xlabel("Upwind fetch (km)")
    ax.set_xlim([0.1,200])
    ax.set_ylim([0.8,1.2])
    ax.set_title("Fetch factor")
    
    print("Done!")
    
    return fig

#%%

if __name__ == "__main__":
    
    test_routine = 4
    
    wind_params = {}
    wind_params['V_ref']=28.2
    wind_params['z0']=0.03
    wind_params['phi']=53.0
    wind_params['direction']=90.0
    
    if test_routine == 1:
        
        print("Demo #1: Equilibrium Wind profiles")
            
        we = WindEnv_equilibrium(**wind_params)
        
        Se = we.calc_exposure_factor()
        we.print_details()
        we.plot_profiles()
        
    elif test_routine == 2:
        
        plot_fetch_factor(53.0)
        
    elif test_routine == 3:
    
        we = WindEnv_single_fetch(X=500,z0_X=0.003,**wind_params)
        we.calc_mean_speed()
        
    elif test_routine == 4:
        
        print("TEST #4: Check of Sw, Sv methods")
              
        we = WindEnv_equilibrium(**wind_params)
        z_vals = numpy.arange(10,301,10)
        n_vals = [0.1,0.15,0.5,1.5]
        Sv, Sw, fig = we.calc_Sv_Sw(z_vals,n_vals,make_plot=True)
