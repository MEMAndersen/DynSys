# -*- coding: utf-8 -*-
"""
Script to demonstrate simulation of turbulence by Monte Carlo method
Refer "Three-Dimensional Wind Simulation" by Veers, 1988
"""

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy
from scipy import spatial
from scipy import interpolate
from scipy import optimize

from numpy import log as ln

from modalsys import ModalSys
from common import check_class

#%%

plt.close('all')
fontsize_labels = 8
   
#%%

class BuffetingAnalysis():
    """
    Class to implement gust-buffeting analysis
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
            
        mesh_obj = sys.mesh
        
    
        
        
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
    


def G_vonKarman(z,f,U_ref=20.0,z_ref=10.0,z0=0.05,Lx=120.0,make_plot=False):
    """
    Von Karman expression for along-wind turbulence autospectrum
    
    z       : Height above ground (m), 1D array expected
    f       : Frequency (Hz), 1D array expected
    
    U_10    : mean wind speed at height z_ref (m/s)
    z_ref   : reference height for mean wind speed (m) - see above
    z0      : ground roughness (m)
    
    Lx      : along-wind turbulence length scale    
    
    """
    
    # Determine shear velocity
    u_star = 0.4 * U_ref / ln(z_ref/z0)
    
    # Calculate mean wind speed at heights requested
    U_z = u_star / 0.4 * ln(z/z0)
    
    # Calculate Lx at each height
    if isinstance(Lx,float):
        Lx_z = Lx * numpy.ones((len(z),))
    else:
        # Assume function provided
        Lx_z = Lx(z)
    
    # Determine along-wind turbulence spectrum at requested frequencies
    Gv = numpy.empty((len(f),len(U_z)))
    
    for i, (U, Lx) in enumerate(zip(U_z,Lx_z)):
        
        numerator = 4 * (5.7 * u_star**2) * (Lx/U)
        denominator = 1.339 * (1 + 39.48 * (f*Lx/U)**2)**(5/6)
        Gv[:,i] = numerator / denominator
        
    if make_plot:
        
        fig,axarr = plt.subplots(2,sharex=True)
        
        ax = axarr[0]
        ax.plot(f,Gv)
        
        ax = axarr[1]
        ax.plot(fm,numpy.abs((Gv.T * f).T))
        ax.set_xlim([-fs/2,+fs/2])
        ax.set_xlabel("Frequency f (Hz)")
        ax.set_ylabel("f.G(f)")
        #ax.set_xscale("log")#, nonposy='clip')
        ax.set_yscale("log")#, nonposy='clip')
        
    return Gv, U_z


def coherance(points,f_vals,U_vals,zi:int=-1,make_plot=False):
    """
    Defines coherance per Eqn 4.1
    
    points   : assumed 3D i.e. array of shape (Np,3) where Np is number of points
    
    f_vals   : frequency vector, shape (Nf,)
    
    U_vals   : mean wind speed at each point, vector of shape (Np,)
    
    """
    
    if isinstance(zi,int):
        zi = [zi]
            
    #Np = points.shape[0]
    #Nf = len(f_vals)
    
    # Compute distance between points
    r_jk = scipy.spatial.distance.cdist(points,points,metric='euclidean')
    
    # Compute coherance decrement
    mu_b = 2*numpy.random.rand() - 1
    b = 12 + 5 * mu_b
    print("b = %.2f" % b)
    
    # Evaluate Cjk per eqn 4.2
    Zm = numpy.mean(numpy.meshgrid(points[:,2],points[:,2]),axis=0)
    Um = numpy.mean(numpy.meshgrid(U_vals,U_vals), axis=0)
    
    C_jk = b * (r_jk/Zm)**0.25
        
    # Evaluate coherance per eqn 4.1
    Coh=[]
    
    f_vals = numpy.abs(f_vals)
    
    for f in f_vals:
        Coh_f = numpy.exp(-C_jk * r_jk * f / Um)
        Coh.append(Coh_f)
        
    Coh = numpy.array(Coh)
    
    if make_plot:
        
        r_Z = r_jk/Zm
        
        fig, axarr = plt.subplots(len(zi))
        
        if not isinstance(axarr,numpy.ndarray):
            axarr = numpy.array([axarr])
        
        for i, ax in zip(zi,axarr):
                
            for Coh_f in Coh:

                ax.plot(r_Z[:,i],Coh_f[:,i])
            
            ax.set_xlim([0,2.0])
            ax.set_ylim([0,1.0])
            
            ax.set_xlabel(r"$\Delta r_{jk} / z$")
            ax.set_ylabel("Coherance")
    
    return Coh


def spectral_density_matrix(G_fm,Coh_fm,df):
    """
    Calculate spectral density matrix using coherance definition e.g. eqn(2.1)
    
    G_f     : ndarray of shape (Nf,Np) expected
    Coh_f   : ndarray of shape (Nf,Np,Np) expected
    
    where Np is number of points and Nf is number of frequencies
    
    """
    
    S_f = []
    
    for Gv, Coh in zip(Gv_fm,Coh_fm):
        S_f.append(Coh * df / 2 * numpy.multiply(*numpy.meshgrid(Gv,Gv)))
    
    return numpy.array(S_f)


def white_noise_mtrx(N):
    """
    Returns matrix of independent white-noise inputs per eqn (2.4)
    """
    
    X = []
    
    for f in enumerate(f_vals):
    
        # Random phase angles in [0,2pi]
        phase_angles = numpy.random.random_sample(N) * 2*numpy.pi
        
        # Assemble diagonal X matrix at each freq
        X.append(numpy.diag(numpy.exp(1j*phase_angles)))
        
    return numpy.array(X)


def weighting_mtrx(S_f):
    """
    Recursively determine H matrix per eqn (2.3)
    """
    
    print("Calculating weighting matrix, H...")
    
    # Check shape
    Nf,Np,Np2 = S_f.shape
    if Np!=Np2:
        raise ValueError("`S_f` unexpected shape: {0}".format(S_f.shape))
    
    H = []
    
    for fi, S in enumerate(S_f):
        
        H_i = numpy.zeros_like(S)
                
        # Iterate over lower triangular matrix
        for j in range(Np):
            
            for k in range(j+1):
                
                H_jk = S[j,k]
                
                if j!=k:
                    
                    for l in range(k):
                        H_jk -= H_i[j,l] * H_i[k,l]
                    
                    H_kk = H_i[k,k]
                    
                    if H_kk==0:
                        raise ValueError("Division by zero!")
                        
                    H_jk /= H_kk
                    
                else:
                    
                    for l in range(k):
                        H_jk -= H_i[k,l]**2
                                        
                    H_jk **= 0.5
                        
                H_i[j,k]=H_jk
                                      
        H.append(H_i)
        
    print("Complete!")
         
    return numpy.array(H)


def check_weighting_mtrx(H_f,S_f):
    """
    Assert that H.HT = S for all frequencies
    """
    for H, S in zip(H_f,S_f):
        
        S_expected = H @ H.T
        numpy.testing.assert_almost_equal(S,S_expected)



def calc_fourier_coeffs(H,X):
    
    V = []
        
    for H_f, X_f in zip(H,X):
        V.append(H_f @ X_f @ numpy.ones((X_f.shape[0],)))

    return numpy.array(V)


# ------------------------------
    
class PointSet():
    """
    Class used to collect point properties
    """
    
    def __init__(self,points):
        """
        points to be of shape (Np,3)
        """
        
        self.nPoints = points.shape[0]
        if points.shape[1]!=3:
            raise ValueError("`points` to be of shape (nPoints,3)")
        
        # Unpack coordinates
        x,y,z = [points[:,i] for i in range(3)]
        self.x = x
        self.y = y
        self.z = z
        
        # Run methods
        self.calc_average_position()
        self.calc_seperation()
        
    
    
    def calc_average_position(self):
        """
        Calculates average position of pairs of points in set
        """
        
        x = self.x
        y = self.y
        z = self.z
        
        vm = []
        
        for v in [x,y,z]:
            
            V1, V2 = numpy.meshgrid(v,v)
            vm.append(numpy.mean([V1,V2],axis=0))
        
        xm,ym,zm = vm
        
        self.xm = xm
        self.ym = ym
        self.zm = zm
        
        return xm,ym,zm
        
        
    def calc_seperation(self):
        """
        Calculates distance seperation in component directions, for pairs of 
        points in set
        """
        
        x = self.x
        y = self.y
        z = self.z
        
        dv = []
        
        for v in [x,y,z]:
            
            V1, V2 = numpy.meshgrid(v,v)
            dv.append(numpy.subtract(V2,V1))
        
        dx,dy,dz = dv
        dr = (dx**2 + dy**2 + dz**2)**0.5
        
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dr = dr
        
        
class WindEnv():
    """
    Base class used to defines those attributes and methods required to 
    define a 'wind environment' for a given site
    
    _Note: implemented as an abstract class. Objects cannot be instantiated_
    """
    
    def __init__(self):
        
        # Prevent direct instatiation of this class
        if type(self) == WindEnv:
            raise Exception("<WindEnv> must be subclassed.")


class WindEnv_equilibrium(WindEnv):
    """
    Defines all attributes and methods required to define wind environment
    """
    
    def __init__(self,V_ref,
                 pointset_obj=None,
                 points_arr=None,
                 mean_wind_dir=numpy.array([1,0,0]),
                 z_ref=10.0,
                 z_min=0.1,
                 z_max=500,
                 nz=100,
                 z0=0.03,d=0.0,
                 phi=53.612,
                 A=-1.0,B=6.0,
                 u_star=None,
                 calc_wind_params=True):
        """
        Calculate coherance in accordance with ESDU data item 86010
        
        points_arr : (Np,3) array defining points to evaluate wind properties at
        (typically this might be gauss points)
        
        V_ref  : wind speed (m/s) at height z_ref
        
        zg     : initial guess at gradient height (m). _This is determined iteratively_
        
        z0     : ground roughness (m). Default corresponds to 'Country'
        
        d      : displacement height (m). Default corresponds to 'Country'
        
        A,B    : parameters required for use in log-law formula
        
        phi    : site latitude (degrees)
        
        
        """
        
        if pointset_obj is None:
            
            if points_arr is None:
                # Create default points set, e.g. for checking wind profile
                z_vals = numpy.geomspace(z_min,z_max,nz)
                Np = len(z_vals)
                points_arr = numpy.zeros((Np,3))
                points_arr[:,2]=z_vals
                
            pointset_obj = PointSet(points_arr)
        
        self.pointset_obj = pointset_obj
        Np = self.pointset_obj.nPoints
        
        self.zg = None
        """
        Gradient height (m)
        """
                
        self.z0 = z0
        """
        Ground roughness (m)
        """
        
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
        
        # Initialise variables defined later
        self.U = None
        """
        Mean wind speed (m/s) at points
        """
        
        self.mean_vector = None
        """
        Mean wind velocity vector at each point: shape (Np,3)
        """
        
        self.i_u = None
        """
        Along-wind turbulence intensity
        """
        
        self.i_v = None
        """
        Horizontal across-wind turbulence intensity
        """
        
        self.i_w = None
        """
        Vertical across-wind turbulence intensity
        """
        
        self.sigma_u = None
        """
        Along-wind RMS turbulence (m/s)
        """
        
        self.sigma_v = None
        """
        Horizontal across-wind RMS turbulence (m/s)
        """
        
        self.sigma_w = None
        """
        Vertical across-wind RMS turbulence (m/s)
        """
        
        # ------------------------
        
        self.xLu = None
        """
        Integral turbulence length scale (m):
        
        * Along-wind component (u)
        
        * Along-wind direction (x)
        """        
        
        self.yLu = None
        """
        Integral turbulence length scale (m):
        
        * Along-wind component (u)
        
        * Horizontal across-wind direction (y)
        """ 
        
        self.zLu = None
        """
        Integral turbulence length scale (m):
        
        * Along-wind component (u)
        
        * Vertical across-wind direction (z)
        """ 
        
        # ----
        
        self.xLv = None
        """
        Integral turbulence length scale (m):
        
        * Horizontal across-wind component (v)
        
        * Along-wind direction (x)
        """        
        
        self.yLv = None
        """
        Integral turbulence length scale (m):
        
        * Horizontal across-wind component (v)
        
        * Horizontal across-wind direction (y)
        """ 
        
        self.zLv = None
        """
        Integral turbulence length scale (m):
        
        * Horizontal across-wind component (v)
        
        * Vertical across-wind direction (z)
        """ 
        
        # ----
        
        self.xLw = None
        """
        Integral turbulence length scale (m):
        
        * Vertical across-wind component (w)
        
        * Along-wind direction (x)
        """        
        
        self.yLw = None
        """
        Integral turbulence length scale (m):
            
        * Vertical across-wind component (w)
        
        * Horizontal across-wind direction (y)
        """ 
        
        self.zLw = None
        """
        Integral turbulence length scale (m):
            
        * Vertical across-wind component (w)
        
        * Vertical across-wind direction (z)
        """ 
        
        self.S_exposure = None
        """
        Ratio of the local mean speed at 10m to the basic mean wind speed in 
        standard terrain
        """
        
        # ---------------------------
        
        self.calc_exposure_factor()
        
        # Evaluate wind params at point set currently defined
        if calc_wind_params:
            self.calculate_wind_params(mean_wind_dir)
        
        
    def calculate_wind_params(self,mean_wind_dir):
        """
        Evaluate wind parameters at current point set
        
        * `mean_wind_dir`, vector defining mean wind direction in world coords
          (xyz), as per point set
          
        """
        self.calc_mean_speed()
        self.calc_mean_vector(mean_wind_dir)
        self.calc_iu()
        self.calc_RMS_turbulence()
        self.calc_turbulence_length_scales()
               
    
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
        z = self.pointset_obj.z
        
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
        
            
        
    def calc_mean_speed(self,z=None):
        """
        Calculate mean wind speed at height `z`, given wind environment 
        parameters already defined
        """
        
        if z is None:
            z = self.get_z()
        
        u_star = self.u_star
        K_z = self._calc_K_z(z=z)
        U_z = K_z * u_star
        
        self.U = U_z
        
        return U_z
    
    
    
    def calc_mean_vector(self,mean_wind_dir):
        """
        Calculates mean wind vector at each point        
        """
        
        U = self.U
        mean_vector = numpy.array([x * mean_wind_dir for x in U])
        self.mean_vector = mean_vector
        return mean_vector
        
    
    
    
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
        
        self.i_u = i_u
        
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
        self.sigma_u = sigma_u        
        return sigma_u
    
    
    def calc_sigma_v(self):
        """
        Calculates RMS cross-wind turbulence component 
        per eqn (6.5), ESDU 86010
        """
        
        z = self.get_z()
        U = self.U
        sigma_u = self.sigma_u
            
        # eqn (6.5), ESDU 86010
        h = self.zg
        sigma_v = sigma_u * (1 - 0.22 * numpy.cos(numpy.pi/2 * z/h)**4)
    
        self.sigma_v = sigma_v
        self.i_v = sigma_v / U
        return sigma_v
    
    
    def calc_sigma_w(self):
        """
        Calculates RMS vertical turbulence component 
        per eqn (6.6), ESDU 86010
        """
        
        z = self.get_z()
        U = self.U
        sigma_u = self.sigma_u
            
        # eqn (6.5), ESDU 86010
        h = self.zg
        sigma_w = sigma_u * (1 - 0.45 * numpy.cos(numpy.pi/2 * z/h)**4)
    
        self.sigma_w = sigma_w
        self.i_w = sigma_w / U
        
        return sigma_w
    
    
    def get_x(self):
        return self.pointset_obj.x

    def get_y(self):
        return self.pointset_obj.y

    def get_z(self):
        return self.pointset_obj.z
        
    def get_xyz(self):
        return self.get_x(), self.get_y(), self.get_z()
    
    
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
        
        self.S_exposure = SE
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
    
    
    def coherance(self):
        
        pointset_obj = self.pointset_obj
        
        # Get seperation of pairs of points
        dr = pointset_obj.dr
    
        # Evaluate rLu_func at mean position of point pairs
        xm = pointset_obj.xm
        ym = pointset_obj.ym
        zm = pointset_obj.zm
        rLu_vals = self.rLu_func(zm,xm,ym)
        
        return zm, dz
    
        
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
        self.yLu = yLu
        self.zLu = zLu
        
        # Evaluate turbulence RMS
        sigma_u = self.calc_sigma_u()
        sigma_v = self.calc_sigma_v()
        sigma_w = self.calc_sigma_w()
        
        # Evaluate cross-wind component length scales
        xLv = xLu * (0.5 * (sigma_v/sigma_u)**3)                    #eqn (6.9)
        yLv = xLu * (1.0 * (2 * yLu / xLu) * (sigma_v/sigma_u)**3)  #eqn (6.10)
        zLv = xLu * (0.5 * (2 * zLu / xLu) * (sigma_v/sigma_u)**3)  #eqn (6.11)
        
        self.xLv = xLv
        self.yLv = yLv
        self.zLv = zLv
        
        # Evaluate vertical component length scales
        xLw = xLu * (0.5 * (sigma_w/sigma_u)**3)                    #eqn (6.12)
        yLw = xLu * (0.5 * (2 * yLu / xLu) * (sigma_w/sigma_u)**3)  #eqn (6.13)
        zLw = xLu * (1.0 * (2 * zLu / xLu) * (sigma_w/sigma_u)**3)  #eqn (6.14)
        
        self.xLv = xLv
        self.yLv = yLv
        self.zLv = zLv
                     
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
        self.xLu = xLu
        
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
        upwind_kwargs['pointset_obj'] = self.pointset_obj
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
    
   
#%%
        
        
# --------------- FUNCTIONS -----------------
    
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

# ------------------- TEST ROUTINES -----------------------
    
def plot_fetch_factor():
    """
    Tests implementation of single roughness change wind model by deriving
    fetch factor for various roughness transitions
    
    Format of figure is intended to replicate Figure 9.9 in Cook part 1
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
        
            we = WindEnv_single_fetch(V_ref=V_ref,z0=z0_1,X=X,z0_X=z0_0)
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
    
    test_routine = 6
    
    if test_routine == 0:
    
        # Override with test array
        S_f = numpy.genfromtxt('S_test.csv',delimiter=',')
        S_f = numpy.array([S_f,S_f])
                
        # Calculate weighting matrix H
        H_f = weighting_mtrx(S_f)
        
        # Check that H.HT = S_f
        check_weighting_mtrx(H_f,S_f)
    
    elif test_routine == 1:
        
        # Define points
        z_vals = numpy.linspace(10,100,20)
        N = len(z_vals) # number of grid points / correlated processes
        points = numpy.zeros((N,3))
        points[:,2]=z_vals
        
        # Test coherance function
        Coh = coherance(points,[0.012,0.037,0.11,0.33],8.0,
                        make_plot=True)#,zi=[5,7])
        
    
    elif test_routine == 2:
        
        # Define points
        z_vals = numpy.linspace(10,100,5)
        N = len(z_vals) # number of grid points / correlated processes
        points = numpy.zeros((N,3))
        points[:,2]=z_vals
        
        P = coherance_ESDU86010(points,[0.012,0.037,0.11,0.33])
        

    elif test_routine ==3:
        
        we = WindEnv_equilibrium(V_ref=28.2,z0=0.03)
        Se = we.calc_exposure_factor()
        we.print_details()
        we.plot_profiles()
        #we.plot_param_3d(param='U')
        #we.plot_param_3d(param='iLu')
        
    elif test_routine ==4:
        
        plot_fetch_factor()
        
    elif test_routine ==5:
    
        we = WindEnv_single_fetch(V_ref=20.0,z0=0.300,X=500,z0_X=0.003)
        we.calc_mean_speed()
        
    elif test_routine ==6:
        
        print("TEST #6: Check of Sw, Sv methods")
        we = WindEnv_equilibrium(V_ref=28.2,z0=0.03) 
        z_vals = numpy.arange(10,301,10)
        n_vals = [0.1,0.15,0.5,1.5]
        Sv, Sw, fig = we.calc_Sv_Sw(z_vals,n_vals,make_plot=True)
        


    else:
            
        # Define points
        z_vals = numpy.linspace(10,100,5)
        N = len(z_vals) # number of grid points / correlated processes
        points = numpy.zeros((N,3))
        points[:,2]=z_vals
        
        # Define sim properties    
        t_duration = 10.0
        fs = 10.0
        dt = 1/fs
        t_vals = numpy.arange(0,t_duration+dt/2,dt)
        
        f_vals = numpy.fft.fftfreq(len(t_vals),dt)
        df = f_vals[1] - f_vals[0]
        Nf = len(f_vals)
        
        # Sort into ascending frequency
        f_order = numpy.argsort(f_vals)
        f_vals = f_vals[f_order]
                       
        # Get centre frequency for bins
        fm = numpy.mean([f_vals[:-1],f_vals[1:]],axis=0)
           
        # Calculate along-wind turbulences at centre freqs
        Gv_fm, U_vals = G_vonKarman(points[:,2],fm)
                
        # Generate coherance matrix at each frequency
        Coh_fm = coherance(points,fm,U_vals)
        
#        fig, axarr = plt.subplots(N,N,sharex=True, sharey=True)
#        fig.subplots_adjust(hspace=0,wspace=0)
#        
#        for j in range(N):
#            for k in range(N):
#                ax = axarr[j,k]
#                ax.plot(fm,Coh_fm[:,j,k])
#                ax.set_yscale("log")#, nonposy='clip')
        
        # Calculate spectral density matrix at each frequency
        S_fm = spectral_density_matrix(Gv_fm,Coh_fm,df)
                
        # Define random diagonal matrices at each freq
        X = white_noise_mtrx(N)
                
        # Calculate weighting matrix H
        H_fm = weighting_mtrx(S_fm)
        
        # Check that H.HT = S_f
        check_weighting_mtrx(H_fm,S_fm)
        
        # Calculate Fourier coeffs
        V = calc_fourier_coeffs(H_fm,X)
        
        # Re
        
        # Calculate time series by inverse FFT
        v = numpy.fft.ifft(V,len(t_vals),axis=0)
        v = numpy.real(v)
        
        fig, ax = plt.subplots()
        h = ax.plot(t_vals,v)
        ax.set_xlabel("Time (secs)")
        ax.set_ylabel("Turbulence component u (m/s)")
        ax.legend(h,["Z = %.0fm" % z for z in z_vals])
    