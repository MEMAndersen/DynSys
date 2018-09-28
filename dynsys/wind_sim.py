# -*- coding: utf-8 -*-
"""
Script to demonstrate simulation of turbulence by Monte Carlo method
Refer "Three-Dimensional Wind Simulation" by Veers, 1988
"""

import numpy
import matplotlib.pyplot as plt
import scipy
from scipy import spatial
from scipy import optimize

#%%

fontsize_labels = 8
   
#%%

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
    u_star = 0.4 * U_ref / numpy.log(z_ref/z0)
    
    # Calculate mean wind speed at heights requested
    U_z = u_star / 0.4 * numpy.log(z/z0)
    
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


class WindEnvironment():
    """
    Defines all attributes and methods required to define wind environment
    """
    
    def __init__(self,V_ref,
                 points_arr=None,
                 z_ref=10.0,
                 z_max=300,
                 dz=10.0,
                 z0=0.03,d=0.0,
                 phi=53.612,
                 A=-1.0,B=6.0):
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
        
        if points_arr is None:
            # Create default points set, e.g. for checking wind profile
            z_vals = numpy.arange(z_ref,z_max+0.5*dz,dz)
            Np = len(z_vals)
            points_arr = numpy.zeros((Np,3))
            points_arr[:,2]=z_vals
        
        self.pointset_obj = PointSet(points_arr)
        
        
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
    
        # Calculate A1 for use in log-law formula
        A1 = 2*(numpy.log(B)-A) - (1/6)
        
        self.A = A
        self.B = B
        self.A1 = A1
        
        # Iteratively calculate gradient height consistent with defined terrain
        self.zg = self.calc_gradient_height()
        """
        Gradient height (m)
        """
        
        # Recalculate u_star now that zg determined
        self.u_star = self._calc_u_star()
        """
        Friction velocity (m/s)
        """
        
        self.U = None
        """
        Mean wind speed (m/s) at points
        """
        
        print("Wind environment initialised")
        
    
    def print_details(self):
        print("Vref = %.1f\t[m/s]" % self.V_ref)
        print("zref = %.0f\t[m]" % self.z_ref)
        print("z0 = %.3f\t[m]" % self.z0)
        print("zg = %.0f\t[m]" % self.zg)
        print("d = %.1f\t\t[m]" % self.d)
        print("phi = %.1f\t[deg]" % self.phi)
        print("u* = %.2f\t[m/s]" % self.u_star)
        
        
    def plot(self):
        
        fig,axarr = plt.subplots(1,3,sharey=True)
        fig.set_size_inches((10,8))
        fig.suptitle("Variation of along-wind properties with height")
        
        self.plot_mean_speed(ax=axarr[0],y_label=True)
        self.plot_iuu(ax=axarr[1])
        self.plot_sigma_u(ax=axarr[2])
        
        
    def plot_mean_speed(self,ax=None,
                        y_label=False,title=False,
                        recalculate=True):
        """
        Plots variation of mean wind speed with height
        """
        
        z = self.pointset_obj.z
        
        if recalculate:
            U = self.calc_mean_speed()
        else:
            U = self.U
            
        if ax is None:
            fig,ax = plt.subplots()
            y_label=True
            title=True
        else:
            fig = ax.get_figure()
            
        ax.plot(U,z)
        ax.set_xlabel("Mean wind speed (m/s)",
                      fontsize=fontsize_labels)
        
        if y_label:
            ax.set_ylabel("Height above ground (m)",
                          fontsize=fontsize_labels)
            
        ax.set_xlim([0,ax.get_xlim()[1]])
        ax.set_ylim([0,ax.get_ylim()[1]])
        
        if title:
            ax.set_title("Variation of mean wind speed with height")
        
    
    def plot_iuu(self,ax=None,write_labels=False,recalculate=True):
        """
        Plots variation of along-wind turbulence intensity with height
        """
        
        z = self.pointset_obj.z
        
        if recalculate:
            iuu = self.calc_iuu()
        else:
            iuu = self.iuu
            
        if ax is None:
            fig,ax = plt.subplots()
            write_labels=True
        else:
            fig = ax.get_figure()
            
        ax.plot(iuu,z)
        
        ax.set_xlabel("Along-wind turbulence intensity, $i_{uu}$",
                      fontsize=fontsize_labels)
        
        if write_labels:
            ax.set_ylabel("Height above ground (m)",
                          fontsize=fontsize_labels)
        
        ax.set_xlim([0,ax.get_xlim()[1]])
        ax.set_ylim([0,ax.get_ylim()[1]])
        
        if write_labels:
            ax.set_title("Variation of along-wind turbulence " + 
                         "intensity with height")
            
    
    def plot_sigma_u(self,ax=None,write_labels=False,recalculate=False):
        """
        Plots variation of along-wind RMS turbulence with height
        """
        
        z = self.pointset_obj.z
        
        sigma_u = self.calc_sigma_u(recalculate=recalculate)
        
        if ax is None:
            fig,ax = plt.subplots()
            write_labels=True
        else:
            fig = ax.get_figure()
            
        ax.plot(sigma_u,z)
        
        ax.set_xlabel(r"Along-wind RMS turbulence $\sigma_{u}$ (m/s)",
                      fontsize=fontsize_labels)
        
        if write_labels: 
            ax.set_ylabel("Height above ground (m)",
                          fontsize=fontsize_labels)
            
        ax.set_xlim([0,ax.get_xlim()[1]])
        ax.set_ylim([0,ax.get_ylim()[1]])
        
        
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
    
    
    def calc_iuu(self,z=None):
        """
        Calculate along-wind turbulence intensity per Deaves and Harris
        
        (including correction per Nick Cooks book)
        """
        #raise ValueError("'calc_iuu' method not yet implemented!")
        
        if z is None:
            z = self.get_z()
        
        zg = self.zg
        z0 = self.z0
        d = self.d
        
        # Define non-dimensional heights used in expression
        z_rel_g = (z-d)/zg
        z_rel_0 = (z-d)/z0
        
        num = 3*(1-z_rel_g)*((0.538+0.09*numpy.log(z_rel_0))**((1-z_rel_g)**(16)))
        denom = numpy.log(z_rel_0)*(1+0.156*numpy.log(6*zg/z0))
        i_uu = num / denom
        self.i_uu = i_uu
        return i_uu
    
    
    def calc_sigma_u(self,z=None,recalculate=False):
        """
        Calculate RMS along-wind turbulence component
        """
        if z is None:
            z = self.get_z()
        else:
            recalculate = True
            
        if recalculate:
            U = self.calc_mean_speed(z=z)
            i_uu = self.calc_iuu(z=z)
            
        else:
            U = self.U
            i_uu = self.i_uu
            
        sigma_u = i_uu * U
        
        self.sigma_u = sigma_u
        return sigma_u
    
    
    def get_z(self):
        return self.pointset_obj.z
        
        
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
        
        K = 2.5*(numpy.log((z-d)/z0) + A1*R
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
        
        return 2*omega*numpy.sin(phi)
    
    
    def _calc_gradient_height(self,u_star=None):
        """
        Calculates gradient height, given friction velocity u_star
        """
        
        if u_star is None:
            u_star = self.u_star
            
        B = self.B
        f = self._calc_coriolis_f()
        
        return u_star / (B*f)
        
        
    
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
    
    
    def calc_xLu(z,x=None,y=None):
        """
        Evaluates xLu at position (x,y,z)
        with xLu per ESDU data item 85020
        """
        pass
        
        
    
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
    
if __name__ == "__main__":
    
    test_routine = 3
    
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
        
        we = WindEnvironment(V_ref=28.2095)
        we.print_details()
        we.plot()

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
    