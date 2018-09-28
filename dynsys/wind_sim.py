# -*- coding: utf-8 -*-
"""
Script to demonstrate simulation of turbulence by Monte Carlo method
Refer "Three-Dimensional Wind Simulation" by Veers, 1988
"""

import numpy
import matplotlib.pyplot as plt
import scipy

#%%
   
def G_vonKarman(z,f,U_ref=20.0,z_ref=10.0,z0=0.05,Lx=120.0):
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
        
    return Gv, U_z


def coherance(points,f_vals,U_vals):
    """
    Defines coherance per Eqn 4.1
    
    points   : assumed 3D i.e. array of shape (Np,3) where Np is number of points
    
    f_vals   : frequency vector, shape (Nf,)
    
    U_vals   : mean wind speed at each point, vector of shape (Np,)
    
    """
    
    #Np = points.shape[0]
    #Nf = len(f_vals)
    
    # Compute distance between points
    r_jk = scipy.spatial.distance.cdist(points,points,metric='euclidean')
    
    # Evaluate Cjk per eqn 4.2
    Zm = numpy.mean(numpy.meshgrid(points[:,2],points[:,2]),axis=0)
    Um = numpy.mean(numpy.meshgrid(U_vals,U_vals), axis=0)
    
    mu_b = 2*numpy.random.rand(*r_jk.shape) - 1
    b = 12 + 5 * mu_b
    C_jk = b * (r_jk/Zm)**0.25
        
    # Evaluate coherance per eqn 4.1
    Coh=[]
    
    f_vals = numpy.abs(f_vals)
    
    for f in f_vals:
        Coh_f = numpy.exp(-C_jk * r_jk * f / Um)
        Coh.append(Coh_f)
        
    Coh = numpy.array(Coh)
    
    return Coh, r_jk, Zm


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

#%%
    
if __name__ == "__main__":
    
    test_routine = 1
    
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
        z_vals = numpy.linspace(1,90,5)
        N = len(z_vals) # number of grid points / correlated processes
        points = numpy.zeros((N,3))
        points[:,2]=z_vals
        
        # Test coherance function
        Coh, r_jk, Zm = coherance(points,[0.012,0.037,0.11,0.33],8.0)
        
        fig,ax = plt.subplots()
        for Coh_f in Coh:
            ax.plot((r_jk/Zm),Coh_f)

    else:
            
        # Define points
        z_vals = numpy.linspace(1,90,5)
        N = len(z_vals) # number of grid points / correlated processes
        points = numpy.zeros((N,3))
        points[:,2]=z_vals
        
        # Define sim properties    
        t_duration = 60.0
        fs = 1.0
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
        
        fig,axarr = plt.subplots(2,sharex=True)
        
        ax = axarr[0]
        ax.plot(fm,Gv_fm)
        
        ax = axarr[1]
        ax.plot(fm,numpy.abs((Gv_fm.T * fm).T))
        ax.set_xlim([-fs/2,+fs/2])
        ax.set_xlabel("Frequency f (Hz)")
        ax.set_ylabel("f.G(f)")
        #ax.set_xscale("log")#, nonposy='clip')
        ax.set_yscale("log")#, nonposy='clip')
        
        #%%
        
        # Generate coherance matrix at each frequency
        Coh_fm = coherance(points,fm,U_vals)[0]
        
        fig, axarr = plt.subplots(N,N,sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0,wspace=0)
        
        for j in range(N):
            for k in range(N):
                ax = axarr[j,k]
                ax.plot(fm,Coh_fm[:,j,k])
                ax.set_yscale("log")#, nonposy='clip')
        
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
        
        plt.plot(t_vals,v)

    