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
    
    for f in f_vals:
        Coh.append(numpy.exp(-C_jk * r_jk * f / Um))
        
    Coh = numpy.array(Coh)
    
    return Coh


def spectral_density_matrix(G_f,Coh_f):
    """
    Calculate spectral density matrix using coherance definition e.g. eqn(2.1)
    
    G_f     : ndarray of shape (Nf,Np) expected
    Coh_f   : ndarray of shape (Nf,Np,Np) expected
    
    where Np is number of points and Nf is number of frequencies
    
    """
    
    S_f = []
    
    for Gv, Coh in zip(Gv_f,Coh_f):
        S_f.append(Coh * numpy.multiply(*numpy.meshgrid(Gv,Gv)))
    
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
            
            for k in range(j):
                                                
                H_jk = S[j,k]
                
                if j!=k:
                    
                    for l in range(k):
                        H_jk -= H_i[j,l]*H_i[k,l]
                    
                    H_kk = H_i[k,k]
                    H_jk /= H_kk
                    
                else:
                    
                    for l in range(k):
                        H_jk -= H_i[k,l]**2
                                        
                    H_jk **= 0.5
                        
                H_i[j,k]=H_jk
                                      
        H.append(H_i)
         
    return numpy.array(H)


def calc_fourier_coeffs(H,X):
    
    V = []
    
    for H_f, X_f in zip(H,X):
        V.append(H_f @ X_f @ numpy.ones((N,)))

    return numpy.array(V)


if __name__ == "__main__":
    
    # Define points
    z_vals = numpy.arange(1,90,10)
    N = len(z_vals) # number of grid points / correlated processes
    points = numpy.zeros((N,3))
    points[:,2]=z_vals
    
    # Define sim properties    
    t_duration = 3.0
    fs = 0.5
    dt = 1/fs
    t_vals = numpy.arange(0,t_duration+dt/2,dt)
    f_vals = numpy.fft.fftfreq(len(t_vals),dt)
    Nf = len(f_vals)
       
    # Calculate along-wind turbulences
    Gv_f, U_vals = G_vonKarman(points[:,2],f_vals)
    
    # Generate coherance matrix at each frequency
    Coh_f = coherance(points,f_vals,U_vals)
    
    # Calculate spectral density matrix at each frequency
    S_f = spectral_density_matrix(Gv_f,Coh_f)
    
    # Define random diagonal matrices at each freq
    X = white_noise_mtrx(N)
    
    # Override with test array
    S_f = []
    for f in f_vals:
        S_f.append(S_testcsv)
        
    S_f = numpy.array(S_f)
    
    # Calculate weighting matrix H
    H = weighting_mtrx(S_f)
    
    # Calculate Fourier coeffs
    V = calc_fourier_coeffs(H,X)
    
    # Calculate time series by inverse FFT
    v = numpy.fft.ifft(V,len(t_vals),axis=0)

    