
# coding: utf-8

"""
Validation script to demonstrate multiple modes + multiple TMDs analysis method

Comparison is made against results and figures presented in the following 
paper: 
[PDF](.../references/The Lateral Dynamic Stablity of Stockton Infinity Footbridge using Complex Modes.pdf)    
"""

__author__ = "Richard Hollamby, COWI UK Bridge, RIHY"

# Standard imports
import numpy
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy

# DynSys imports
import modalsys
import damper

#%%

# Define modal system
bridge_sys = modalsys.ModalSys(name="Stockton Infinity Footbridge",
                               fname_modeshapes='deck_modeshapes.csv',
                               fname_modalParams='modal_params.csv')

#%%

# Define TMD systems and append to bridge system
damper.append_TMDs(modal_sys=bridge_sys,
                   fname="TMD_defs.csv",
                   append=True,
                   verbose=True)

bridge_sys.PlotModeshapes()

#%%

def pedestrian_damping(N,L=None,cp=300):
    """
    cp : damping per pedestrian (Ns/m). 300Ns/m is value quoted in paper
    N  : number of pedestrians on bridge
    L  : length of bridge (m)
    """
    
    # Evaluate integral of modeshape-squared along bridge    
    modal_integral = bridge_sys.CalcModeshapeIntegral(power=2)
    
    if L is None:
        L = bridge_sys.Ltrack
    
    # Determine damping due to pedestrians per unit length
    pedestrian_damping = cp*N/L
    
    # Determine modal damping (nb negative usually!)
    modal_damping = - pedestrian_damping * modal_integral
    
    # Return as diagonal matrix
    return numpy.diag(modal_damping)
    
eta_vals = []
fn_vals = []

C_bridge = bridge_sys._C_mtrx

for N in range(0,2000,10):
    
    # Reset bridge intrinsic damping matrix at each iteration
    bridge_sys._C_mtrx = C_bridge
    
    # Calculate change in modal damping due to pedestrians
    C_pa = pedestrian_damping(N=100)
    
    # Adjust bridge damping matrix
    bridge_sys._C_mtrx += C_pa
    
    # Carry out eigevalue analysis
    eig_props = bridge_sys.CalcEigenproperties()
    eta_vals.append(eig_props["eta"])
    fn_vals.append(eig_props["f_n"])
    
fig, ax = plt.subplots()
ax.plot(eta_vals,fn_vals,'.')
ax.set_xlim([-0.02,0.22])
ax.set_ylim([0.8,1.05])


#%%
#    
#cp = 300               # pedestrian force (taken as invariant of freq)
#L = 180                # deck length
#dL = 180 / (Nd-1)      # spacing between deck nodes
#
#C_pa = cp * dL / L * modeshape_TMD.T * modeshape_TMD
#
#C_p1 = npy.hstack((C_pa,npy.asmatrix(npy.zeros((Nm,N_T)))))
#C_p2 = npy.asmatrix(npy.zeros((N_T,Nm+N_T)))
#C_p = npy.vstack((C_p1,C_p2))
#
#
## In[99]:
#
#
#def DefineSystemMatrix(Np):
#
#    # Define system state matrix A
#    nDOF = Nm + N_T
#
#    # Define inverse mass matrix
#    M_aT_inv = LA.inv(M_aT)
#    
#    # Define effective damping matrix
#    C_eff = C_aT - Np * C_p
#
#    # Assemble system matrix A
#    
#    A11 = npy.asmatrix(npy.zeros((nDOF,nDOF)))
#    A12 = npy.asmatrix(npy.identity(nDOF))
#    A1 = npy.hstack((A11,A12))
#
#    A21 = - M_aT_inv * K_aT
#    A22 = - M_aT_inv * C_eff
#    A2 = npy.hstack((A21,A22))
#
#    A = npy.vstack((A1,A2))
#
#    return A
#
#
## In[105]:
#
#
#Np = npy.arange(start=0,stop=5000,step=30)    # defines pedestrian numbers to try
#
#
#for n in range(0,Np.shape[0]):
#    
#    # Derive system matrix
#    A = DefineSystemMatrix(Np[n])
#    
#    # Eigenvalue decomposition
#    s,u = LA.eig(A)
#    
#    # Plot eigenvalues on complex plane
#    if n == 0:
#        
#        gs = gridspec.GridSpec(1, 2)
#        fig = plt.figure(num=1)
#        fig.set_size_inches(16, 7)
#        
#        ax1 = fig.add_subplot(gs[0, 0])
#        
#        ax1.plot(npy.real(s),npy.imag(s),"bo")
#        ax1.set_xlabel("Real(eig(A))")
#        ax1.set_ylabel("Imag(eig(A))")
#        ax1.set_title("Eigenvalues of A (%d modes, %d TMDs)" % (Nm, N_T))
#        ax1.axvline(0,color = 'r')
#        
#    else:
#        
#        ax1.plot(npy.real(s),npy.imag(s),"k.")
#        
#    # Calculate effective damping and damped natural freq
#    eta_eff = - npy.real(s) / npy.absolute(s)
#    omega_d = npy.imag(s)
#    f_d = omega_d / (2 * npy.pi)
#    
#    # Plot damping vs freq
#    if n == 0:
#        
#        ax2 = fig.add_subplot(gs[0, 1])
#        
#        ax2.plot(eta_eff,f_d,"bo")
#        ax2.set_xlabel("Effective damping ratio")
#        ax2.set_ylabel("Damped natural frequency (Hz)")
#        ax2.set_title("Frequency vs effective damping (%d modes, %d TMDs)" % (Nm, N_T))
#        ax2.axvline(0,color = 'r')
#        ax2.set_ylim([0.8,1.05])
#        
#    else:
#        
#        ax2.plot(eta_eff,f_d,"k.")
#    
    

