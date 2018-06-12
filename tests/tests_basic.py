# -*- coding: utf-8 -*-
"""
Tests to verify key functionality of `dynsys` modules
"""

__version__ = "0.1.0"
__author__ = "Richard Hollamby, COWI UK Bridge, RIHY"

# Testing library functions
import unittest
from hypothesis import given
import hypothesis.strategies as st

# Standard imports
import numpy

# Modules to be tested
import msd_chain
import dynsys

# Define specific strategies for use with `given`
pos_floats = st.floats(min_value=0.01,max_value=100.0,allow_infinity=False,allow_nan=False)

class test_DynSys_methods(unittest.TestCase):
    """
    Class containing test functions to test `DynSys` class
    """
    
    @given(pos_floats,pos_floats,pos_floats,pos_floats,pos_floats,pos_floats)
    def test_eigprops(self,m1,m2,k1,k2,c1,c2):
        
        # Define matrices for system with constraints
        M1 = numpy.array([[m1,0,0], [0,0,0],    [0,0,m2]    ])
        K1 = numpy.array([[k1,0,0], [0,k2,-k2], [0,-k2,k2]  ])
        C1 = numpy.array([[c1,0,0], [0,c2,-c2], [0,-c2,c2]  ])
        J1 = numpy.asmatrix([1.0,-1.0,0.0])
    
        # Define system given the above matrices and compute eigenproperties
        sys1 = dynsys.DynSys(M=M1,C=C1,K=K1,J_dict={0:J1},
                             name="sys1",showMsgs=False)
        eig_props1 = sys1.CalcEigenproperties()
        s1 = eig_props1["s"]
        X1 = eig_props1["X"]
        Y1 = eig_props1["Y"]
        
        # Define matrices for merged system (i.e. constraints hard-coded)
        M2 = numpy.array([[m1,0],       [0,m2]      ])
        K2 = numpy.array([[k1+k2,-k2],  [-k2,k2]    ])
        C2 = numpy.array([[c1+c2,-c2],  [-c2,c2]    ])
        
        sys2 = dynsys.DynSys(M=M2,C=C2,K=K2,
                             name="sys2",showMsgs=False)
        eig_props2 = sys2.CalcEigenproperties()
        s2 = eig_props2["s"]
        X2 = eig_props1["X"]
        Y2 = eig_props1["Y"]
        
        # Unity-normalise eigenvectors (to allow direct comparison)
        X1 = X1 / numpy.max(numpy.abs(X1),axis=0)
        X2 = X2 / numpy.max(numpy.abs(X2),axis=0)
        Y1 = Y1 / numpy.max(numpy.abs(Y1),axis=0)
        Y2 = Y2 / numpy.max(numpy.abs(Y2),axis=0)
        
        # Check eigenproperties are the same
        s_ok = numpy.testing.assert_array_almost_equal(s1,s2)
        X_ok = numpy.testing.assert_array_almost_equal(X1,X2)
        Y_ok = numpy.testing.assert_array_almost_equal(Y1,Y2)
    
    
    @given(pos_floats,pos_floats,pos_floats,pos_floats)
    def test_undamped_2DOF(self,M1,M2,K1,K2):
        """
        Obtain angular natural frequencies for a 2 degree of freedom undamped 
        system as per the following sketch :
            
        ![2dof_undamped](../img/2dof_undamped.PNG)
        
        Expected results are obtained using the following equations, reproduced 
        from John Rees' (EJR, COWI UK Bridge) TMD lecture notes:
        [PDF](../ref/Lecture Notes - Damping and Tuned Mass Dampers (Rev. 0.7).pdf)
        
        ![angular_freq_eqns](../img/2dof_undamped_angularFreq.PNG)
        
        """
        
        # Create msd_chain system
        msd_sys = msd_chain.MSD_Chain(M_vals=[M1,M2],
                                      K_vals=[K1,K2],
                                      C_vals=[0,0],showMsgs=False)
        
        # Obtain undamped natural frequencies
        w_n = msd_sys.CalcEigenproperties()["w_n"]
        
        # Note eigenvalues are in conjugate pairs
        w1 = numpy.abs(w_n[1])
        w2 = numpy.abs(w_n[3])
        
        # Implement equations presented in EJR's note
        w1_2_bar = K1 / (M1+M2)
        w2_2_bar = K2 / M2
        mu = M2 / M1
        term1 = 0.5 * (w1_2_bar + w2_2_bar) * (1 + mu)
        term2 = 0.5 * (((w1_2_bar + w2_2_bar) * (1 + mu))**2 - 4 * w1_2_bar * w2_2_bar* (1 + mu))**0.5
        w1_expected = (term1 - term2)**0.5
        w2_expected = (term1 + term2)**0.5
        
        # Check results agree
        self.assertAlmostEqual(w1,w1_expected)
        self.assertAlmostEqual(w2,w2_expected)
    

# Run test cases
if __name__ == '__main__':
    unittest.main()

