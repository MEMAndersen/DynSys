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

# Modules to be tested
import msd_chain

# Define specific strategies for use with `given`
pos_floats = st.floats(min_value=0.01,max_value=100.0,allow_infinity=False,allow_nan=False)

class test_DynSys_methods(unittest.TestCase):
    """
    Class containing test functions to test `DynSys` class
    """
    
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
                                      C_vals=[0,0])
        
        # Obtain undamped natural frequencies
        w_n = msd_sys.CalcEigenproperties()["w_n"]
        
        # Note eigenvalues are in conjugate pairs
        w1 = w_n[1]
        w2 = w_n[3]
        
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

