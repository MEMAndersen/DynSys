# -*- coding: utf-8 -*-
"""
Example to demonstrate (and test) use of constraints
"""

# dynsys library imports
import modalsys
import msd_chain

# Define a modal system
my_modal_sys = modalsys.ModalSys(name="my_modal_sys")
#my_modal_sys.PrintSystemMatrices()

# Define some TMD systems
TMD1 = msd_chain.MSD_Chain(name="TMD1",
                           M_vals=[0.001,10],
                           C_vals=[0.0,0.1],
                           K_vals=[0,10])
#TMD1.PrintSystemMatrices()

TMD2 = msd_chain.MSD_Chain(name="TMD2",
                           M_vals=[0.001,20],
                           C_vals=[0.0,0.05],
                           K_vals=[0,50])
#TMD2.PrintSystemMatrices(printValues=True)

# Append TMDs to modal system
my_modal_sys.AppendSystem(child_sys=TMD1,Xpos_parent=30.0,DOF_child=0)
my_modal_sys.PrintSystemMatrices(printValues=True)

# Merge to get full system
d = my_modal_sys.GetSystemMatrices()
full_sys = d["DynSys_full"]
full_sys.PrintSystemMatrices(printValues=True)
