# -*- coding: utf-8 -*-
"""
Example to demonstrate (and test) use of constraints
"""

# dynsys library imports
import modalsys
import msd_chain
import dyn_analysis
import loading

# Define a modal system
my_modal_sys = modalsys.ModalSys(name="my_modal_sys")
my_modal_sys.AddOutputMtrx(fName='outputs.csv')

# Define some TMD systems
TMD1 = msd_chain.MSD_Chain(name="TMD1",
                           M_vals=[0.001,10],
                           C_vals=[0.0,0.1],
                           K_vals=[0,10])
print(TMD1.output_mtrx.shape)
TMD1.AddOutputMtrx(output_mtrx=[-1,1,0,0,0,0],output_names=["Relative disp [m]"])

TMD2 = msd_chain.MSD_Chain(name="TMD2",
                           M_vals=[0.001,20],
                           C_vals=[0.0,0.05],
                           K_vals=[0,50])
TMD2.AddOutputMtrx(output_mtrx=[-1,1,0,0,0,0],output_names=["Relative disp [m]"])

# Append TMDs to modal system
my_modal_sys.AppendSystem(child_sys=TMD1,Xpos_parent=30.0,DOF_child=0)
my_modal_sys.AppendSystem(child_sys=TMD2,Xpos_parent=50.0,DOF_child=0)
my_modal_sys.PrintSystemMatrices(printValues=True)

# Plot modeshapes
my_modal_sys.PlotModeshapes()

# Define basic moving load
loading_obj = loading.LoadTrain(loadX=[0.0],loadVals=[100.0],name="test load")

# Carry out moving load analysis
analysis_obj = dyn_analysis.MovingLoadAnalysis(modalsys_obj=my_modal_sys,
                                               name="test analysis",
                                               loadtrain_obj=loading_obj,
                                               dt=0.01
                                               )
results_obj = analysis_obj.run()

results_obj.PlotResults(dynsys_obj=my_modal_sys)
results_obj.PrintResponseStats()

