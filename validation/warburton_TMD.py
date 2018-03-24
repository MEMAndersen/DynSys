# -*- coding: utf-8 -*-
"""
Validation script to compare frequency response obtained using `dynsys` 
routines against Warburton and Ayorinde's 1980 derivation concerning a 
single damped mode with a single damped TMD: 
[PDF](.../references/Warburton and Ayorinde (1980).pdf)
"""

__author__ = "Richard Hollamby, COWI UK Bridge, RIHY"

# Modules to be tested
import msd_chain
from dynsys import PlotFrequencyResponse


if __name__=="__main__":
    
    fmax=2.0
    
    # Define two-mass system using Warburtons notation
    d, main_sys_with_TMD, f, G_f = msd_chain.warburton_TMD(fmax=fmax)
    plot_dict = PlotFrequencyResponse(f,G_f,
                                      label_str="per Warburton & Ayorinde eqn(1)")
    
    
    
    # Obtain frequency response using CalcFreqResponse()
    f2 , G_f2 = main_sys_with_TMD.CalcFreqResponse(fmax=fmax)
    
    # Overlay to compare frequency response
    plot_dict = PlotFrequencyResponse(f2,G_f2[0,0,:],
                                      label_str="using CalcFreqResponse()",
                                      ax_magnitude=plot_dict["ax_magnitude"],
                                      ax_phase=plot_dict["ax_phase"])
    
    # Edit line styles
    lines2edit = [plot_dict["ax_magnitude"].lines[1],
                  plot_dict["ax_phase"].lines[1]]
    
    for line in lines2edit:
        line.set_linestyle("--")
    
    
    
    # Obtain frequency response for main mass only
    m_M = d["m_M"]
    k_M = d["k_M"]
    c_M = d["c_M"]
    main_sys = msd_chain.MSD_Chain(M_vals = m_M,
                                   K_vals = k_M,
                                   C_vals = c_M)
    f3 , G_f3 = main_sys.CalcFreqResponse(fmax=fmax)
    
    # Overlay to compare frequency response
    PlotFrequencyResponse(f3,G_f3[0,0,:],
                          label_str="using CalcFreqResponse(), no TMD",
                          ax_magnitude=plot_dict["ax_magnitude"],
                          ax_phase=plot_dict["ax_phase"])
    
    
    
                          
    
