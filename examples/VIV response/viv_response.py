# -*- coding: utf-8 -*-
"""
Example to demonstrate usage of (and assist development of) vortex-induced
vibrations analysis

@author: RIHY
"""

import wind_env
import wind_response 
import nodle
import modalsys

# Define mesh of system
mesh = nodle.read_mesh(fname='EMLEY_MOOR.xlsx')

# Define wind sections


# Define modal system, assigning the above mesh
sys = modalsys.ModalSys(name='Emley Moor Tower',
                        fname_modalParams='modal_params.csv',
                        fname_modeshapes='modeshapes.csv',
                        mesh_obj=mesh)

wind_obj = wind_env.WindEnv_equilibrium(V_ref=25.0,
                                        z_ref=10.0,
                                        phi=53,
                                        z0=0.03)

analysis_obj = wind_response.VIV(sys,wind_obj)



analysis_obj.plot()
