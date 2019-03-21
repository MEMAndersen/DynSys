# -*- coding: utf-8 -*-
"""
Example to demonstrate usage of (and assist development of) gust buffeting 
analysis class

@author: RIHY
"""

import wind 
import nodle
import modalsys

mesh = nodle.read_mesh(fname='EMLEY_MOOR.xlsx')


sys = modalsys.ModalSys(name='Emley Moor Tower',
                        fname_modalParams='modal_params.csv',
                        fname_modeshapes='modeshapes.csv',
                        mesh_obj=mesh)

wind_env = wind.WindEnv_equilibrium(V_ref=25.0,
                                    z_ref=10.0,
                                    z0=0.03)

gb_analysis = wind.BuffetingAnalysis(sys,wind_env)

gb_analysis.plot()
