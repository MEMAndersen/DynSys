# -*- coding: utf-8 -*-
"""
Example to demonstrate usage of (and assist development of) gust buffeting 
analysis class

@author: RIHY
"""

import wind_env
import wind_response
from wind_section import read_wind_sections
import nodle
import modalsys

mesh_obj = nodle.read_mesh(fname='EMLEY_MOOR.xlsx')



sys_obj = modalsys.ModalSys(name='Emley Moor Tower',
                            fname_modalParams='modal_params.csv',
                            fname_modeshapes='modeshapes.csv',
                            mesh_obj=mesh_obj)

wind_obj = wind_env.WindEnv_equilibrium(V_ref=25.0,
                                    z_ref=10.0,
                                    phi=53.0,
                                    direction=30.0,
                                    z0=0.03)

gb_analysis = wind_response.Buffeting(sys_obj,wind_obj)
ws_dict = read_wind_sections('wind_sections.csv')
#assignment_dict = read_wind_section_assignments('wind_section_assignments.csv')
#mesh_obj.assign_wind_sections(ws_dict)

gb_analysis.plot()
