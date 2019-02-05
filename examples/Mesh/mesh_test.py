# -*- coding: utf-8 -*-
"""
Demonstrates functionality in mesh.py

@author: RIHY
"""


from mesh import Mesh, Node, Element


# Create blank meshes
mesh_obj1 = Mesh(name="Mesh1")
mesh_obj2 = Mesh(name="Mesh2",mesh_objs=[mesh_obj1])
mesh_obj3 = Mesh(name="Mesh3")

# Define some nodes
mesh_obj1.define_nodes(fname="mesh1_nodes.csv")
mesh_obj1.define_elements(fname="mesh1_elements.csv")
mesh_obj2.define_nodes(fname="mesh2_nodes.csv")
mesh_obj2.define_elements(fname="mesh2_elements.csv")

mesh_obj1.plot()
mesh_obj2.plot()

print(mesh_obj1.calc_extents())
print(mesh_obj2.calc_extents())
print(mesh_obj3.calc_extents())