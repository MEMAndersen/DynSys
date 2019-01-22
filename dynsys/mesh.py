"""
Class used to defined meshes, i.e. nodes interconnected by elements (lines)
"""

# Library imports
import numpy as npy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas
from itertools import count
from collections import OrderedDict


class Mesh:
    """
    Class is used to define a mesh, i.e. nodes inter-connected by elements
    
    Meshes of meshes are also permissible
    """
        
    # Class properties
    _ids = count(0)  # used to assign IDs to objects
    
    # Class methods
    
    def __init__(self,
                 name=None,
                 mesh_objs:list=[],
                 node_objs:list=[],
                 element_objs:list=[]):
        """
        Initialises mesh with sub-meshes, nodes and elements
        
        These can be passed in via dicts `mesh_objs`, `node_objs` and 
        `element_objs` or alternatively be appended once object initialised 
        using class methods provided for the purpose
        """
        
        self.id = next(self._ids)
        """
        Sequential integer ID, assigned to uniquely identify object
        """        
        
        if name is None:
            name = "Mesh %d" % self.id
            
        self.name = name
        """
        User-assigned name, usually of _string_ type
        """
        
        # Create empty dictionary attributes to contain objects within
        self.mesh_objs = OrderedDict({self.name : self})
        """
        Dict of mesh objects related to this mesh. Will always include `self` 
        but also any sub-meshes defined. Keys are names of each mesh
        """
        
        self.node_objs = OrderedDict()
        """
        Dict of node objects directly related to this mesh. Keys are names of 
        each node
        """
        
        self.element_objs = OrderedDict()
        """
        Dict of element objects directly related to this mesh. Keys are names 
        of each node
        """
        
        # Append objects passed-in via lists
        self.append_objs(mesh_objs)
        self.append_objs(element_objs)
        self.append_objs(node_objs)
        
        
    def __del__(self):
        pass
    
    def calc_extents(self):
        """
        Determine xyz extents of mesh
        """
        
        # Get list mesh objects to include in extent calculation
        mesh_list = self.get_connected_meshes(get_full_tree=False)

        # Loop over all nodes in all meshes
        node_count = 0
        for mesh_obj in mesh_list:
            
            node_list = mesh_obj.node_objs.values()
            
            for node_obj in node_list:
               
                node_count += 1                
                xyz_vals=node_obj.get_xyz()
                
                if node_count == 1:
                    xyz_min = xyz_vals
                    xyz_max = xyz_vals
                
                xyz_min = npy.minimum(xyz_min,xyz_vals)
                xyz_max = npy.maximum(xyz_max,xyz_vals)
                  
        if node_count > 0:
                    
            # Reshape into nested list
            xyz_lim = npy.vstack((xyz_min,xyz_max)).T.tolist()
            return xyz_lim
                    
        else:
            print("Warning: mesh '%s' does not have any nodes!\n" % self.name + 
                  "Could not evaluate mesh extents")
            return None
        
    
    def append_objs(self,obj_list:list):
        """
        Append objects to mesh
        """
        
        # create list in case of single object input
        if not hasattr(obj_list, '__iter__'):
            obj_list=[obj_list]

        for obj in obj_list:
          
            # Create link to this mesh object, to which object now belongs
            obj.parent_mesh=self 
    
            # Add new objects to dictionaries according to class
            class_name = obj.__class__.__name__
            
            if class_name == "Mesh":
                self.mesh_objs[obj.name] = obj
                
            elif class_name == "Element":
                self.element_objs[obj.name] = obj
                
            elif class_name == "Node":
                self.node_objs[obj.name] = obj
            
            else:
                raise ValueError("Unexpected object! Class '%s'" % class_name)
            
            
    def define_nodes(self,fname="node_defs.csv",**kwargs):
        """
        Function to define nodes based on tabular .csv data
        
        | Name | X   | Y   | Z   | 
        | ...  | ... | ... | ... | 
        """
        # Use pandas to read in data as dataframe
        df = pandas.read_csv(fname,
                             header=0,
                             index_col=None,
                             converters={'Name': str},
                             **kwargs)
        
        # Loop through rows in dataframe
        node_list = []
        for index, row in df.iterrows():
            
            # Define new node object
            new_node = Node(parent_mesh=self, name = row["Name"],
                            xyz=[row['X'],row['Y'],row['Z']])
            
            node_list.append(new_node)
            
        self.append_objs(node_list)
        return node_list
    
            
    def define_elements(self,fname="element_defs.csv",**kwargs):
        """
        Function to read element connectivity from a csv file
        
        | Name | Node1 | Node2 |
        | ...  | ...   | ...   |
        """
        # Use pandas to read in data as dataframe
        df = pandas.read_csv(fname,header=0,index_col=None,dtype=str,**kwargs)
            
        # Loop throuh rows in dataframe
        element_list = []
        for index, row in df.iterrows():
            
            # Define new node object
            node1_obj = self.node_objs[row["Node1"]]
            node2_obj = self.node_objs[row["Node2"]]
            
            new_obj = Element(parent_mesh=self,
                              name=row["Name"],
                              connected_nodes=[node1_obj,node2_obj])
            
            element_list.append(new_obj)
            
        self.append_objs(element_list)
        return element_list
    
    
    def get_connected_meshes(self,get_full_tree=True):
        """
        Returns list including all mesh objects associated with mesh object 
        
        Optional:
            
        * `get_full_tree`, _boolean_, if True then full mesh tree will be 
          returned. If False (default) then only meshes which are sub-meshes to 
          _self_ will be returned
        
        """
        
        def dfs(mesh_obj, visited=None, add_parent=get_full_tree):
            """
            Method to implement depth-first search through mesh tree
            """
        
            if visited is None:
                visited = set()
            
            visited.add(mesh_obj)
                
            # Get adjacency list for current mesh object
            connected_meshes = list(mesh_obj.mesh_objs.values())
            
            if add_parent:
                attr = 'parent_mesh'
                if hasattr(mesh_obj,attr):
                    connected_meshes.append(getattr(mesh_obj,attr))
                
            # Recursively run function if current mesh object not in visited set
            for next in set(connected_meshes) - visited:
                dfs(next, visited=visited, add_parent=True)
                
            return visited
    
        mesh_obj_list = dfs(self)
        return mesh_obj_list

    
    def plot(self,ax=None,
             set_axis_limits=True,
             axis_limits=[[-5,5],[-10,10],[0,10]]):
        """
        Plots mesh and all sub-meshes
        """
        
        print("Plotting mesh '%s'..." % self.name)
        
        # Get list of all related meshes
        mesh_list = self.get_connected_meshes(get_full_tree=False)
        
        for mesh_obj in mesh_list:
                
            ax = mesh_obj._plot_init(ax=ax)
            mesh_obj._plot_update()
        
        if set_axis_limits:
            ax.set_xlim(axis_limits[0])
            ax.set_ylim(axis_limits[1])
            ax.set_zlim(axis_limits[2])
        
        ax.legend()
        ax.set_title("%s" % self.name)
        
        return ax
    
    
    def _plot_init(self,ax=None):
        """
        Initialises mesh plot
        """
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
        
        # Define plot artists (n.b. data set in plot_update)
        self.plot_artists = {}
        
        self.plot_artists['elements'] = ax.plot([],[],[],'-',label=self.name)[0]
        self.plot_artists['nodes'] = ax.plot([],[],[],'k.')[0]
        
        return ax
    
    
    def _plot_update(self):
        """
        Updates mesh plot
        """
        
        # Plot all nodes
        
        node_list = self.node_objs.values()
        XYZ = npy.asarray([obj.xyz for obj in node_list])
        
        node_artist = self.plot_artists['nodes']
        node_artist.set_data(XYZ[:,0],XYZ[:,1])
        node_artist.set_3d_properties(XYZ[:,2])
#                
#        element_objs = self.element_objs
#        nElements = len(element_objs)
#        
#        nan = npy.full((nElements,),npy.nan)
#        
#        xyz_end1 = npy.array([element_obj.connected_nodes[0].get_xyz() 
#                             for element_obj in element_objs.values()])
#        
#        xyz_end2 = npy.array([element_obj.connected_nodes[1].get_xyz() 
#                             for element_obj in element_objs.values()])
#        
#        X = npy.ravel(npy.vstack((xyz_end1[:,0],xyz_end2[:,0],nan)).T)
#        Y = npy.ravel(npy.vstack((xyz_end1[:,1],xyz_end2[:,1],nan)).T)
#        Z = npy.ravel(npy.vstack((xyz_end1[:,2],xyz_end2[:,2],nan)).T)
#    
#        elements = self.plot_artists['elements']
#        elements.set_data(X,Y)
#        elements.set_3d_properties(Z)
#        
#        
#        
#        
        return self.plot_artists


        
class Element:
    """
    Base class used to define elements, i.e. entities connecting nodes (points in space)
    """
        
    # Class properties
    _ids = count(0)  # used to assign IDs to objects
    
    # Class methods
    
    def __init__(self, parent_mesh, connected_nodes:list, name:str=None):
        """
        Creates new element, in the context of `parent_mesh`
        """
        
        if len(connected_nodes)!=2:
            raise ValueError("len(connected_nodes)==2 expected!")
        
        self.parent_mesh = parent_mesh
        """
        Parent mesh to which element directly relates
        """
        
        self.id = next(self._ids)
        """
        Sequential integer ID, assigned to uniquely identify object
        """        
        
        if name is None:
            name = "Mesh %d" % self.id
            
        self.name = name
        """
        User-assigned name, usually of _string_ type
        """
        
        self.connected_nodes = []
        """
        List of node objects connecting to this element. Keys are node names
        """
        
        # Connect to parent mesh
        self.parent_mesh.append_objs([self])

        # Connect any nodes passed in as list
        self.connect_nodes(connected_nodes)
        
    
    def connect_nodes(self,node_objs:list):
        
        for node_obj in node_objs:

            # Append object to connected nodes list
            self.connected_nodes.append(node_obj)
            
            # Create two-way linked list
            node_obj.connect_elements(self)
            
            
    def get_connected_node(node_index:int):
        
        return 
         
# *****************************************************************************
    
class Point:
    """
    Base class used to define discrete points in space    
    """
    
    # Class properties
    _ids = count(0)  # used to assign IDs to objects
    
    def __init__(self,xyz,name:str=None):
        """
        Creates new point instance
        
        ***
        Required:
            
        * `xyz`, list or array of length 3, defines position of node
        
        ***
        Optional:
            
        * `name`, _string_ name to identify points
        
        """
        
        self.id = next(self._ids)
        """
        Sequential integer ID, assigned to uniquely identify object
        """  
        
        if name is None:
            name = "Mesh %d" % self.id
            
        self.name = name
        """
        User-assigned name, usually of _string_ type
        """
        
        self.xyz = xyz
        
    
    # ------------ATTRIBUTES ---------------
        
    @property
    def x(self):
        return self._xyz[0]
    
    def get_x(self):
        return self.x
    
    @x.setter
    def x(self,value):
        self._xyz[0] = value
        
    # -----
        
    @property
    def y(self):
        return self._xyz[1]
    
    def get_y(self):
        return self.y
    
    @y.setter
    def y(self,value):
        self._xyz[1] = value
        
    # -----
        
    @property
    def z(self):
        return self._xyz[2]
    
    def get_z(self):
        return self.z
    
    @z.setter
    def z(self,value):
        self._xyz[2] = value
        
    # -----
    
    @property
    def xyz(self):
        """
        Array of shape (3,) defining position of node
        """
        return self._xyz
    
    def get_xyz(self):
        return self.xyz
    
    @xyz.setter
    def xyz(self,value):
        value = npy.array(value)
        if value.shape != (3,):
            raise ValueError("`xyz` unexpected shape!\n" + 
                             "Shape: {0}".format(value.shape))
        self._xyz = value
        
# *****************************************************************************
        
class Node(Point):
    """
    Class used to define nodes i.e. points in space which define the 
    vertices (end points in 1D) of elements and which are thus related via a 
    mesh
    """
    
    # Class properties
    _ids = count(0)  # used to assign IDs to objects
    
    # Class methods
    
    def __init__(self,parent_mesh, xyz, name:str=None, connected_elements:list=[]):
        """
        Creates new node, in the context of `parent_mesh`
        
        ***
        Required:
            
        * `parent_mesh`, instance of _mesh_ class. Defines mesh context in 
          which node is defined
          
        * `xyz`, list or array of length 3, defines position of node
        
        ***
        Optional:
            
        * `name`, _string_ name to identify nodes
        
        * `connected_elements`, list of _element_ objects connected to this 
          node. Will normally be empty, with nodes connected to elements when 
          elements are defined
        
        """
        
        super().__init__(xyz,name)
    
        self.parent_mesh = parent_mesh
        """
        Parent mesh to which element directly relates
        """
        
        self.connected_elements = OrderedDict()
        """
        Dict of element objects connected to this node. Keys are element names
        """
        
        # Connect to parent mesh
        self.parent_mesh.append_objs([self])
        
        # Connect any elements passed in via list
        self.connect_elements(connected_elements)
        
    def connect_elements(self,element_objs:list):
        """
        Method to associate element objects with node object
        """
        
        # create list in case of single object input
        if not hasattr(element_objs, '__iter__'):
            element_objs=[element_objs]
        
        for obj in element_objs:
            self.connected_elements[obj.name]=obj
            
    

class GaussPoint(Point):
    """
    Defines gauss point, i.e. point at specific position within an element
    but not at its vertices. Gauss points have weights, which are used when 
    performing gauss quadrature (numerical integration)
    """
    pass


# ********************** FUNCTIONS ****************************************
     
def integrate_across_mesh(mesh_obj,
                          integrand_func:callable):
    """
    Function to perform gauss integration across the domain of a mesh
    
    ***
    Required:
        
    * `mesh_obj`, instance of Mesh class, defining mesh to be integrated over
    
    * `integrand_func`, _callable_ defining the integrand, i.e. the function to 
      be integrated over the mesh
          
    """  
    
    pass
            
        
            
    
# ********************** TEST ROUTINES ****************************************
        
if __name__ == "__main__":
    
    testRoutine2Run=1
    
    if testRoutine2Run==1:
        
        print("*** TEST ROUTINE 1 COMMENCED ***")
        print("")
    
        # Define new mesh
        meshObj1 = Mesh(name="myMesh")
        
        # Define some nodes
        xyz = npy.asarray([[0,0,0],[1,0,1],[2,3,4],[2,1,-1],[-0.2,-0.4,1.0]])
        Nn = xyz.shape[0]
        
        for n in range(Nn):
            Node(meshObj1,xyz=xyz[n,:],name="Node %d" % (n+1))
            
        # Plot mesh
        meshObj1.plot()
            
#        # Define some elements
#        elemTopo = npy.asarray([[0,1],[1,2],[0,2]])
#        Ne = elemTopo.shape[0]
#        
#        for e in range(Ne):
#            Element(meshObj1,elemTopo[e,:],name="Element %d" (e+1))
#            
#        # Define a mesh of meshes
#        meshObj2 = Mesh(name="myMeshOfMeshes")
#        meshObj2.appendObjs("mesh",meshObj1)
#        meshObj2.printAttrs()
#        
#        # Define deformed configuration of mesh
#        #vMask="101"
#        meshObj1.nodeObjs[0].setDeformedPos([0.2,0.1],vmask="101")
#        meshObj1.nodeObjs[1].setDeformedPos([-0.2,0.2,-0.1])
#        meshObj1.nodeObjs[2].setDeformedPos([0.2,0.5],vmask="110")
#        meshObj1.nodeObjs[3].setDeformedPos([-0.2,-0.4,0.1])
#        
#        # Produce 3D plot of mesh
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        meshObj2.plot(ax,plotDeformed=True)
#        plt.show()
##        
#    elif testRoutine2Run==2:
#        
#        print("*** TEST ROUTINE 2 COMMENCED ***")
#        print("")
#        
#        # Define new mesh, reading data from file
#        meshObj3 = mesh(name="myCoolMesh")
#        meshObj3.defineFromFiles()
#        
#        # Read deformations from file
#        meshObj3.readDeformations("Step2")
#        
#        # Produce 3D plot of mesh
#        fig = plt.figure(figsize=(9,9))
#        ax = fig.gca(projection='3d')
#        dwgObjs=meshObj3.plot(ax,printOutput=False,plotDeformed=True)
#        plt.show()
#        
#        # Read deformations from file and update plot
#        meshObj3.readDeformations("Step3")
#        dwgObjs=meshObj3.plot(ax,updatePlot=True,dwgObjs=dwgObjs,printOutput=False,plotDeformed=True)
#        
#    elif testRoutine2Run==3:
#        
#        print("*** TEST ROUTINE 2 COMMENCED ***")
#        print("")
#        
#        # Define new mesh, reading data from file
#        meshObj1 = mesh(name="myCoolMesh")
#        meshObj1.defineFromFiles()
#        
#        # Create new figure with 3D axes system
#        fig = plt.figure(figsize=(9,9))
#        ax = fig.gca(projection='3d')
#        
#        # Set appropriate limits for plot
#        xyz_min,xyz_max=[-10,-10,0],[10,10,10]
#        ax.set_xlim([xyz_min[0],xyz_max[0]])
#        ax.set_ylim([xyz_min[1],xyz_max[1]])
#        ax.set_zlim([xyz_min[2],xyz_max[2]])
#        
#         # Set axes labels etc.
#        ax.set_xlabel("x")
#        ax.set_ylabel("y")
#        ax.set_zlabel("z")
#        ax.set_title("My mesh animation")
#        
#        # Create drawing objects (with no data)
#        dwgObjs=meshObj1.plot(ax,
#                              printOutput=False,plotNull=True,
#                              plotDeformed=True,
#                              plotNodes=False,plotElements=True,
#                              elementLineStyle_undeformed='--',
#                              elementLineStyle_deformed='-',
#                              nodeStyle='.',nodeColor='m')
#        time_template = 'Time = %.3fs'
#        time_text = ax.text(0.05, 0.9, 0,'', transform=ax.transAxes)
#        
#        # Define function used to draw a clear frame
#        def init():
#            return animate(0,dt,dwgObjs,time_text)
#            
#        # Define function to run repeatedly to create animation
#        def animate(i,dt,dwgObjs,time_text):
#            
#            t_val=i*dt
#            
#            # Read deformations from file
#            meshObj1.readDeformations("Step{0}".format(i+1),printOutput=False)
#        
#            # Update plot
#            dwgObjs=meshObj1.plot(ax,
#                                  updatePlot=True,dwgObjs=dwgObjs,
#                                  printOutput=False,plotDeformed=True,
#                                  plotNodes=False,plotElements=True)
#            
#            # Update time lobel
#            time_text.set_text(time_template % t_val)
#        
#            return dwgObjs,time_text
#        
#        dt=1
#        delay=2
#        ani = animation.FuncAnimation(fig, animate,
#                                      frames=3,
#                                      fargs=(dt,dwgObjs,time_text,),
#                                      interval=dt*1000,
#                                      repeat=True,repeat_delay=delay*1000,
#                                      init_func=init)
#        plt.show()
    
    else:
        print("(No valid test routine selected)")
        
   

    
    
    