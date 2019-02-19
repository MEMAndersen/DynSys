"""
Class used to defined meshes, i.e. nodes interconnected by elements (lines)
"""

# Library imports
import numpy as npy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas
from itertools import count
from inspect import getmro
from numpy.linalg import norm
from common import check_class, set_equal_aspect_3d, rotate_about_axis

vertical_direction = npy.array([0.0,0.0,1.0])
default_y_direction = npy.array([0.0,1.0,0.0])


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
        self.mesh_objs = {self.name : self}
        """
        Dict of mesh objects related to this mesh. Will always include `self` 
        but also any sub-meshes defined. Keys are names of each mesh
        """
        
        self.node_objs = {}
        """
        Dict of node objects directly related to this mesh. Keys are names of 
        each node
        """
        
        self.element_objs = {}
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
            class_name_list = getmro(obj.__class__)
            
            if Mesh in class_name_list:
                self.mesh_objs[obj.name] = obj
                
            elif Element in class_name_list:
                self.element_objs[obj.name] = obj
                
            elif Node in class_name_list:
                self.node_objs[obj.name] = obj
            
            else:
                raise ValueError("Unexpected object!\n\n" + 
                                 "Class list{0}".format(class_name_list))
            
            
    def define_nodes(self,df=None,fname='nodes.csv',**kwargs):
        """
        Function to define nodes from tabulated input
        
        ***
        Pandas Dataframe containing the following columns to be provided:
        
        | Node | X   | Y   | Z   | 
        | ...  | ... | ... | ... | 
        
        or alternatively `fName` defines .csv file containing similar input
        """
        # Use pandas to read in data as dataframe
        if df is None:
            df = pandas.read_csv(fname,
                                 header=0,
                                 names=['Node','X','Y','Z'],
                                 index_col=None,
                                 converters={'Node': int},
                                 **kwargs)
        
        # Loop through rows in dataframe
        node_list = []
        for index, row in df.iterrows():
            
            # Define new node object
            new_node = Node(parent_mesh=self, name = row["Node"],
                            xyz=[row['X'],row['Y'],row['Z']])
            
            node_list.append(new_node)
            
        self.append_objs(node_list)
        return node_list
    
            
    def define_line_elements(self,df=None,fname="elements.csv",**kwargs):
        """
        Function to read element connectivity from tabulated input
        
        ***
        Pandas Dataframe containing the following columns to be provided:
        
        | Member | EndJ  | EndK  |
        | ...    | ...   | ...   |
        
        or alternatively `fName` defines .csv file containing similar input
        """
        # Use pandas to read in data as dataframe
        if df is None:
            df = pandas.read_csv(fname,header=0,index_col=None,dtype=str,
                                 **kwargs)
            
        # Loop throuh rows in dataframe
        element_list = []
        for index, row in df.iterrows():
            
            # Get node objects for each end
            node1_obj = self.node_objs[row["EndJ"]]
            node2_obj = self.node_objs[row["EndK"]]
            
            # Define new line element
            new_obj = LineElement(parent_mesh=self,
                              name=row["Member"],
                              connected_nodes=[node1_obj,node2_obj])
            
            element_list.append(new_obj)
            
        self.append_objs(element_list)
        return element_list
    
    
    def define_gauss_points(self,N_gp=2,verbose=False):
        """
        Loops over all elements in mesh, defining gauss points
        """
        
        element_list = self.element_objs.values()
        
        if isinstance(N_gp, list):
            if len(N_gp)!=len(element_list):
                raise ValueError("Length of `N_gp` list does not agree " + 
                                 "with total number of elements")
                
        elif not isinstance(N_gp,int):
            raise ValueError("`N_gp` to be either `int` or `list`")
                
                        
        for i, element_obj in enumerate(element_list):
            
            # Determine number of gauss points required for this element
            if isinstance(N_gp, list):
                n = N_gp[i]
                
            else:
                n = N_gp
                
            # Define gps
            element_obj.define_gauss_points(N_gp=n, verbose=verbose)
    
    
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


    def get_gauss_points(self):
        """
        Returns all nested list to describe all gauss points associated with 
        mesh via its elements
        """
        gp_list = []
        for e in self.element_objs.values():
            gp_list.append(e.gauss_points)
        return gp_list

    
    def plot(self,ax=None,
             set_axis_limits=True,
             set_square_axes=True,
             **kwargs):
        """
        Plots mesh and all sub-meshes
        """
        
        print("Plotting mesh '%s'..." % self.name)
        
        # Get list of all related meshes
        mesh_list = self.get_connected_meshes(get_full_tree=False)
        
        for mesh_obj in mesh_list:
                
            ax = mesh_obj._plot_init(ax=ax,**kwargs)
            mesh_obj._plot_update()
        
        if set_axis_limits:
            
            axis_limits = self.calc_extents()
                            
            if not set_square_axes:
                ax.set_xlim(axis_limits[0])
                ax.set_ylim(axis_limits[1])
                ax.set_zlim(axis_limits[2])
                
            else:
                
                # Draw hidden line to ensure proper scaling of axes
                set_equal_aspect_3d(ax,axis_limits)
                
        
        ax.legend()
        ax.set_title("%s" % self.name)
        
        return ax
    
    
    def _plot_init(self,ax=None,plot_gps=False,plot_axes=False):
        """
        Initialises mesh plot
        """
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            
        self.ax = ax
        
        # Define plot artists (n.b. data set in plot_update)
        self.plot_artists = {}
        
        self.plot_artists['elements'] = ax.plot([],[],[],'-',label=self.name)[0]
        self.plot_artists['nodes'] = ax.plot([],[],[],'k.')[0]
        
        if plot_gps:
            self.plot_artists['gauss_points'] = ax.plot([],[],[],'g.')[0]
            
        if plot_axes: 
            self.plot_artists['x_axes'] = ax.quiver([],[],[],
                                                     [],[],[],
                                                     linewidth=1.0,
                                                     color='r')
            
            self.plot_artists['y_axes'] = ax.quiver([],[],[],
                                                     [],[],[],
                                                     linewidth=1.0,
                                                     color='g')
            
            self.plot_artists['z_axes'] = ax.quiver([],[],[],
                                                     [],[],[],
                                                     linewidth=1.0,
                                                     color='b')
        
        return ax
    
    
    def _plot_update(self):
        """
        Updates mesh plot
        """
        
        # Plot all nodes
        if self.has_nodes():
            
            node_list = self.node_objs.values()
            XYZ = npy.asarray([obj.xyz for obj in node_list])
            
            node_artist = self.plot_artists['nodes']
            node_artist.set_data(XYZ[:,0],XYZ[:,1])
            node_artist.set_3d_properties(XYZ[:,2])
        
        # Plot all elements
        if self.has_elements():
            
            element_list = self.element_objs.values()
            
            xyz_end1 = npy.array([element_obj.connected_nodes[0].get_xyz() 
                                 for element_obj in element_list])
            
            xyz_end2 = npy.array([element_obj.connected_nodes[1].get_xyz() 
                                 for element_obj in element_list])
        
            nan = npy.full((len(element_list),),npy.nan) # used to create gaps
            
            X = npy.ravel(npy.vstack((xyz_end1[:,0],xyz_end2[:,0],nan)).T)
            Y = npy.ravel(npy.vstack((xyz_end1[:,1],xyz_end2[:,1],nan)).T)
            Z = npy.ravel(npy.vstack((xyz_end1[:,2],xyz_end2[:,2],nan)).T)
        
            elements = self.plot_artists['elements']
            elements.set_data(X,Y)
            elements.set_3d_properties(Z)
            elements.set_alpha(0.5)
        
        # Plot gauss points
        attr = 'gauss_points'
        if attr in self.plot_artists:
            
            gp_artist = self.plot_artists[attr]
            
            gp_XYZ = npy.array([gp.xyz for e in element_list
                                for gp in e.gauss_points])
        
            gp_artist.set_data(gp_XYZ[:,0],gp_XYZ[:,1])
            gp_artist.set_3d_properties(gp_XYZ[:,2])
            
        # Plot local axes
        attr = 'x_axes'
        if attr in self.plot_artists:
            
            x_axes = self.plot_artists['x_axes']
            y_axes = self.plot_artists['y_axes']
            z_axes = self.plot_artists['z_axes']
            
            element_list = self.element_objs.values()
            
            origin = npy.array([e.get_midpoint_xyz() for e in element_list])
            
            axes = npy.array([e.get_axes() for e in element_list])
            ex = axes[:,0,:]
            ey = axes[:,1,:]
            ez = axes[:,2,:]
            
            sf = 0.4 # scale factor for element axes
            x_axes.set_segments([[a,a+b] for a,b in zip(origin,sf*ex)])
            y_axes.set_segments([[a,a+b] for a,b in zip(origin,sf*ey)])
            z_axes.set_segments([[a,a+b] for a,b in zip(origin,sf*ez)])
        
        
        return self.plot_artists


    def has_nodes(self):
        return len(self.node_objs)>=1
    
    def has_elements(self):
        return len(self.element_objs)>=1
    
    
class MeshChain(Mesh):
    """
    Class to implement a chain mesh, i.e. series of elements forming a chain
    """
    
    def __init__(self,node_list,**kwargs):
        """
        Required:
            
        * `node_list`, list of Node objects defining chain
        
        Other keyword arguments are passed to `Mesh.__init__()` method
        """
        
        # Define elements linking nodes
        
        nNodes = len(node_list)
            
        if nNodes < 2:
            raise ValueError("Cannot define mesh chain from only 1 node!")
           
        element_list = []
        for i in range(nNodes-1):
            
            node1 = node_list[i]
            node2 = node_list[i+1]
            
            element_list.append(LineElement(parent_mesh=None, # append later
                                            connected_nodes=[node1,node2]))
        
        super().__init__(node_objs=node_list,
                         element_objs=element_list,
                         **kwargs)
    
        
class Element:
    """
    Base class used to define elements, i.e. entities connecting nodes 
    (points in space)
    
    _Note: implemented as an abstract class. Objects cannot be instantiated_
    """
        
    # Class properties
    _ids = count(0)  # used to assign IDs to objects
    
    _nNodes_expected = None
    
    # Class methods
    
    def __init__(self, parent_mesh, connected_nodes:list, name:str=None):
        """
        Creates new element, in the context of `parent_mesh`
        """
        
        # Prevent direct instatiation of this class
        if type(self) == Element:
            raise Exception("<Element> must be subclassed.")
                
        self.parent_mesh = parent_mesh
        """
        Parent mesh to which element directly relates
        """
        
        self.id = next(self._ids)
        """
        Sequential integer ID, assigned to uniquely identify object
        """        
        
        if name is None:
            name = "%s %d" % (self.__class__.__name__,self.id)
            
        self.name = name
        """
        User-assigned name, usually of _string_ type
        """
        
        self.connected_nodes = []
        """
        List of node objects connecting to this element. Keys are node names
        """
        
        self.gauss_points = []
        """
        List of gauss point objects associated with this element
        """
        
        # Connect to parent mesh
        if parent_mesh is not None:
            self.parent_mesh.append_objs([self])

        # Connect any nodes passed in as list
        self.connect_nodes(connected_nodes)
        
    
    # -----
          
    
    def connect_nodes(self,node_objs:list):
        
        self._check_node_objs(node_objs)
        
        for node_obj in node_objs:
            
            check_class(node_obj,Node)

            # Append object to connected nodes list
            self.connected_nodes.append(node_obj)
            
            # Create two-way linked list
            node_obj.connect_elements(self)
            
            
    def _check_node_objs(self,node_objs):
        
        N = self._nNodes_expected
        
        if N is not None:
            
            if len(node_objs)!= N:
                raise ValueError("Unexpected number of nodes supplied "
                                 "to define element!")
                
            
    def get_connected_node(node_index:int):
        
        return 
    
    
    
    
    
    def define_gauss_points(self):
        """
        Defines gauss points to be associated with element
        
        _To be overridden by derived class method_
        """
        raise ValueError("Not implemented for base class! To be overridden")
        
    
    
class LineElement(Element):
    """
    Class to implement line elements, i.e. element between two nodes in 3d
    """
    
    _nNodes_expected = 2


    def __init__(self,parent_mesh,connected_nodes:list,
                 skew_angle=0.0,**kwargs):
        
        self.skew_angle = skew_angle
        """
        Clockwise angle [radians] by which the local y- and -z axes of the 
        element are rotated, about the local x-axis.
        """
        
        super().__init__(parent_mesh, connected_nodes,**kwargs)
        
    
    @property
    def wind_sections(self):
        """
        Object list (len=2) of `WindSection` class instances, which 
        define aerodynamic properties of element at each end.
        """
        return [self._wind_section_end1, self._wind_section_end2]
    
    
    def define_wind_sections(self,obj_list):
        """
        Object or object list (len=2) of `WindSection` class instances, which 
        define the aerodynamic properties of the line element.
            
        If an object is provided, section is considered 'uniform', i.e. has 
        the same properties at both ends. Alternatively, if an object list is 
        provided, this must be of length=2 and consist of a pair of objects 
        to define wind section at each end of the element
        """
        # Duplicate WindSection object to define uniform cross-secton
        if not isinstance(obj_list,list):
            obj = obj_list
            obj_list = [obj,obj]
            
        self._wind_section_end1 = obj_list[0]
        self._wind_section_end2 = obj_list[1]
            

    def get_axes(self,verbose=False):
        """
        Returns list of (3,) arrays defining unit vectors for each element axis
        """
        
        # Calculate x-axis, taking as being direction vector from end1 to end2
        r1, r2 = self.get_end_positions()
        x = r2 - r1
        x /= norm(x)
        
        # Calculate y-axis
        # y to lie in horizontal plane orthogonal to x
        y = npy.cross(vertical_direction,x)
        
        if norm(y)==0:
            # Special case for vertical members
            if verbose:
                print("Element '%s' is vertical" % self.name)
            y = default_y_direction
            
        # Calculate z-axis to be orthogonal to x and y
        z = npy.cross(x,y)
        
        # Rotate axes if required
        phi = self.skew_angle
        if phi != 0.0:
            y = rotate_about_axis(y,x,phi)
            z = rotate_about_axis(z,x,phi)
    
        return x, y, z
        
        
    def get_end_positions(self):
        """
        Returns position vectors of each end of elements
        """
        r1 = self.connected_nodes[0].get_xyz()
        r2 = self.connected_nodes[1].get_xyz()
        return r1, r2
    
    
    def length(self):
        """
        Returns length of element i.e. distance between end nodes
        """
        r1, r2 = self.get_end_positions()
        r12 = r2 - r1
        return sum(r12**2)**0.5
    
    
    def get_midpoint_xyz(self):
        """
        Returns position vector denoting the midpoint of element
        """
        r1, r2 = self.get_end_positions()
        return 0.5*(r1+r2)

    
    
    def define_gauss_points(self,N_gp:int=3,verbose=False):
        """
        Defines gauss points to be associated with element
        
        ***
        Required:
            
        * `N_gp`, integer, defines number of gauss points to be defined per 
          element. This relates to accuracy of integration.
          
        """
        # Define weights and locations for gauss points based on [-1,1] domain
        # Refer https://en.wikipedia.org/wiki/Gaussian_quadrature
        
        if N_gp == 1:
            
            locs = [0.0]
            weights = [2.0]
        
        elif N_gp == 2:
            
            val = (1/3)**(0.5)
            locs = [-val, +val]
            del val
            weights = [1.0,1.0]
            
        elif N_gp == 3:
            
            val = (3/5)**(0.5)
            locs = [-val, 0.0, +val]
            del val
            weights = [(5/9),(8/9),(5/9)]
            
        else:
            raise ValueError("N_gp>3 not yet implemented!")
            
        # Convert to numpy arrays
        locs = npy.array(locs)
        weights = npy.array(weights)
        
        if verbose:
            print("Gauss points for element '%s'\n" % self.name + 
                  "Locs: {0}\nWeights:{1}".format(locs,weights))
        
        # Convert to [0,L] domain
        L = self.length()
        weights *= L/2
        
        # Define gauss points
        locs = (locs + 1)/2                     # convert to [0,1] domain
        r1, r2 = self.get_end_positions()
        r12 = r2 - r1
            
        self.gauss_points = []
        for (loc,weight) in zip(locs,weights):
            
            # Calculate position of gauss point
            xyz = r1 + loc * r12
            
            # Define new GaussPoint object and append to list
            self.gauss_points.append(GaussPoint(loc=loc,weight=weight,xyz=xyz))
            
        return self.gauss_points
         
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
            name = "%s %d" % (self.__class__.__name__,self.id)
            
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
        
        self.connected_elements = {}
        """
        Dict of element objects connected to this node. Keys are element names
        """
        
        self.lcase_disp = DispResults(node_obj=self,results_arr=None)
        """
        Instance of `DispResults` class used as container for 
        displacements as obtained for ordinary loadcases / combinations
        """
        
        self.modal_disp = DispResults(node_obj=self,results_arr=None)
        """
        Instance of `DispResults` class used as container for 
        mode displacements as obtained from eigenvalue analysis
        """
        
        # Connect to parent mesh
        if self.parent_mesh is not None:
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
            
            check_class(obj,Element)
            self.connected_elements[obj.name]=obj
            
    

class GaussPoint(Point):
    """
    Defines gauss point, i.e. point at specific position within an element
    but not at its vertices. Gauss points have weights, which are used when 
    performing gauss quadrature (numerical integration)
    """
    
    # Class properties
    _ids = count(0)  # used to assign IDs to objects
    
    def __init__(self,loc,weight,**kwargs):
               
        self.loc = loc
        """
        Location of gauss point within [0,1] domain along element to which 
        gauss point relates (useful for interpolation of properties / results 
        found at the end of the element)
        """
        
        self.weight = weight
        """
        Weight associated with gauss point, for use when carrying out 
        integration by gauss quadrature
        """
        
        super().__init__(**kwargs)
      
        
# ------------------- CLASSES TO STORE MESH RESULTS -----------------------
        
class MeshResults():
    """
    Base class used to store results with relation to mesh objects
    (e.g. nodes for displacements, member-node locations for forces)
    """
    
    nComponents = None   # integer defining expected number of components
    
    def __init__(self, obj_list, results_arr=None, verbose=False):
        
        self.obj_list = obj_list
        """
        List of mesh objects to which results relate. E.g. displacements 
        relate to nodes (i.e. list length = 1), whereas forces relate to 
        elements and nodes (i.e. list of length = 2)
        """
        
        self.clear()
        
        if results_arr is not None:
            self.add(results_arr)
        
        if verbose:
            print("New instance of '%s' created" % self.__class__.__name__)
            
            
    def __repr__(self):
        
        print_str = ""
        print_str += "Class:\t%s\n" % self.__class__.__name__
        print_str += "Name:\t%s\n" % self.name
        print_str += "Shape of values array:\n{0}".format(self.values.shape)
        print_str += "\n"
        return print_str
    
        
    @property
    def values(self):
        """
        2d-array of vector results for multiple loadcases. 
        Shape is [nLoadcase,nComponents]
        """
        return self._values
    
    
    @property
    def name(self):
        """
        Returns list comprising names of mesh objects associated
        """
        return [x.name for x in self.obj_list]
       
        
    def add(self,new_results,verbose=False):
        """
        Appends results for new loadcase
        """
        if verbose:
            print("Appending new results to location {0}".format(self.name))
            
        self._values = npy.vstack((self.values, new_results))
        
        
    def clear(self):
        """
        Clears results held within object
        """
        self._values = npy.zeros((0,self.nComponents))
        
        
    
        
        
    
class DispResults(MeshResults):
    """
    Class used to store displacement results, with relation to a given node
    """
    
    nComponents = 6
    
    def __init__(self, node_obj, results_arr=None):
        
        super().__init__([node_obj], results_arr)
        
        
class ReactionResults(MeshResults):
    """
    Class used to store reaction results, with relation to a given node
    """
    
    nComponents = 6
    
    def __init__(self, node_obj, results_arr=None):
        
        super().__init__([node_obj], results_arr)
        
        
class ForceResults(MeshResults):
    """
    Class used to store force results, with regards to a given [element, node]
    location
    """
    
    nComponents = 6
    
    def __init__(self, element_obj, node_obj, results_arr=None):
        
        super().__init__([element_obj,node_obj], results_arr)
    


# ********************** FUNCTIONS ****************************************
     
def integrate_over_mesh(mesh_obj,
                        integrand_func:callable,
                        args=[],
                        kwargs={},
                        cumulative=False):
    """
    Function to perform gauss integration across the domain of a mesh
    
    ***
    Required:
        
    * `mesh_obj`, instance of Mesh class, defining mesh to be integrated over
    
    * `integrand_func`, _callable_ defining the integrand, i.e. the function to 
      be integrated over the mesh:
          
          * First argument shall be 'GaussPoint' instance
          
          * Other arguments may be passed via `args` and `kwargs` parameters.
      
    ***
    Optional:
    
    * `cumulative`, boolean, if True the integral as evaluated over individual 
      elements will be returned. If False (default) only total integral over 
      all elements in the mesh will be returned.
      
    ***
    Returns:
        
    Float or numpy array, depending on `cumulative` parameter.
          
    """  
    if not isinstance(mesh_obj, Mesh):
        raise ValueError("`mesh_obj` to be of Mesh type")
        
    full_gp_list = mesh_obj.get_gauss_points()
        
    mesh_integral = []
        
    for i, gp_list in enumerate(full_gp_list): # iterate over all elements
        
        element_integral = 0.0
        for gp in gp_list: # iterate over all gps (within each element)
            
            element_integral += gp.weight * integrand_func(gp,*args,**kwargs)
        
        mesh_integral += [element_integral]
    
    # Determine which type of summation to use to aggregate integrals over mesh
    if cumulative:
        integral = npy.sum(mesh_integral)
    else:
        integral = npy.cumsum(mesh_integral)
    
    return integral
            

def integrate_gauss(f:callable,x,
                    f_args=[],f_kwargs={},
                    n_gp=3,
                    make_plot=True,
                    ax_list=None,
                    **kwargs):
    """
    Evaluates integral by sub-dividing x-domain into a list of elements and 
    using gauss quadrature to obtain integral across each element.
    
    ***
    Required:
    
    * `f`, integrand function f(x)
    
    * `x`, 1d-array providing salient values to use in integration:
        
        * Bounds for integration are x[0], x[-1]
        
        * Other values may be provided to control the mesh used for gauss 
          quadrature. E.g. more accurate integration is obtained by 
          narrowing the step (element length) in parts of the function where 
          integrand function is known a-priori to vary strongly.
          
    ***
    Optional:
        
    * `f_args`, `f_kwargs`; _list_ and _dict_ respectively, used to pass 
      arguments and keyword arguments to `f`.
      
    * `n_gp`, either int or _list_ of length `len(x)-1`, used to define 
      number of gauss points per element
      
    Any further keyword arguments provided are passed to 
    `integrate_over_mesh()` method, which is being used to implement 
    integration over a mesh.      
    """
    
    # Define mesh i.e. series of elements forming a chain
    x = list(x)
    if len(x)<2:
        raise ValueError("`x` to be list of length 2\n" +
                         "x[0] and x[-1] denote bounds for integration")
    
    node_list = []
    for _x in x:
        node_list.append(Node(parent_mesh=None, #append later by MeshChain init
                              xyz=[_x,0.0,0.0]))
        
    integration_mesh = MeshChain(node_list=node_list,name='integration_mesh')
    
    integration_mesh.define_gauss_points(N_gp=n_gp,verbose=False)
    
    # Define wrapper around passed function to implement using gauss point data
    def f_gp(gp_obj,*a,**b):
        return f(gp_obj.x,*a,**b)
    
    # Evaluate integral
    integral_vals = integrate_over_mesh(integration_mesh,
                                        f_gp,args=f_args,kwargs=f_kwargs,
                                        cumulative=False)
    # Make plot
    if make_plot:
        if ax_list is None:
            fig, (ax1, ax2) = plt.subplots(2,sharex=True)
        else:
            ax1, ax2 = ax_list
            
        ax1.plot(x,f(x,*f_args,**f_kwargs),'g.',label='$f(x_{salient})$')
        ax2.plot(x,npy.hstack((0.0,integral_vals)),'g.',label='gauss')
    
    return integration_mesh, integral_vals
    
    
            
    
# ********************** TEST ROUTINES ****************************************
        
if __name__ == "__main__":
    
    testRoutine2Run=4
    
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
        
    if testRoutine2Run==2:
        
        print("*** TEST ROUTINE 2 COMMENCED *****")
        print("--- Test of local element axes ---")
        print("")
    
        # Define new mesh
        meshObj1 = Mesh(name="Element axes test")
        
        # Define some nodes
        xyz = npy.asarray([[0,0,0],[0,0,1.1],[0,1,0],[0.5,0.5,1],[0,1,-1],[2,0,0]])
        Nn = xyz.shape[0]
        
        for n in range(Nn):
            Node(meshObj1,xyz=xyz[n,:],name=n)
            
        # Define some elements
        for n in range(1,Nn):
            
            node1 = meshObj1.node_objs[0]
            node2 = meshObj1.node_objs[n]
            LineElement(meshObj1,[node1,node2])
            
        # Plot mesh
        meshObj1.plot(plot_axes=True)
        
    if testRoutine2Run==3:
        
        print("*** TEST ROUTINE 3 COMMENCED *****")
        print("--- Test of integration by gauss quadrature ---")
        
        # Define arbitrary polynomial
        p = npy.poly1d([3,2, 1, 2, -5])
        x = npy.linspace(-2.0,1.5,100)
        fig, (ax1,ax2) = plt.subplots(2,sharex=True)
        ax1.plot(x,p(x))
        
        # Perform integration over an uneven mesh using just two elements
        x_salient = [x[0],0.1,x[-1]]
        ax1.plot(x_salient,p(x_salient),'r.')
        ax1.axhline(0.0,color='k',alpha=0.3) # overlay y=0 line
        
        mesh, integral_vals = integrate_gauss(p,x_salient,
                                              n_gp=3,
                                              cumulative=True,
                                              make_plot=True,
                                              ax_list=[ax1,ax2])
        
        # Derive exact integrated polynomial
        p_integral = npy.polyint(p)
        p0 = p_integral(x[0]) # integration constant
        ax2.plot(x,p_integral(x)-p0,label='exact')
        ax1.legend()
        ax2.legend()
        
    if testRoutine2Run==4:
        
        print("*** TEST ROUTINE 4 COMMENCED *****")
        print("--- Test of element skew angles ---")
        
        # Define new mesh
        meshObj1 = Mesh(name="Skew angles test")
        
        xyz = npy.array([[0,0,0],[1,1,1]])
        offset = npy.array([0.0,0.5,0])
        
        nElements = 9
        xyz_arr = npy.array([xyz+i*offset for i in range(nElements)])
        
        phi_vals = npy.deg2rad(npy.linspace(0,360,nElements))
        
        element_list = []
        
        for phi, (xyz1,xyz2) in zip(phi_vals,xyz_arr):
            
            node1 = Node(meshObj1,xyz1)
            node2 = Node(meshObj1,xyz2)
            element = LineElement(meshObj1,[node1,node2],skew_angle = phi)
            element_list.append(element)
        
        # Plot mesh
        meshObj1.plot(plot_axes=True)
        
            
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
        
   

    
    
    