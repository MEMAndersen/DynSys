"""
Class used to defined meshes, i.e. nodes interconnected by elements (lines)
"""

# Library imports
import matplotlib as mpl
import mpl_toolkits.mplot3d# import Axes3D
import numpy as npy
import matplotlib.pyplot as plt
import pandas
import matplotlib.animation as animation


class mesh:
    """
    Class is used to define a mesh, i.e. nodes inter-connected by elements
    meshes of meshes are also possible
    """
        
    # Class properties
    
    # Class methods
    
    def __init__(self,name,**kwargs):
        
        self.name = "mesh_"+name
        
        # Create empty dictionary attributes to contain objects within
        self.meshObjs={}
        self.meshObjs[self.name]=self
        self.nodeObjs={}
        self.elementObjs={}
        
        # Append objects passed-in via kwargs
        if "meshObjs" in kwargs:        self.AppendObjs("mesh",kwargs.get("meshObjs"))
        if "nodeObjs" in kwargs:        self.AppendObjs("node",kwargs.get("nodeObjs"))
        if "elementObjs" in kwargs:     self.AppendObjs("element",kwargs.get("elementObjs"))
        
    def __del__(self):
        pass
    
    def extent(self):
        """
        Determine xyz extent of mesh
        """
        xyz_min=[0,0,0]
        xyz_max=[0,0,0]
        
        for meshkey in self.meshObjs:
            
            meshObj=self.meshObjs[meshkey]
            
            for nodekey in meshObj.nodeObjs:
               
                nodeObj=meshObj.nodeObjs[nodekey]
                xyz_vals=nodeObj.xyz
                
                for d in range(3):
                    
                    # Revise envelope
                    if xyz_vals[d]>xyz_max[d]:
                        xyz_max[d]=xyz_vals[d]
                    if xyz_vals[d]<xyz_min[d]:
                        xyz_min[d]=xyz_vals[d]
                        
        return xyz_min,xyz_max
                    
    
    def appendObjs(self,objType,newObjs):
        """
        Append objects to mesh
        objType may be:
            mesh     : mesh objects
            node     : node objects
            element  : element objects
        """
        
        # create list in case of single object input
        if not hasattr(newObjs, '__iter__'):
            newObjs=[newObjs]

        # create link to this mesh object to which object belongs
        for obj in newObjs:
            obj.meshObj=self 
    
        # add new objects to dictionaries
        if str.lower(objType)=="mesh":
            for obj in newObjs: self.meshObjs[obj.name]=obj
        elif str.lower(objType)=="node":
            for obj in newObjs: self.nodeObjs[obj.name]=obj
        elif str.lower(objType)=="element":
            for obj in newObjs: self.elementObjs[obj.name]=obj
        else:
            raise ValueError("Unexpected objType requested!")
            
    def readNodePos(self,fname="nodePos.csv",header=0,indexCol=None,**kwargs):
        """
        Function to read node positions in from a csv file
        """
        # Use pandas to read in data as dataframe
        df = pandas.read_csv(fname,header=header,index_col=indexCol,
                                     converters={'Name': str})
        #print(df)    
        
        # Loop throuh rows in dataframe
        for index, row in df.iterrows():
            
            # Define new node object
            new_obj=node.node(row["Name"],xyz=[row['X'],row['Y'],row['Z']])
            self.appendObjs("node",new_obj)
            
    def readElemConn(self,fname="elemConn.csv",header=0,indexCol=None,**kwargs):
        """
        Function to read element connectivity from a csv file
        """
        # Use pandas to read in data as dataframe
        df = pandas.read_csv(fname,header=header,index_col=indexCol,dtype=str)
        #print(df)    
        
        # Loop throuh rows in dataframe
        for index, row in df.iterrows():
            
            # Define new node object
            new_obj=element.element(row["Name"])
            self.appendObjs("element",new_obj)
            new_obj.connectNodes([row["Node1"],row["Node2"]])
            
    def defineFromFiles(self,**kwargs):
         """
         High-level function used to define mesh using .csv file data
         """
         self.readNodePos()
         self.readElemConn()
         
    def readDeformations(self,stepRef,fname="nodeDisp.csv",header=0,indexCol=None,printOutput=False,**kwargs):
        """
        Function to read-in displacement information from .csv file
        """
        
        if printOutput==True: print("Reading deformations for case {0} from file {1}".format(stepRef,fname))
        
        # Use pandas to read in data as dataframe
        df = pandas.read_csv(fname,header=header,index_col=indexCol,
                             usecols=['Name','DOF',stepRef],
                             converters={'Name': str,'DOF':str})
        #print(df)    
        
        # Reshape dataframe based on Name,DOF to give results for stepRef
        df = df.pivot(index='Name',columns='DOF',values=stepRef)
        #print(df)
        
        # Loop throuh rows in dataframe
        for index, row in df.iterrows():
            nodeName="node_"+index
            self.nodeObjs[nodeName].setDeformedPos([row['DX'],row['DY'],row['DZ']])
        
    def printAttrs(self,addSpaces=True):
        """
        Prints mesh attributes, i.e. details of all meshes, nodes and elements
        """
        
        print("Mesh attributes relating to mesh {0}:".format(self.name))
        if addSpaces: print("")
        
        for key in self.meshObjs:
            
            meshObj=self.meshObjs[key]
            print("Mesh name={0}".format(meshObj.name))
            
            print("nMeshes={0}, nNodes={1}, nElements={2}".format(len(meshObj.meshObjs),len(meshObj.nodeObjs),len(meshObj.elementObjs)))
            if addSpaces: print("")
            
            print("Nodes within mesh {0}:".format(meshObj.name))
            if addSpaces: print("")
            
            for key in meshObj.nodeObjs:
                nodeObj=meshObj.nodeObjs[key]
                nodeObj.printAttrs()
            if addSpaces: print("")
            
            print("Elements within mesh {0}:".format(meshObj.name))
            if addSpaces: print("")
            
            for key in meshObj.elementObjs:
                elementObj=meshObj.elementObjs[key]
                elementObj.printAttrs()
            if addSpaces: print("")
        
    def plot(self,ax,dwgObjs=None,updatePlot=False,
             plotNull=False,printOutput=True,
             plotNodes=False,plotElements=True,
             elementLineStyle_undeformed=None,elementLineStyle_deformed=None,
             elementLineColor_undeformed=None,elementLineColor_deformed=None,
             nodeColor=None,nodeStyle=None,
             **kwargs):
        """
        Produces a plot of the system configuration 
        Optional keyword arguments:
            updatePlot   : True=update plot, False=Draw new plot
            dwgObjs      : dict of drawing objects, required to update plot
            plotDeformed : True=plot deformed configuraton
        """
        if not updatePlot:
            str1="Plotting"
        else:
            str1="Updating"
            
        if printOutput:
            print("{0} mesh {1}".format(str1,self.name))
        
        # Determine whether deformed plot is required
        if "plotDeformed" in kwargs:
            plotDeformed=kwargs.get("plotDeformed")
            if printOutput:
                print("Both undeformed and deformed configurations will be plotted")
            
        else:
            plotDeformed=False
            if printOutput:
                print("Only undeformed configuration will be plotted")
        
        if printOutput: print("")
        
        
        # If plotting for first time create empty dict to store drwObjs in
        if not updatePlot:
            dwgObjs={}
            
        # Plot function 
        def func_plotObj(obj,
                         updatePlot=False,dwgObjs=None,
                         plotDeformed=False,
                         lineStyle=None,lineColor=None,
                         markerStyle=None,markerColor=None):
            
            key=obj.name
            if plotDeformed: key=key+"_deformed"
            #print("dwgObj key={0}".format(key))
            dwgObj=None
            if not updatePlot:
                
                dwgObjs[key] = obj.plot(ax,
                                        printOutput=printOutput,
                                        plotNull=plotNull,
                                        plotDeformed=plotDeformed,
                                        lineStyle=lineStyle,
                                        lineColor=lineColor,
                                        markerStyle=markerStyle,
                                        markerColor=markerColor)
            
            else:
                
                dwgObj = dwgObjs[key][key]
                
                dwgObj = obj.plot(ax,
                                  updatePlot=True,dwgObj=dwgObj,
                                  printOutput=printOutput,
                                  plotNull=plotNull,
                                  plotDeformed=plotDeformed,
                                  lineStyle=lineStyle,
                                  lineColor=lineColor,
                                  markerStyle=markerStyle,
                                  markerColor=markerColor)
            
            
        # Plot all meshes
        meshNames=[]
        for meshkey in self.meshObjs:
            
            meshObj=self.meshObjs[meshkey]
            meshNames.append(meshObj.name)
            
            if printOutput:
                print("{0} mesh {1}".format(str1,meshObj.name))
                print("")
            
            # Plot all nodes in mesh
            if meshObj.nodeObjs:
                
                if plotNodes:
                    if printOutput: print("Plotting all nodes in mesh {0}"
                                          .format(meshObj.name))
                    for nodekey in meshObj.nodeObjs:
                        
                        # Get node object
                        nodeObj=meshObj.nodeObjs[nodekey]
                        
                        # Undeformed configuration
                        func_plotObj(nodeObj,
                                     updatePlot=updatePlot,dwgObjs=dwgObjs,
                                     plotDeformed=False,
                                     markerStyle=nodeStyle,markerColor=nodeColor)
                        
                        # Deformed configuration
                        if plotDeformed:
                            
                            func_plotObj(nodeObj,
                                         updatePlot=updatePlot,dwgObjs=dwgObjs,
                                         plotDeformed=True,
                                         markerStyle=nodeStyle,markerColor=nodeColor)                          
                    
                    if printOutput: print("")
            
            # Plot all elements in mesh
            if meshObj.elementObjs:
                
                if plotElements:
                    if printOutput: print("Plotting all elements in mesh {0}"
                                          .format(meshObj.name))
                    for elemkey in meshObj.elementObjs:
                        
                        # Get element object
                        elemObj=meshObj.elementObjs[elemkey]
                        
                        # Undeformed configuration
                        func_plotObj(elemObj,
                                     updatePlot=updatePlot,dwgObjs=dwgObjs,
                                     plotDeformed=False,
                                     lineStyle=elementLineStyle_undeformed,
                                     lineColor=elementLineColor_undeformed)
                        
                        # Deformed configuration
                        if plotDeformed:
                            func_plotObj(elemObj,
                                         updatePlot=updatePlot,dwgObjs=dwgObjs,
                                         plotDeformed=True,
                                         lineStyle=elementLineStyle_deformed,
                                         lineColor=elementLineColor_deformed)                                   
                                
                    if printOutput: print("")
            
        # Return dictionary of drawing objects
        return dwgObjs
            
            

        
class element:
    """
    Base class used to define elements, i.e. entities connecting nodes (points in space)
    """
        
    # Class properties
    
    
    # Class methods
    
    def __init__(self,name,**kwargs):
        
        self.name = "elem_"+name
        self.connectedNodes=[] # store as list; order matters
        if "connectedNodeNames" in kwargs: self.connectNodes(kwargs.get("connectedNodeNames"))
        
    def __def__(self):
        pass
    
    def connectNodes(self,nodeNames):
        
        for name2find in nodeNames:
        
            # Get node object of this name
            name2find="node_"+name2find
            nodeObj=self.meshObj.nodeObjs[name2find]
  
            if nodeObj is None:
                raise ValueError("Error: node object {0} could not be found within mesh {1}".format(name2find,self.meshObj.name))

            # Append object to connectNodes list
            self.connectedNodes.append(nodeObj)
            
            # Create two-way link between nodes and the elements they are connected to
            nodeObj.connectElements(self)
            
    def printAttrs(self,printName=True,printConnectivity=True):
        """
        Prints element attributes
        """
        if printName:       print("Element name: {0}".format(self.name))
        
        if printConnectivity:
            nodeNames=[]
            if self.connectedNodes:
                for nodeObj in self.connectedNodes:
                    nodeNames.append(nodeObj.name)
            print("Connected nodes: {0}".format(nodeNames))
            print("")
        else:
            print("")
            
    def plot(self,ax,
             updatePlot=False,dwgObj=None,
             plotNull=False,
             printOutput=True,plotDeformed=False,
             lineColor=None,lineStyle=None,**kwargs):
        """
        Plots element using ax
        """
        if printOutput: print("Plotting element {0}".format(self.name))
              
        # Get XYZ of connected nodes in requested configuration
        xvals=[]
        yvals=[]
        zvals=[]
        
        makePlot=True  # initialised
        
        if not plotNull:
            
            for i in range(len(self.connectedNodes)):
                
                # Get node position to plot
                if plotDeformed:
                    
                    if not hasattr(self.connectedNodes[i],"xyz_deformed"):
                        if not plotNull:
                            print("*** Warning: no deformations defined "
                                  "for node {0} connecting to element {1} ***"
                                  .format(self.connectedNodes[i].name,self.name))
                            print("*** Element {0} cannot be plotted "
                                  "in deformed configuration ***".format(self.name))
                            makePlot=False
                            
                        xyz_vals=None
                    else:  
                        xyz_vals=self.connectedNodes[i].xyz_deformed
                        
                else:
                    xyz_vals=self.connectedNodes[i].xyz
                     
                # Split xyz into coordinates and append to plot list
                xvals.append(xyz_vals[0])
                yvals.append(xyz_vals[1])
                zvals.append(xyz_vals[2])
        
        # Define default line colors to use
        if lineColor is None:
            if plotDeformed:
                lineColor='m'
            else:
                lineColor='b'
                
        if lineStyle is None:
            if plotDeformed:
                lineStyle='-'
            else:
                lineStyle='-'
            
        if makePlot:
            
            if not updatePlot:
                # Make new line to plot element
                dwgObj,=ax.plot(xvals, yvals, zvals,color=lineColor,linestyle=lineStyle)
            else:
                # Update data values used to plot element
                dwgObj.set_data(xvals,yvals)
                dwgObj.set_3d_properties(zvals)
                
        key=self.name
        if plotDeformed: key=key+"_deformed"
            
        return {key:dwgObj}
        
            
            
            
            
            
            
            
            
            
            
            
# ********************* DERIVED CLASS DEFINITIONS *****************************            
            
            
            
            
            
        
            
    
# ********************** TEST ROUTINES ****************************************
# (Only execute when running as a script / top level)
if __name__ == "__main__":
    
    testRoutine2Run=3

    if testRoutine2Run==1:
        
        print("*** TEST ROUTINE 1 COMMENCED ***")
        print("")
    
        # Define new mesh
        meshObj1 = mesh(name="myMesh")
        
        # Define some nodes
        xyz = npy.asarray([[0,0,0],[1,0,1],[2,3,4],[2,1,-1]])
        Nn = xyz.shape[0]
        
        for n in range(Nn):
            new_obj=node(n,xyz=xyz[n,:])
            meshObj1.appendObjs("node",new_obj)
            
        # Define some elements
        elemTopo = npy.asarray([[0,1],[1,2],[0,2]])
        Ne = elemTopo.shape[0]
        
        for e in range(Ne):
            new_obj=element(e)
            meshObj1.appendObjs("element",new_obj)
            new_obj.connectNodes(elemTopo[e,:])
            
        # Define a mesh of meshes
        meshObj2 = mesh(name="myMeshOfMeshes")
        meshObj2.appendObjs("mesh",meshObj1)
        meshObj2.printAttrs()
        
        # Define deformed configuration of mesh
        #vMask="101"
        meshObj1.nodeObjs[0].setDeformedPos([0.2,0.1],vmask="101")
        meshObj1.nodeObjs[1].setDeformedPos([-0.2,0.2,-0.1])
        meshObj1.nodeObjs[2].setDeformedPos([0.2,0.5],vmask="110")
        meshObj1.nodeObjs[3].setDeformedPos([-0.2,-0.4,0.1])
        
        # Produce 3D plot of mesh
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        meshObj2.plot(ax,plotDeformed=True)
        plt.show()
        
    elif testRoutine2Run==2:
        
        print("*** TEST ROUTINE 2 COMMENCED ***")
        print("")
        
        # Define new mesh, reading data from file
        meshObj3 = mesh(name="myCoolMesh")
        meshObj3.defineFromFiles()
        
        # Read deformations from file
        meshObj3.readDeformations("Step2")
        
        # Produce 3D plot of mesh
        fig = plt.figure(figsize=(9,9))
        ax = fig.gca(projection='3d')
        dwgObjs=meshObj3.plot(ax,printOutput=False,plotDeformed=True)
        plt.show()
        
        # Read deformations from file and update plot
        meshObj3.readDeformations("Step3")
        dwgObjs=meshObj3.plot(ax,updatePlot=True,dwgObjs=dwgObjs,printOutput=False,plotDeformed=True)
        
    elif testRoutine2Run==3:
        
        print("*** TEST ROUTINE 2 COMMENCED ***")
        print("")
        
        # Define new mesh, reading data from file
        meshObj1 = mesh(name="myCoolMesh")
        meshObj1.defineFromFiles()
        
        # Create new figure with 3D axes system
        fig = plt.figure(figsize=(9,9))
        ax = fig.gca(projection='3d')
        
        # Set appropriate limits for plot
        xyz_min,xyz_max=[-10,-10,0],[10,10,10]
        ax.set_xlim([xyz_min[0],xyz_max[0]])
        ax.set_ylim([xyz_min[1],xyz_max[1]])
        ax.set_zlim([xyz_min[2],xyz_max[2]])
        
         # Set axes labels etc.
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("My mesh animation")
        
        # Create drawing objects (with no data)
        dwgObjs=meshObj1.plot(ax,
                              printOutput=False,plotNull=True,
                              plotDeformed=True,
                              plotNodes=False,plotElements=True,
                              elementLineStyle_undeformed='--',
                              elementLineStyle_deformed='-',
                              nodeStyle='.',nodeColor='m')
        time_template = 'Time = %.3fs'
        time_text = ax.text(0.05, 0.9, 0,'', transform=ax.transAxes)
        
        # Define function used to draw a clear frame
        def init():
            return animate(0,dt,dwgObjs,time_text)
            
        # Define function to run repeatedly to create animation
        def animate(i,dt,dwgObjs,time_text):
            
            t_val=i*dt
            
            # Read deformations from file
            meshObj1.readDeformations("Step{0}".format(i+1),printOutput=False)
        
            # Update plot
            dwgObjs=meshObj1.plot(ax,
                                  updatePlot=True,dwgObjs=dwgObjs,
                                  printOutput=False,plotDeformed=True,
                                  plotNodes=False,plotElements=True)
            
            # Update time lobel
            time_text.set_text(time_template % t_val)
        
            return dwgObjs,time_text
        
        dt=1
        delay=2
        ani = animation.FuncAnimation(fig, animate,
                                      frames=3,
                                      fargs=(dt,dwgObjs,time_text,),
                                      interval=dt*1000,
                                      repeat=True,repeat_delay=delay*1000,
                                      init_func=init)
        plt.show()
    
    else:
        print("(No valid test routine selected)")
        
        
        
class node:
    """
    Base class used to define nodes i.e. points in space    
    """
    
    # Class properties
    
    
    # Class methods
    
    def __init__(self,name,**kwargs):
        
        self.name = "node_"+name
        
        if "xyz" in kwargs:    self.xyz = npy.asarray(kwargs.get("xyz"))
        self.connectedElements={}
        if "connectedElements" in kwargs: self.connectElements(kwargs.get("connectedElements"))
    
    def __del__(self):
        pass
    
    def connectElements(self,elementObjs):
        
        # create list in case of single object input
        if not hasattr(elementObjs, '__iter__'):
            elementObjs=[elementObjs]
        
        for obj in elementObjs:
            self.connectedElements[obj.name]=obj
            
    def setDeformedPos(self,v,vmask=None):
        """
        Sets the deformed position of the node, given displacements v
        Required arguments:
            v       : usually a [3,] array
        Optional arguments:    
            vmask   : string displacement mask applied
                      (e.g. for 1D analysis in XY plane; dofs in Y:  "001")
                      (e.g. for 2D analysis in XZ plane; dofs in X,Z:"101")
        """
        
        # Covert v to npy.array format
        v = npy.asarray(v)
        
        # Parse displacement mask to establish active dimensions
        nDims=3
        
        if vmask is None:
            
            #[3,] array expected
            if v.shape[0]!=3:
                raise ValueError("Error: expected v.shape[0]==3")
                
        else:
            
            v_new = npy.zeros((3,)) # create 3D vector to store displacements in
            
            if len(vmask)!=3:
                raise ValueError("Error: 3-character string expected")
            
            dim=-1
            for i in range(len(vmask)):
                if vmask[i]=='0':
                    nDims=nDims-1
                elif vmask[i]=='1':
                    dim=dim+1
                    v_new[i]=v[dim]  # assign freedom provided to 3D array
                else:
                    raise ValueError("Error: unexpected character in vmask")
                    
            # Check nDims corresponds to length of v array provided
            if nDims!=v.shape[0]:
                raise ValueError("Error: vmask implies {0} dofs, "
                                 "but v.shape[0]={1}".format(nDims,v.shape[0]))
                    
            # Re-assign v
            v = v_new
            del v_new
        
        # Add displacements to undeformed positions
        self.xyz_deformed = self.xyz + v
    
    def printAttrs(self,printName=True,printXYZ=True,printConnectivity=True):
        """
        Prints nodes attributes
        """
        if printName:           print("Node name:  {0}".format(self.name))
        if printXYZ:            print("XYZ coords: {0}".format(self.xyz))
        
        if printConnectivity:
            elementNames=[]
            if self.connectedElements:
                for key in self.connectedElements:
                    obj=self.connectedElements[key]
                    elementNames.append(obj.name)
            print("Connected elements: {0}".format(elementNames))
            print("")
        else:
            print("")
            
    def plot(self,ax,
             updatePlot=False,dwgObj=None,plotNull=False,
             printOutput=True,plotDeformed=False,
             markerStyle=None,markerColor=None,**kwargs):
        """
        Plots node using ax
        """
        if printOutput: print("Plotting node {0}".format(self.name))
        
        # Get XYZ of connected nodes
        makePlot=True
        
        if plotDeformed:
            if not hasattr(self,"xyz_deformed"):
                if not plotNull:
                    print("*** Warning: no deformations defined for node {0}  ***"
                          .format(self.name))
                    print("*** Undeformed position of this node will not be plotted ***")
                    makePlot=False
                    
                xyz=None
            else:
                xyz = self.xyz_deformed    
        else:
            xyz = self.xyz
        
        # Set default marker styles and colors
        if markerColor is None:
            if plotDeformed:
                markerColor='m'
            else:
                markerColor='b'
        
        if markerStyle is None:
            if plotDeformed:
                markerStyle='.'
            else:
                markerStyle='.'
        
        # Produce plot
        if makePlot:
            
            # Get x,y,z vals to plot
            xvals=[]
            yvals=[]
            zvals=[]
            if not plotNull:
                xvals.append(xyz[0])
                yvals.append(xyz[1])
                zvals.append(xyz[2])
        
            if not updatePlot:
                # Plot node
                dwgObj,= ax.plot(xvals, yvals, zvals,marker=markerStyle,color=markerColor)
            else:
                # Update data values used to plot node
                dwgObj.set_data(xvals,yvals)
                dwgObj.set_3d_properties(zvals)
                
        key=self.name
        if plotDeformed: key=key+"_deformed"
        
        return {key:dwgObj}
    