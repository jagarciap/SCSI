#Data structures to hold domain information
from functools import reduce

import copy
import numba as nb
import numpy
import os
import pdb
import vtk

import evtk.hl as evtk
from evtk.vtk import VtkGroup

import accelerated_functions as af
import Boundaries.outer_2D_rectangular as ob
import Boundaries.inner_2D_rectangular as ib
import constants as c
import cylindrical_mesh_tools as cmt
from solver import location_indexes_inv

#Mesh (Abstract)(Association between Mesh and PIC):
#
#Definition = Defines the type of mesh.
#Attributes:
#	+nPoints (int) = Number of points in the mesh.
#       +boundaries ([Boundary]) = List of different boundaries that define the mesh.
#	+volumes ([double]) = Volume of each node.
#       +location ([int]) = indexes indicating all the nodes that are boundaries in the mesh. There might be different arrangement according to
#           every type of mesh. (Repeated indices deprecated in 2020_11_20).
#       +location_sat ([int]) = indexes indicating the nodes that are boundaries with the satellite.
#       +direction_sat ([int]) = array of indexes of the size of 'location_sat', indicating for each node the direction towards the satellite.
#       +area_sat ([double]) = array indicating the area that represents each satellite node.
#Methods:
#       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints, boundaries and any other subclass variable.
#       +checkPositionInMesh([double, double] pos): [Boolean] = For each position it retuns a bool indicating wether the position lies inside the mesh.
#	+getPosition([int] i): [double, double y] = For a each index return its real position.
#	+getIndex([double,double] pos): [double,double] = For each real position returns its index value. Important to remember that the number of columns may vary
#           depending on the actual type of mesh subclass used.
#	+arrayToIndex([ind] array): [int, int] = For the indexes in the 1D array, obtain the indexes used for the particular mesh.
#	+indexToArray([ind, ind] index): [int] = For the indexes used for the particular mesh, obtain the 1D version for the array.
#       +reverseVTKOrdering(array): array = The array received as argument is ordered in such a way it can be stored ina VTK file.
#           The result is returned as a new array.
#       +vtkOrdering(array): array = The array received as argument comes with vtk ordering and is reshaped to be stored properly in the code.
#       +vtkReader(): Reader = Return the reader from module vtk that is able to read the vtk file.
#       +saveVTK(string filename, dict dictionary) = It calls the appropiate method in 'vtk' module to print the information of the system in a '.vtk' file.
#       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preoladed density and velocity of that node. This will depend on each type of mesh.
#	+print() = Print a VTK file / Matplotlib visualization of the mesh (points and connections between nodes). Also print volumes.
class Mesh (object):
    def __init__(self):
        setDomain()

    def setDomain(self):
        pass

    def checkPositionInMesh(self, pos):
        pass

    def getPosition(self, ind):
        pass

    def getIndex(self, pos):
        pass

    def arrayToIndex(self, array):
        pass

    def indexToArray(self, ind):
        pass

    def vtkOrdering(self, array):
        pass

    def reverseVTKOrdering(self, array):
        pass

    def vtkReader(self):
        pass

    def saveVTK(self, filename, dictionary):
        pass

    def loadSpeciesVTK(self, species):
        pass

    def print(self):
        pass


#Mesh_recursive (Abstract):
#
#Definition = Abstract class that works as an interface for classes meant to be recursive.
#Attributes:
#	+root (Boolean) = Identifies whether the object is the root of the tree or not.
#	+id (String) = String that uniquely identifies the mesh among the meshes of the domain. The identity is assigned as:
#           ID is organized as a chain of numbers separated by dashes, everything stored as a string. The first number is "0", indicating root of the tree.
#           The second level is a 0-based index describing the position of the child in the parent mesh. 
#           For the third and successive levels, the same approach is applied, until the mesh being identified is finally reached.
#	+start_ind (int) = node index in the parent mesh where this mesh starts.
#       +children ([Mesh]) = List of all the children of the mesh.
#Methods:
#	+flatIndexing((String) ID, (int) index): int flat_ind = The function receives a unique description of a node in the domain by the index of the node in its mesh (index), and the ID of such mesh (ID).
#           Then, the function returns the index that represents the node under the 'Flat indexation' rule, counting from the mesh that executes the function.
#       +executeFunctionByIndex((String) ID, (int) index, (method) func, *args, **kwargs): This function searches for the mesh that contains the node denoted by 'index' (Counting from the mesh that executes
#           'executeFunctionByIndex' for the first time) and then executes 'self.func(*args, **kwargs)'.
#       +orderedIndexing((int) index): (String) ID, (int) index = This functions operates inversely to 'flatIndexing(ID, index)'. It receives a node identification, counting from the mesh that
#           executes the function, and returns its unique identification by giving the index of the node in its correspondent mesh as well as the 'ID' of such mesh.
#       +sortArrayByMeshes((ndarray) array, seedList = []) [ndarray] = This function receives the values for the nodes of the mesh and its NGs, ordered by 'Flat indexation rule', 
#           and returns a list of arrays with the same values of the original array but now sorted by 'Ordered rule', each array containing the values of one mesh.
#       +executeFunctionByMeshes(Function func, List seedList = [], *args, **kwargs): List = This method receives a function (func) to be executed in all the nodes of the mesh tree.
#           The argument function can receive normal arguments as well as keyword arguments. The method receives all the arguments intended for func in the same positional order, but for each type of argument,
#           it receives all the arguments for the execution of func throughout the nodes of the tree, stored in a list ordered by the 'Ordered rule'. For keyword arguments, if func receives key = value,
#           the method must receive key = [value0, value1, ...]. The method stores the results of the executions of func in the optional key argument 'seedList', and the results are also ordered by the
#           'Ordered rule'. These results can be either obtained by passing a list object to seedList or by capturing the return of the method, which is only possible if the node that called the function
#           is the root of the tree.
#       +sortPositionsByMeshes([double, double] pos, List seedList): List = This function receives an array of positions and organize the positions in ndarrays stored in a list ordered by the 'Ordered rule'.

class Mesh_recursive(object):
    def __init__(self, n_children, n_root, n_id, n_start_ind, n_type):
        n_type += " - Recursive"
        self.root = n_root
        self.id = n_id
        self.start_ind = n_start_ind
        self.children = n_children
        self.accPoints = self.nPoints+reduce(lambda x, y: x+y.accPoints, self.children, 0)
        
        #Unique features for the root
        if self.root == True:
            #Location sat
            def func(mesh, acc = None):
                if acc is None:
                    acc = [0]
                local = mesh.location_sat + acc[0]
                acc[0] += mesh.nPoints
                for child in mesh.children:
                    local = numpy.append(local, func(child, acc = acc))
                return local
            self.overall_location_sat = func(self)
            #Volumes
            def func(mesh, attribute):
                acc_list = numpy.copy(mesh.__getattribute__(attribute))
                for child in mesh.children:
                    acc_list = numpy.append(acc_list, func(child, attribute), axis = 0)
                return acc_list
            self.volumes = func(self, "volumes")
            self.overall_area_sat = func(self, "area_sat")

            #Setting up potential and location_index_inv for use in the main program
            if len(self.overall_location_sat) > 0:
                test = location_indexes_inv([self.location_sat[0]], store = True, location = self.overall_location_sat)[0]
                assert test == 0, "location_indexes_inv is not correctly set up"
        
    # This function works well with numpy arrays as long as the indeces are all from the same mesh (which makes sense to have it only working like that)
    # flatIndexing( String ID, [int] index): [int] = The function receives indices according to the index system of its parent mesh, identified by ID, and
    #   returns their flat indexation from the mesh that executed this function for the first time.
    def flatIndexing(self, ID, index):
        assert ID.startswith(self.id), "The function is being used in a branch of the tree where the node is not in"
        if self.id == ID:
            return index
        else:
            child_pos = int((ID.partition(self.id+"-")[2]).partition("-")[0])
            acc = 0
            for i in range(child_pos):
                acc += self.children[i].accPoints
            return self.nPoints + acc + self.children[child_pos].flatIndexing(ID, index)
    
    def executeFunctionByIndex(self, index, func, *args, ind = None, **kwargs):
        assert numpy.all(self.accPoints >= index), "The index passed as argument is invalid"
        if numpy.all(index < self.nPoints):
            return func(index, *args, **kwargs)
        else:
            c = self.nPoints+self.children[0].accPoints
            i = 0
            while i < len(self.children) and c <= index:
                c += self.children[i].accPoints
                i += 1
            return self.children[i].executeFunctionByIndex(index-c+self.children[i].accPoints, func, *args, ind = ind, **kwargs)

    def orderedIndexing(self, index):
        def func(self, index):
            return self.id, index
        return self.executeFunctionByIndex(index, func)

    def sortArrayByMeshes(self, array, seedList = None):
        if type(array) == numpy.ndarray:
            array = list(array)
            seedList = []
        seedList.append(numpy.asarray(array[:self.nPoints]))
        del array[:self.nPoints]
        for child in self.children:
            child.sortArrayByMeshes(array, seedList = seedList)
        return seedList

    def executeFunctionByMeshes(self, func, seedList = None, *args, **kwargs):
        if self.root:
            seedList = []
        seedList.append(func(*map(lambda x: x.pop(0), args), *map(lambda x: {x : kwargs[x].pop(0)}, kwargs)))
        for child in self.children:
            child.executeFunctionByMeshes(func, seedList = seedList, *args, **kwargs)
        if self.root:
            return seedList

    def sortPositionsByMeshes(self, pos, seedList = None, return_ind = None, indexes = None, surface = False):
        mask = numpy.zeros(numpy.shape(pos)[0])
        for i in range(len(self.children)):
            mask += (i+1)*self.children[i].checkPositionInMesh(pos, surface = surface).astype(numpy.int_)
        if seedList is None:
            seedList = []
        seedList.append(pos[numpy.logical_not(mask.astype(numpy.bool_)),:])
        if return_ind is not None:
            if indexes is None:
                indexes = numpy.arange(numpy.shape(pos)[0], dtype = 'uint32')
            return_ind.append(indexes[numpy.logical_not(mask.astype(numpy.bool_))])
        for i in range(len(self.children)):
            if return_ind is None:
                self.children[i].sortPositionsByMeshes(pos[mask == i+1], seedList = seedList, return_ind = None, indexes = None, surface = surface)
            else:
                self.children[i].sortPositionsByMeshes(pos[mask == i+1], seedList = seedList, return_ind = return_ind, indexes = indexes[mask == i+1], surface = surface)
            if return_ind == None:
                return seedList
            else:
                return seedList, return_ind

#   +sortIndexByMeshes(ind, List seedList, int acc_index): [[int]] = This function organizes an array of indexes in flat-ordering into a list of arrays, where the arrays are organized with the tree order. 
#       Each array contains the nodes that correspond to each mesh. The rest of arguments are used only for recursive purposes.
#       The 'shift' keyword indicates whether the indexes are shifted from flat indexation rule to the index values of each mesh or not.
    def sortIndexByMeshes(self, ind, shift = True, seedList = None, accIndex = None):
        if seedList is None and accIndex is None:
            seedList = []
            accIndex = [0]
        c1 = numpy.greater_equal(ind, accIndex[0])
        temp = accIndex[0]
        accIndex[0] += self.nPoints
        c2 = numpy.less(ind, accIndex[0])
        c3 = numpy.logical_and(c1, c2)
        if shift:
            seedList.append(ind[c3]-temp)
        else:
            seedList.append(ind[c3])
        ind = ind[numpy.logical_not(c3)]
        for child in self.children:
            child.sortIndexByMeshes(ind, shift = shift, seedList = seedList, accIndex = accIndex)
        return seedList

#   +groupIndexByPosition([int] ind, positions = False) = This function receives a list of indexes and groups them in a list of lists, where each list contains the nodes that
#       refer to the same physical position (less than 'err' apart). If 'position = True', a numpy array is returned with the positions linked to the lists.
    def groupIndexByPosition(self, ind, positions = False, seedList = None, posList = None, accIndex = None, err = 1e-4):
        if self.root:
            ind = self.sortIndexByMeshes(ind, shift = False)
            ind_i = ind.pop(0)
            posList = [super(Mesh_recursive, self).getPosition(ind_i)]
            seedList = [[x] for x in ind_i]
            accIndex = [self.nPoints]
        else:
            ind_i = ind.pop(0)-accIndex[0]
            n_pos = super(Mesh_recursive, self).getPosition(ind_i)
            limit = len(seedList)
            for ind_ii, n_pos_i in zip(ind_i, n_pos):
                ind_n = numpy.flatnonzero(numpy.linalg.norm(posList[0][:limit]-n_pos_i, axis = 1) < err)
                if len(ind_n) > 0:
                    seedList[ind_n[0]].append(ind_ii+accIndex[0])
                else:
                    seedList.append([ind_ii+accIndex[0]])
                    posList[0] = numpy.append(posList[0], n_pos_i.reshape((1,2)), axis = 0)
            accIndex[0] += self.nPoints
        for child in self.children:
            child.groupIndexByPosition(ind, positions = positions, seedList = seedList, posList = posList, accIndex = accIndex)
        if positions:
            return seedList, posList[0]
        else:
            return seedList

#       +vtkOrdering(array): array = The array received as argument is ordered in such a way it can be stored in a VTK file.
#           The array received, if it is root, is the array including information of all the meshes. The function thus returns
#           a list of numpy arrays, each one prepared for storing in a VTK file.
    def vtkOrdering(self, array, accIndex = [0]):
        if self.root:
            array = self.sortArrayByMeshes(array)
            accIndex = [0]
        array[accIndex[0]] = super(Mesh_recursive,self).vtkOrdering(array[accIndex[0]])
        accIndex[0] += 1
        for child in self.children:
            child.vtkOrdering(array, accIndex = accIndex)
        return array

#       +reverseVTKOrdering(array): array = The array received as argument is ordered in such a way it can be stored ina VTK file.
    def reverseVTKOrdering(self, array):
        dims = numpy.shape(array)
        if len(dims) == 1:
            return array.reshape((self.nPoints), order = 'F')
        else:
            return array.reshape((self.nPoints, 3), order = 'F')[:,:2]

#    def saveVTK(self, filename, dictionary, files = None, accIndex = None):
#        if self.root:
#            #Creating folder
#            cwd, name = os.path.split(filename)
#            path = os.path.join(cwd, 'mesh_components',name)
#            try:
#                os.makedirs(path)
#            except FileExistsError:
#                pass
#            files = []
#            accIndex = [0]
#            #dictionary = {key: self.sortArrayByMeshes(value) for key, value in dictionary.items()}
#            nmeshes = len(list(dictionary.values())[0])
#            dicts = [dict() for i in range(nmeshes)]
#            for key, value in dictionary.items():
#                for i in range(nmeshes):
#                    dicts[i][key] = value[i]
#            dictionary = dicts
#        #Creation of individual files
#        cwd, name = os.path.split(filename)
#        filename_comp = os.path.join(cwd,'mesh_components',name,'{}_{}'.format(name, self.id))
#        super(Mesh_recursive, self).saveVTK(filename_comp, dictionary[accIndex[0]])
#        #TODO: Fix later: I need to change the extension of the file manually everytime I use a different file type
#        files.append(filename_comp+'.vtr')
#        accIndex[0] += 1
#        #Recursion
#        for child in self.children:
#            child.saveVTK(filename, dictionary, files = files, accIndex = accIndex)
#        if self.root:
#            g = VtkGroup(filename)
#            for comp in files:
#                g.addFile(comp, sim_time = 0.0)
#            g.save()

    def saveVTK(self, filename, dictionary, files = None, accIndex = None):
        if self.root:
            ##Creating folder
            #cwd, name = os.path.split(filename)
            #path = os.path.join(cwd, 'mesh_components',name)
            #try:
            #    os.makedirs(path)
            #except FileExistsError:
            #    pass
            files = []
            accIndex = [0]
            #dictionary = {key: self.sortArrayByMeshes(value) for key, value in dictionary.items()}
            nmeshes = len(list(dictionary.values())[0])
            dicts = [dict() for i in range(nmeshes)]
            for key, value in dictionary.items():
                for i in range(nmeshes):
                    dicts[i][key] = value[i]
            dictionary = dicts
        #Creation of individual files
        cwd, name = os.path.split(filename)
        filename_comp = os.path.join(cwd, '{}_{}'.format(self.id, name))
        super(Mesh_recursive, self).saveVTK(filename_comp, dictionary[accIndex[0]])
        #TODO: Fix later: I need to change the extension of the file manually everytime I use a different file type
        files.append(filename_comp+'.vtr')
        accIndex[0] += 1
        #Recursion
        for child in self.children:
            child.saveVTK(filename, dictionary, files = files, accIndex = accIndex)
        #if self.root:
        #    g = VtkGroup(filename)
        #    for comp in files:
        #        g.addFile(comp, sim_time = 0.0)
        #    g.save()

        
#NOTE: Backup
#    def orderedIndexing(self, index):
#        assert self.accPoints <= index, "The index passed as argument is invalid"
#        if index < self.nPoints:
#            return self.id, index
#        else:
#            c = self.nPoints+self.children[0].accPoints
#            i = 0
#            while i < len(self.children) and c <= index:
#                c += self.children[i].accPoints
#                i += 1
#            return self.children[i].orderedIndexing(index-c+self.children[i].accPoints)


#Mesh_2D_rm (Inherits from Mesh):
#
#Definition = Mesh class for a 2D rectangular mesh. The organization of the points will work as 0<=i<nx and 0<=j<ny. Also, for k parameter 0<=k<nPoints, k = nx*j+i.
#Attributes:
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#	+depth (double) = Artificial thickness of the domain, to make it three-dimensional.
#	+nx (int) = Number of nodes in the x direction.
#	+ny (int) = Number of nodes in the y direction.
#       +dx (float32) = Distance between adyacent horizontal nodes
#       +dy (float32) = Distance between adyacent vertical nodes
#       +boundaries ([Boundary]) = It is [Outer_2D_Rectangular].
#       +Mesh class attributes.
#Methods:
#	+Implementation of Mesh methods.
class Mesh_2D_rm (Mesh):
    type = "2D_rm"
    def __init__(self, xmin, xmax, ymin, ymax, dx, dy, depth, boundaries):
        self.depth = depth

        self.boundaries = boundaries
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dx = dx
        self.dy = dy
        self.nx = numpy.rint((xmax-xmin)/dx+1).astype('uint32')
        self.ny = numpy.rint((ymax-ymin)/dy+1).astype('uint32')
        self.boundaries = boundaries
        self.setDomain()

#       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints and any other subclass variable.
    def setDomain(self):
        self.nPoints = numpy.uint32(self.nx*self.ny)

        self.volumes = (self.dx*self.dy*self.depth)*numpy.ones((self.nPoints), dtype = 'float32')
        self.volumes[:self.nx] /= 2
        self.volumes[self.nx*(self.ny-1):] /= 2
        self.volumes[self.nx-1::self.nx] /= 2
        self.volumes[:self.nx*self.ny:self.nx] /= 2
        
        #Initializing the rest of the attributes of the outer boundary
        self.boundaries[0].bottom = numpy.arange(0,self.nx, dtype = 'uint32')
        b_areas = self.dx*self.depth*numpy.ones_like(self.boundaries[0].bottom)
        b_directions = numpy.zeros_like(self.boundaries[0].bottom, dtype = numpy.uint8)
        b_adjacent = numpy.append(self.boundaries[0].bottom[1:], self.boundaries[0].bottom[-1]+self.nx)

        self.boundaries[0].left = numpy.arange(0, self.nx*self.ny, self.nx, dtype = 'uint32')
        l_areas = self.dy*self.depth*numpy.ones_like(self.boundaries[0].left)
        l_directions = 3*numpy.ones_like(self.boundaries[0].left, dtype = numpy.uint8)
        l_adjacent = numpy.append(self.boundaries[0].left[1:], self.boundaries[0].left[-1]+1)

        self.boundaries[0].right = numpy.arange(self.nx-1, self.nx*self.ny, self.nx, dtype = 'uint32')
        r_areas = copy.copy(l_areas)
        r_directions = numpy.ones_like(self.boundaries[0].right, dtype = numpy.uint8)
        r_adjacent = self.boundaries[0].right+self.nx
        r_adjacent[-1] = r_adjacent[-2]

        self.boundaries[0].top = numpy.arange(self.nx*(self.ny-1), self.nx*self.ny, dtype = 'uint32')
        t_areas = copy.copy(b_areas)
        t_directions = 2*numpy.ones_like(self.boundaries[0].top, dtype = numpy.uint8)
        t_adjacent = self.boundaries[0].top+1
        t_adjacent[-1] = t_adjacent[-2]

        self.boundaries[0].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].bottom, \
                                                                                   self.boundaries[0].left),\
                                                                                   self.boundaries[0].right),\
                                                                                   self.boundaries[0].top), return_index = True)
        ind2 = numpy.argsort(self.boundaries[0].location)
        self.boundaries[0].location = self.boundaries[0].location[ind2]
        self.boundaries[0].areas = numpy.append(numpy.append(numpy.append(b_areas,\
                                                                          l_areas),\
                                                                          r_areas),\
                                                                          t_areas)[ind1][ind2]
        self.boundaries[0].directions = numpy.append(numpy.append(numpy.append(b_directions,\
                                                                               l_directions),\
                                                                               r_directions),\
                                                                               t_directions)[ind1][ind2]

        adjacent_nodes = numpy.append(numpy.append(numpy.append(b_adjacent,\
                                                                l_adjacent),\
                                                                r_adjacent),\
                                                                t_adjacent)[ind1][ind2]

        self.location = self.boundaries[0].location
        self.location_sat = numpy.zeros((0), dtype = numpy.uint8) if self.boundaries[0].material == "space" else self.boundaries[0].location
        self.area_sat = numpy.zeros((0)) if self.boundaries[0].material == "space" else self.boundaries[0].areas
        self.direction_sat = numpy.zeros((0)) if self.boundaries[0].material == "space" else self.boundaries[0].directions

        test = location_indexes_inv([self.location[0]], store = True, location = self.location)[0]
        assert test == 0, "location_indexes_inv is not correctly set up"
        adjacent_nodes = location_indexes_inv(adjacent_nodes, store = False)
        self.boundaries[0].adjacent = [{self.boundaries[0].directions[i]:adjacent_nodes[i]} for i in range(len(adjacent_nodes))]
        self.boundaries[0].adjacent[0].update({l_directions[0]: self.nx})
        self.boundaries[0].adjacent[-1].update({2:self.boundaries[0].adjacent[-1][1]})
        #Corners
        corners = location_indexes_inv([self.boundaries[0].bottom[-1], self.boundaries[0].left[-1]], store = False)
        self.boundaries[0].adjacent[corners[0]].update({1: self.boundaries[0].adjacent[corners[0]][0]})
        self.boundaries[0].adjacent[corners[1]].update({2: self.boundaries[0].adjacent[corners[1]][3]})
        self.adjacent_sat = [] if self.boundaries[0].material == "space" else self.boundaries[0].adjacent

        #Setting up potential and location_index_inv for use in the main program
        if len(self.location_sat) > 0:
            test = location_indexes_inv([self.location_sat[0]], store = True, location = self.location_sat)[0]
            assert test == 0, "location_indexes_inv is not correctly set up"

    def checkPositionInMesh(self, pos, surface = False):
        cumul = numpy.ones((pos.shape[0]), dtype = numpy.bool_)
        for boundary in self.boundaries:
            cumul = numpy.logical_and(cumul, boundary.checkPositionInBoundary(pos, surface = surface))
        return cumul

#    def checkPositionInMesh(self, pos, surface = False):
#        xmin = self.xmin
#        xmax = self.xmax
#        ymin = self.ymin
#        ymax = self.ymax
#        if surface and self.boundaries[0].material != "space":
#            return af.gleq_p(pos, xmin, xmax, ymin, ymax)
#        else:
#            return af.gl_p(pos, xmin, xmax, ymin, ymax)

#	+getPosition([int] i): [double, double y] = For a each index return its real position.
    def getPosition(self, ind):
        index2D = self.arrayToIndex(ind)
        dx = self.dx
        dy = self.dy
        xmin = self.xmin
        ymin = self.ymin

        posx, posy = numpy.around(af.getPosition_p(index2D[:,0], index2D[:,1], dx, dy, xmin, ymin), decimals = c.INDEX_PREC)
        return numpy.append(posx[:,None], posy[:,None], axis = 1)

#	+getIndex([double,double] pos): [double,double] = For each real position returns its index value.
    def getIndex(self, pos):
        dx = self.dx
        dy = self.dy
        xmin = self.xmin
        ymin = self.ymin
        indx, indy =  af.getIndex_p(pos[:,0], pos[:,1], dx, dy, xmin, ymin)
        return numpy.around(numpy.append(indx[:,None], indy[:,None], axis = 1), decimals = c.INDEX_PREC)

#       +arrayToIndex([ind] array): [int, int] = For the indexes in the 1D array, obtain the indexes used for the particular mesh.
    def arrayToIndex(self, array):
        j, i = numpy.divmod(array, self.nx)
        return numpy.append(i[:,None], j[:,None], axis = 1)

#	+indexToArray([ind, ind] index): [int] = For the indexes used for the particular mesh, obtain the 1D version for the array.
    def indexToArray(self, ind):
        nx = self.nx
        ind = ind.astype('uint32')
        return af.indexToArray_p(ind[:,0], ind[:,1], nx)

#       +vtkOrdering(array): array = The array received as argument is ordered in such a way it can be stored ina VTK file.
#           The result is returned as a new array.
    def vtkOrdering(self, array):
        dims = numpy.shape(array)
        if len(dims) == 1:
            return array.reshape((self.nx, self.ny, 1), order = 'F')
        else:
            tpl = tuple(numpy.reshape(copy.copy(array[:,i]),(self.nx, self.ny, 1), order = 'F') for i in range(dims[1]))
            if len(tpl) < 3:
                for i in range (3-len(tpl)):
                    tpl += (numpy.zeros_like(array[:,0].reshape((self.nx,self.ny,1))),)
            return tpl

#       +reverseVTKOrdering(array): array = The array received as argument is ordered in such a way it can be stored ina VTK file.
    def reverseVTKOrdering(self, array):
        dims = numpy.shape(array)
        if len(dims) == 1:
            return array.reshape((self.nPoints), order = 'F')
        else:
            return array.reshape((self.nPoints, 3), order = 'F')[:,:2]

#       +vtkReader(): Reader = Return the reader from module vtk that is able to read the vtk file.
    def vtkReader(self):
        return vtk.vtkXMLRectilinearGridReader()

#       +saveVTK(string filename, dict dictionary) = It calls the appropiate method in 'vtk' module to print the information of the system in a '.vtk' file.
    def saveVTK(self, filename, dictionary):
        i = numpy.arange(self.xmin, self.xmax+self.dx/2, self.dx)
        j = numpy.arange(self.ymin, self.ymax+self.dy/2, self.dy)
        temp = numpy.zeros((1), dtype = 'int32')

        evtk.gridToVTK(filename, i, j, temp, pointData = dictionary)

#       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preoladed density and velocity of that node. This will depend on each type of mesh.
    def loadSpeciesVTK(self, species, pic):
        ##Preparing things for numpy functions use
        #particles = (species.mesh_values.density*self.volumes/species.spwt).astype(int)
        #ind = numpy.arange(self.nPoints)
        #index = numpy.repeat(ind, particles)
        ##Setting up positions
        #pos = self.getPosition(ind)[index]
        #random = numpy.random.rand(*numpy.shape(pos))
        #random += numpy.where(random == 0, 1e-3, 0)
        #pos[:,0] += numpy.where(index%self.nx != 0, (random[:,0]-0.5)*self.dx, random[:,0]/2*self.dx)
        #pos[:,0] -= numpy.where(index%self.nx == self.nx-1, random[:,0]/2*self.dx, 0)
        #pos[:,1] += numpy.where(index>self.nx, (random[:,1]-0.5)*self.dy, random[:,1]/2*self.dy) 
        #pos[:,1] -= numpy.where(index>=self.nx*(self.ny-1), random[:,1]/2*self.dy, 0) 
        ##Setting up velocities
        #vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature, species.m), particles)

        ##Adding particles
        #self.boundaries[0].addParticles(species, pos, vel)
        #self.boundaries[0].updateTrackers(species, species.part_values.current_n)

        #Preparing things for numpy functions use
        #Volume
        dv = self.dx*self.dy*self.depth
        particles = (species.mesh_values.density[:self.nPoints]*dv/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        index = numpy.repeat(ind, particles)
        #Setting up positions
        pos = self.getPosition(index)
        random = numpy.random.rand(*numpy.shape(pos))
        random += numpy.where(random == 0, 1e-3, 0)
        pos += (random-0.5)*numpy.asarray((self.dx,self.dy)).T
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

#	+print() = Print a VTK file / Matplotlib visualization of the mesh (points and connections between nodes). Also print volumes.
    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})


#Mesh_2D_rm_sat (Inherits from Mesh):
#
#Definition = Mesh class for a 2D rectangular mesh with a rectangular satellite at its center.
#   The organization of the points will work as 0<=i<nx and 0<=j<ny, but taking into account the hole for the sattelite.
#Attributes:
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#	+xminsat (double) = Left limit of the satellite (closest to the Sun).
#	+xmaxsat (double) = Right limit of the satellite (farthest from the Sun).
#	+yminsat (double) = Bottom limit of the satellite.
#	+ymaxsat (double) = Top limit of the satellite.
#	+depth (double) = Artificial thickness of the domain, to make it three-dimensional.
#   +dx (float32) = Distance between adyacent horizontal nodes
#   +dy (float32) = Distance between adyacent vertical nodes
#	+nx (int) = Number of nodes in the x direction.
#	+ny (int) = Number of nodes in the y direction.
#	+nxsat (int) = Number of nodes in the x direction.
#	+nysat (int) = Number of nodes in the y direction.
#   +boundaries ([Boundary]) = It is [Outer_2D_Rectangular, Inner_2D_Rectangular].
#   +Mesh class attributes.
#Methods:
#	+Implementation of Mesh methods.
class Mesh_2D_rm_sat (Mesh_2D_rm):
    type = "2D_rm_sat"
    def __init__(self, xmin, xmax, ymin, ymax,\
                 xminsat, xmaxsat, yminsat, ymaxsat,\
                 dx, dy, depth, boundaries):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xminsat = xminsat
        self.xmaxsat = xmaxsat
        self.yminsat = yminsat
        self.ymaxsat = ymaxsat
        self.dx = dx
        self.dy = dy
        self.nx = numpy.rint((xmax-xmin)/dx+1).astype('uint32')
        self.ny = numpy.rint((ymax-ymin)/dy+1).astype('uint32')
        self.nxsat = numpy.rint((xmaxsat-xminsat)/dx+1).astype('uint32')
        self.nysat = numpy.rint((ymaxsat-yminsat)/dy+1).astype('uint32')
        self.sat_i = numpy.rint(((yminsat-ymin)/dy)*self.nx+(self.xminsat-self.xmin)/self.dx).astype('uint32')
        self.depth = depth

        self.boundaries = boundaries
        self.setDomain()

    #       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints and any other subclass variable.
    def setDomain(self):
        self.nPoints = numpy.uint32(self.nx*self.ny)

        #Initializing the rest of the attributes of the outer boundary
        self.boundaries[0].bottom = numpy.arange(0,self.nx, dtype = 'uint32')
        b_areas = self.dx*self.depth*numpy.ones_like(self.boundaries[0].bottom)
        b_directions = numpy.zeros_like(self.boundaries[0].bottom, dtype = numpy.uint8)
        b_adjacent = numpy.append(self.boundaries[0].bottom[1:], self.boundaries[0].bottom[-1]+self.nx)

        self.boundaries[0].left = numpy.arange(0, self.nx*self.ny, self.nx, dtype = 'uint32')
        l_areas = self.dy*self.depth*numpy.ones_like(self.boundaries[0].left)
        l_directions = 3*numpy.ones_like(self.boundaries[0].left, dtype = numpy.uint8)
        l_adjacent = numpy.append(self.boundaries[0].left[1:], self.boundaries[0].left[-1]+1)

        self.boundaries[0].right = numpy.arange(self.nx-1, self.nx*self.ny, self.nx, dtype = 'uint32')
        r_areas = copy.copy(l_areas)
        r_directions = numpy.ones_like(self.boundaries[0].right, dtype = numpy.uint8)
        r_adjacent = self.boundaries[0].right+self.nx
        r_adjacent[-1] = r_adjacent[-2]

        self.boundaries[0].top = numpy.arange(self.nx*(self.ny-1), self.nx*self.ny, dtype = 'uint32')
        t_areas = copy.copy(b_areas)
        t_directions = 2*numpy.ones_like(self.boundaries[0].top, dtype = numpy.uint8)
        t_adjacent = self.boundaries[0].top+1
        t_adjacent[-1] = t_adjacent[-2]

        self.boundaries[0].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].bottom, \
                                                                                   self.boundaries[0].left),\
                                                                                   self.boundaries[0].right),\
                                                                                   self.boundaries[0].top), return_index = True)
        ind2 = numpy.argsort(self.boundaries[0].location)
        self.boundaries[0].location = self.boundaries[0].location[ind2]
        self.boundaries[0].areas = numpy.append(numpy.append(numpy.append(b_areas,\
                                                                          l_areas),\
                                                                          r_areas),\
                                                                          t_areas)[ind1][ind2]
        self.boundaries[0].directions = numpy.append(numpy.append(numpy.append(b_directions,\
                                                                               l_directions),\
                                                                               r_directions),\
                                                                               t_directions)[ind1][ind2]
        
        adjacent_nodes = numpy.append(numpy.append(numpy.append(b_adjacent,\
                                                                l_adjacent),\
                                                                r_adjacent),\
                                                                t_adjacent)[ind1][ind2]
        adjacent_nodes = location_indexes_inv(adjacent_nodes, store = True, location = self.boundaries[0].location)
        self.boundaries[0].adjacent = [{self.boundaries[0].directions[i]:adjacent_nodes[i]} for i in range(len(adjacent_nodes))]
        self.boundaries[0].adjacent[0].update({l_directions[0]: self.nx})
        self.boundaries[0].adjacent[-1].update({2:self.boundaries[0].adjacent[-1][1]})
        #Corners
        corners = location_indexes_inv([self.boundaries[0].bottom[-1], self.boundaries[0].left[-1]], store = False)
        self.boundaries[0].adjacent[corners[0]].update({1: self.boundaries[0].adjacent[corners[0]][0]})
        self.boundaries[0].adjacent[corners[1]].update({2: self.boundaries[0].adjacent[corners[1]][3]})

        #Satellite borders
        topleft = self.sat_i+(self.nysat-1)*self.nx

        self.boundaries[1].bottom = numpy.arange(self.sat_i, self.sat_i+self.nxsat, dtype = 'uint32')
        b_areas = self.dx*self.depth*numpy.ones_like(self.boundaries[1].bottom)
        b_directions = numpy.zeros_like(self.boundaries[1].bottom, dtype = numpy.uint8)
        b_adjacent = numpy.append(self.boundaries[1].bottom[1:], self.boundaries[1].bottom[-1]+self.nx)

        self.boundaries[1].left = numpy.arange(self.sat_i, topleft+self.nx, self.nx, dtype = 'uint32')
        l_areas = self.dy*self.depth*numpy.ones_like(self.boundaries[1].left)
        l_directions = 3*numpy.ones_like(self.boundaries[1].left, dtype = numpy.uint8)
        l_adjacent = numpy.append(self.boundaries[1].left[1:], self.boundaries[1].left[-1]+1)

        self.boundaries[1].right = numpy.arange(self.sat_i+self.nxsat-1, topleft+self.nxsat-1+self.nx, self.nx, dtype = 'uint32')
        r_areas = copy.copy(l_areas)
        r_directions = numpy.ones_like(self.boundaries[1].right, dtype = numpy.uint8)
        r_adjacent = self.boundaries[1].right+self.nx
        r_adjacent[-1] = r_adjacent[-2]

        self.boundaries[1].top = numpy.arange(topleft, topleft+self.nxsat, dtype = 'uint32')
        t_areas = copy.copy(b_areas)
        t_directions = 2+b_directions
        t_adjacent = self.boundaries[1].top+1
        t_adjacent[-1] = t_adjacent[-2]

        self.boundaries[1].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[1].bottom, \
                                                                                   self.boundaries[1].left),\
                                                                                   self.boundaries[1].right),\
                                                                                   self.boundaries[1].top), return_index = True)
        ind2 = numpy.argsort(self.boundaries[1].location)
        self.boundaries[1].location = self.boundaries[1].location[ind2]
        self.boundaries[1].areas = numpy.append(numpy.append(numpy.append(b_areas,\
                                                                          l_areas),\
                                                                          r_areas),\
                                                                          t_areas)[ind1][ind2]
        self.boundaries[1].directions = numpy.append(numpy.append(numpy.append(b_directions,\
                                                                               l_directions),\
                                                                               r_directions),\
                                                                               t_directions)[ind1][ind2]

        adjacent_nodes = numpy.append(numpy.append(numpy.append(b_adjacent,\
                                                                l_adjacent),\
                                                                r_adjacent),\
                                                                t_adjacent)[ind1][ind2]
        adjacent_nodes = location_indexes_inv(adjacent_nodes, store = True, location = self.boundaries[1].location)
        self.boundaries[1].adjacent = [{self.boundaries[1].directions[i]:adjacent_nodes[i]} for i in range(len(adjacent_nodes))]
        #Corners
        self.boundaries[1].adjacent[0].update({l_directions[0]: self.nxsat})
        corners = location_indexes_inv([self.boundaries[1].bottom[-1], self.boundaries[1].left[-1]], store = False)
        self.boundaries[1].adjacent[corners[0]].update({1: self.boundaries[1].adjacent[corners[0]][0]})
        self.boundaries[1].adjacent[corners[1]].update({2: self.boundaries[1].adjacent[corners[1]][3]})
        self.boundaries[1].adjacent[-1].update({2:self.boundaries[1].adjacent[-1][1]})
        #Little correction to make the bottom-left node account as an impact in the left side of the spacecraft
        self.boundaries[1].directions[0] = int(3)

        self.boundaries[1].ind_inner = numpy.concatenate(tuple(numpy.arange(self.boundaries[1].left[i]+1, self.boundaries[1].right[i])\
                                                          for i in range(1, int(self.nysat-1))))

        #Volume
        self.volumes = (self.dx*self.dy*self.depth)*numpy.ones((self.nPoints), dtype = 'float32')
        numpy.divide.at(self.volumes , self.boundaries[0].location, 2)
        numpy.divide.at(self.volumes , self.boundaries[1].location, 2)
        #Corners of the outer boundary
        numpy.divide.at(self.volumes, [self.boundaries[0].bottom[0],\
                                         self.boundaries[0].bottom[-1],\
                                         self.boundaries[0].top[0],\
                                         self.boundaries[0].top[-1]], 2)
        #Corners of the satellite
        numpy.multiply.at(self.volumes, [self.boundaries[1].bottom[0],\
                                         self.boundaries[1].bottom[-1],\
                                         self.boundaries[1].top[0],\
                                         self.boundaries[1].top[-1]], 3/2)

        #Locations of borders in the mesh
        self.location = numpy.append(self.boundaries[0].location, self.boundaries[1].location)
        indices = numpy.argsort(self.location)
        self.location = self.location[indices]
        #Location of the satellite
        self.location_sat = numpy.zeros((0), dtype ='uint32')
        self.area_sat = numpy.zeros((0), dtype ='float32')
        self.direction_sat = numpy.zeros((0), dtype ='uint8')
        self.adjacent_sat = []
        if self.boundaries[0].material != "space":
            self.location_sat = numpy.append(self.location_sat, self.boundaries[0].location)
            self.area_sat = numpy.append(self.area_sat, self.boundaries[0].areas)
            self.direction_sat = numpy.append(self.direction_sat, self.boundaries[0].directions)
            self.adjacent_sat.extend(self.boundaries[0].adjacent)
        if self.boundaries[1].material != "space":
            self.location_sat = numpy.append(self.location_sat, self.boundaries[1].location)
            self.area_sat = numpy.append(self.area_sat, self.boundaries[1].areas)
            self.direction_sat = numpy.append(self.direction_sat, self.boundaries[1].directions)
            self.adjacent_sat.extend(self.boundaries[1].adjacent)
        if self.boundaries[0].material != "space" and boundaries[1].material != "space":
            self.location_sat = self.location_sat[indices]
            self.area_sat = self.area_sat[indices]
            self.direction_sat = self.direction_sat[indices]
            #TODO: Needs to be fixed later. The values of the dictionaries are wrong.
            self.adjacent_sat = [self.adjacent_sat[i] for i in range(len(indices))]

        #Setting up potential and location_index_inv for use in the main program
        if len(self.location_sat) > 0:
            test = location_indexes_inv([self.sat_i], store = True, location = self.location_sat)[0]
            assert test == 0, "location_indexes_inv is not correctly set up"


#Mesh_2D_rm_sat_HET (Inherits from Mesh_2D_rm_sat):
#
#Definition = Mesh class for a 2D rectangular mesh with a rectangular satellite at its center, which contains a Hall Effect Thruster.
#   The organization of the points will work as 0<=i<nx and 0<=j<ny, but taking into account the hole for the sattelite.
#Attributes:
#   +boundaries ([Boundary]) = It is [Outer_2D_Rectangular, Inner_2D_Rectangular, Inner_1D_HET].
#   +Mesh_2D_rm_sat attributes class attributes.
#Methods:
#	+Implementation of Mesh methods.
class Mesh_2D_rm_sat_HET (Mesh_2D_rm_sat):
    type = "2D_rm_sat_HET"
    def __init__(self, xmin, xmax, ymin, ymax,\
                 xminsat, xmaxsat, yminsat, ymaxsat,\
                 dx, dy, depth, boundaries):
        super().__init__(xmin, xmax, ymin, ymax, xminsat, xmaxsat, yminsat, ymaxsat, dx, dy, depth, boundaries)
        self.setDomain()

    #       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints and any other subclass variable.
    def setDomain(self):
        super().setDomain()

        #Initializing the rest of the attributes of the HET boundary
        center = self.boundaries[1].top[len(self.boundaries[1].top)//2]
        boundary_nx = numpy.rint((self.boundaries[2].xmax-self.boundaries[2].xmin)/self.dx).astype('uint32')
        half_nx = boundary_nx//2
        self.boundaries[2].location = numpy.arange(center-half_nx, center+half_nx+1, dtype = 'uint32')
        self.boundaries[2].directions = 2*numpy.ones_like(self.boundaries[2].location, dtype = 'uint8')
        self.boundaries[2].exit_nodes = numpy.asarray([self.boundaries[2].location[0], self.boundaries[2].location[-1]])
        self.boundaries[2].exit_pot_nodes = numpy.asarray([self.boundaries[2].location[0], self.boundaries[2].location[1],\
                                                           self.boundaries[2].location[-2], self.boundaries[2].location[-1]])


#Mesh_2D_rm_separateBorders (Inherits from Mesh_2D_rm):
#
#Definition = Mesh class for a 2D rectangular mesh. The organization of the points will work as 0<=i<nx and 0<=j<ny. Also, for k parameter 0<=k<nPoints, k = nx*j+i.
#               It differes from 'Mesh_2D_rm' in that instead of a single rectangular boundary, they bondaries are 4 1D boundaries, organized as
#               [bottom, right, top, left].
#Attributes:
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#	+depth (double) = Artificial thickness of the domain, to make it three-dimensional.
#	+nx (int) = Number of nodes in the x direction.
#	+ny (int) = Number of nodes in the y direction.
#       +dx (float32) = Distance between adyacent horizontal nodes
#       +dy (float32) = Distance between adyacent vertical nodes
#       +boundaries ([Boundary]) = It is [Outer_1D_Rectangular x 4], with the order [bottom, right, top, left].
#       +Mesh class attributes.
#Methods:
#	+Implementation of Mesh methods.
class Mesh_2D_rm_separateBorders(Mesh_2D_rm):
    type = "2D_rm_separateBorders"
    def __init__(self, xmin, xmax, ymin, ymax, dx, dy, depth, boundaries):
        super().__init__(xmin, xmax, ymin, ymax, dx, dy, depth, boundaries)

#       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints and any other subclass variable.
    def setDomain(self):
        self.nPoints = numpy.uint32(self.nx*self.ny)

        self.volumes = (self.dx*self.dy*self.depth)*numpy.ones((self.nPoints), dtype = 'float32')
        self.volumes[:self.nx] /= 2
        self.volumes[self.nx*(self.ny-1):] /= 2
        self.volumes[self.nx-1::self.nx] /= 2
        self.volumes[:self.nx*self.ny:self.nx] /= 2
        
        #Initializing the boundaries
        #Bottom
        self.boundaries[0].location = numpy.arange(0,self.nx, dtype = 'uint32')
        self.boundaries[0].areas = self.dx*self.depth*numpy.ones_like(self.boundaries[0].location)
        self.boundaries[0].areas[[0,-1]] /= 2
        self.boundaries[0].directions = numpy.zeros_like(self.boundaries[0].location, dtype = numpy.uint8)
        self.boundaries[0].adjacent = [{0:i+1} for i in range(self.nx)]
        self.boundaries[0].adjacent[-1][0] = self.boundaries[0].adjacent[-2][0]

        #Right
        self.boundaries[1].location = numpy.arange(self.nx-1, self.nx*self.ny, self.nx, dtype = 'uint32')
        self.boundaries[1].areas = self.dy*self.depth*numpy.ones_like(self.boundaries[1].location)
        self.boundaries[1].areas[[0,-1]] /= 2
        self.boundaries[1].directions = numpy.ones_like(self.boundaries[1].location, dtype = numpy.uint8)
        self.boundaries[1].adjacent = [{1:i+1} for i in range(self.ny)]
        self.boundaries[1].adjacent[-1][1] = self.boundaries[1].adjacent[-2][1]

        #Top
        self.boundaries[2].location = numpy.arange(self.nx*(self.ny-1), self.nx*self.ny, dtype = 'uint32')
        self.boundaries[2].areas = self.dx*self.depth*numpy.ones_like(self.boundaries[2].location)
        self.boundaries[2].areas[[0,-1]] /= 2
        self.boundaries[2].directions = 2*numpy.ones_like(self.boundaries[2].location, dtype = numpy.uint8)
        self.boundaries[2].adjacent = [{2:i+1} for i in range(self.nx)]
        self.boundaries[2].adjacent[-1][2] = self.boundaries[2].adjacent[-2][2]

        #Left
        self.boundaries[3].location = numpy.arange(0, self.nx*self.ny, self.nx, dtype = 'uint32')
        self.boundaries[3].areas = self.dy*self.depth*numpy.ones_like(self.boundaries[3].location)
        self.boundaries[3].areas[[0,-1]] /= 2
        self.boundaries[3].directions = 3*numpy.ones_like(self.boundaries[3].location, dtype = numpy.uint8)
        self.boundaries[3].adjacent = [{3:i+1} for i in range(self.ny)]
        self.boundaries[3].adjacent[-1][3] = self.boundaries[3].adjacent[-2][3]

        #Set up the rest of the attributes and check for satellite borders
        self.location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].location, \
                                                                                   self.boundaries[1].location),\
                                                                                   self.boundaries[2].location),\
                                                                                   self.boundaries[3].location), return_index = True)
        ind2 = numpy.argsort(self.location)
        self.location = self.location[ind2]

        self.location_sat = numpy.zeros((0), dtype = numpy.uint8)
        self.area_sat = numpy.zeros((0))
        self.direction_sat = numpy.zeros((0), dtype = numpy.uint8)
        self.adjacent_sat = []
        for boundary in self.boundaries:
            if boundary.material != 'space':
                self.location_sat = numpy.append(self.location_sat, boundary.location)
                self.area_sat = numpy.append(self.area_sat, boundary.areas)
                self.direction_sat = numpy.append(self.direction_sat, boundary.directions)
                for dic in boundary.adjacent:
                    self.adjacent_sat.append({key: boundary.location[value] for key, value in dic.items()})

        self.location_sat, ind1 = numpy.unique(self.location_sat, return_index = True)
        ind2 = numpy.argsort(self.location_sat)

        self.location_sat = self.location_sat[ind2]
        self.area_sat = self.area_sat[ind1][ind2]
        self.direction_sat = self.direction_sat[ind1][ind2]

        #Setting up location_index_inv for use in the main program
        if len(self.location_sat) > 0:
            test = location_indexes_inv([self.location_sat[0]], store = True, location = self.location_sat)[0]
            assert test == 0, "location_indexes_inv is not correctly set up"

        temp = []
        for i in range(len(ind2)):
            dic = self.adjacent_sat[ind1[ind2[i]]]
            temp.append({})
            temp[-1].update({key:location_indexes_inv([value], store = False)[0] for key, value in dic.items()})
        self.adjacent_sat = temp
        #Corners
        if 0 in self.direction_sat and 3 in self.direction_sat:
            self.adjacent_sat[0].update({3:self.nx})
        if 0 in self.direction_sat and 1 in self.direction_sat:
            corner = location_indexes_inv([self.boundaries[0].location[-1]], store = False)[0]
            self.adjacent_sat[corner].update({1: self.adjacent_sat[corner][0]})
        if 2 in self.direction_sat and 3 in self.direction_sat:
            corner = location_indexes_inv([self.boundaries[3].location[-1]], store = False)[0]
            self.adjacent_sat[corner].update({3: self.adjacent_sat[corner][3]})


#Mesh_2D_cm (Inherits from Mesh_2D_rm):
#
#Definition = Mesh class for a 2D cylindrical (z-r) mesh. The organization of the points will work as 0<=i<nx and 0<=j<ny. Also, for k parameter 0<=k<nPoints, k = nx*j+i.
#Attributes:
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#	+nx (int) = Number of nodes in the x direction.
#	+ny (int) = Number of nodes in the y direction.
#       +dx (float32) = Distance between adyacent horizontal nodes
#       +dy (float32) = Distance between adyacent vertical nodes
#       +boundaries ([Boundary]) = It is [Outer_2D_Cylindrical].
#       +Mesh class attributes.
#Methods:
#	+Implementation of Mesh methods.
class Mesh_2D_cm (Mesh_2D_rm):
    type = "2D_cm"
    def __init__(self, xmin, xmax, ymin, ymax, dx, dy, boundaries):
        self.boundaries = boundaries
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dx = dx
        self.dy = dy
        self.nx = numpy.rint((xmax-xmin)/dx+1).astype('uint32')
        self.ny = numpy.rint((ymax-ymin)/dy+1).astype('uint32')
        self.boundaries = boundaries
        self.setDomain()

#       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints and any other subclass variable.
    def setDomain(self):
        self.nPoints = numpy.uint32(self.nx*self.ny)
        y = (numpy.arange(self.nPoints)//self.nx)*self.dy+self.ymin
        if self.ymin == 0.0:
            y[:self.nx] = self.dy/8

        self.volumes = 2*numpy.pi*y*self.dy*self.dx
        self.volumes[:self.nx] /= 1 if self.ymin == 0.0 else 2
        self.volumes[self.nx*(self.ny-1):] /= 2
        self.volumes[self.nx-1::self.nx] /= 2
        self.volumes[:self.nx*self.ny:self.nx] /= 2
        
        #Initializing the rest of the attributes of the outer boundary
        self.boundaries[0].bottom = numpy.arange(0,self.nx, dtype = 'uint32')
        if self.ymin == 0.0:
            b_areas = 2*numpy.pi*y[:self.nx:self.nx-1]*self.dx
            b_directions = numpy.zeros_like(b_areas, dtype = numpy.uint8)
            b_adjacent = numpy.append(self.boundaries[0].bottom[0]+self.nx, self.boundaries[0].bottom[-1]+self.nx)
        else:
            b_areas = 2*numpy.pi*y[:self.nx]*self.dx
            b_directions = numpy.zeros_like(self.boundaries[0].bottom, dtype = numpy.uint8)
            b_adjacent = numpy.append(self.boundaries[0].bottom[1:], self.boundaries[0].bottom[-1]+self.nx)

        self.boundaries[0].left = numpy.arange(0, self.nx*self.ny, self.nx, dtype = 'uint32')
        l_areas = 2*numpy.pi*y[::self.nx]*self.dy
        l_directions = 3*numpy.ones_like(self.boundaries[0].left, dtype = numpy.uint8)
        l_adjacent = numpy.append(self.boundaries[0].left[1:], self.boundaries[0].left[-1]+1)

        self.boundaries[0].right = numpy.arange(self.nx-1, self.nx*self.ny, self.nx, dtype = 'uint32')
        r_areas = copy.copy(l_areas)
        r_directions = numpy.ones_like(self.boundaries[0].right, dtype = numpy.uint8)
        r_adjacent = self.boundaries[0].right+self.nx
        r_adjacent[-1] = r_adjacent[-2]

        self.boundaries[0].top = numpy.arange(self.nx*(self.ny-1), self.nx*self.ny, dtype = 'uint32')
        t_areas = 2*numpy.pi*y[self.nx*(self.ny-1):]*self.dx
        t_directions = 2*numpy.ones_like(self.boundaries[0].top, dtype = numpy.uint8)
        t_adjacent = self.boundaries[0].top+1
        t_adjacent[-1] = t_adjacent[-2]

        if self.ymin == 0.0:
            self.boundaries[0].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].bottom[0::self.nx-1], \
                                                                                       self.boundaries[0].left),\
                                                                                       self.boundaries[0].right),\
                                                                                       self.boundaries[0].top), return_index = True)
        else:
            self.boundaries[0].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].bottom, \
                                                                                       self.boundaries[0].left),\
                                                                                       self.boundaries[0].right),\
                                                                                       self.boundaries[0].top), return_index = True)

        ind2 = numpy.argsort(self.boundaries[0].location)
        self.boundaries[0].location = self.boundaries[0].location[ind2]
        self.boundaries[0].areas = numpy.append(numpy.append(numpy.append(b_areas,\
                                                                          l_areas),\
                                                                          r_areas),\
                                                                          t_areas)[ind1][ind2]
        self.boundaries[0].directions = numpy.append(numpy.append(numpy.append(b_directions,\
                                                                               l_directions),\
                                                                               r_directions),\
                                                                               t_directions)[ind1][ind2]

        adjacent_nodes = numpy.append(numpy.append(numpy.append(b_adjacent,\
                                                                l_adjacent),\
                                                                r_adjacent),\
                                                                t_adjacent)[ind1][ind2]

        self.location = self.boundaries[0].location
        self.location_sat = numpy.zeros((0), dtype = numpy.uint8) if self.boundaries[0].material == "space" else self.boundaries[0].location
        self.area_sat = numpy.zeros((0)) if self.boundaries[0].material == "space" else self.boundaries[0].areas
        self.direction_sat = numpy.zeros((0)) if self.boundaries[0].material == "space" else self.boundaries[0].directions

        test = location_indexes_inv([self.location[0]], store = True, location = self.location)[0]
        assert test == 0, "location_indexes_inv is not correctly set up"
        adjacent_nodes = location_indexes_inv(adjacent_nodes, store = False)
        self.boundaries[0].adjacent = [{self.boundaries[0].directions[i]:adjacent_nodes[i]} for i in range(len(adjacent_nodes))]
        self.boundaries[0].adjacent[0].update({l_directions[0]: self.nx})
        self.boundaries[0].adjacent[-1].update({2:self.boundaries[0].adjacent[-1][1]})
        #Corners
        corners = location_indexes_inv([self.boundaries[0].bottom[-1], self.boundaries[0].left[-1]], store = False)
        self.boundaries[0].adjacent[corners[0]].update({1: self.boundaries[0].adjacent[corners[0]][0]})
        self.boundaries[0].adjacent[corners[1]].update({2: self.boundaries[0].adjacent[corners[1]][3]})
        self.adjacent_sat = [] if self.boundaries[0].material == "space" else self.boundaries[0].adjacent

        #Setting up potential and location_index_inv for use in the main program
        if len(self.location_sat) > 0:
            test = location_indexes_inv([self.location_sat[0]], store = True, location = self.location_sat)[0]
            assert test == 0, "location_indexes_inv is not correctly set up"

#       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preoladed density and velocity of that node. This will depend on each type of mesh.
    def loadSpeciesVTK(self, species, pic, prec = 10**(-c.INDEX_PREC)):
        #Number of particles created
        particles = (species.mesh_values.density[:self.nPoints]*self.volumes[:self.nPoints]/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(ind)
        rmin = numpy.where(pos[:,1] == self.ymin, pos[:,1], pos[:,1]-self.dy/2)
        rmax = numpy.where(pos[:,1] == self.ymax, pos[:,1], pos[:,1]+self.dy/2)
        pos = numpy.repeat(pos, particles, axis = 0)
        pos[:,1] = cmt.randomYPositions_2D_cm(particles, rmin, rmax)
        random = numpy.random.rand(numpy.shape(pos)[0])
        shifts = numpy.where((pos[:,0]-self.xmin) < prec, random*self.dx/2, (random-0.5)*self.dx)
        shifts -= numpy.where((pos[:,0]-self.xmax) > -prec, random*self.dx/2, 0)
        pos[:,0] += shifts
        pos[:,1] += numpy.where((pos[:,1] - self.ymin) < prec, prec, 0)
        pos[:,1] += numpy.where((pos[:,1] - self.ymax) > -prec, -prec, 0)
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)


#Mesh_2D_cm_sat (Inherits from Mesh_2D_cm):
#
#Definition = Mesh class for a 2D cylindrical (z-r) mesh with a cylindrical satellite at its center.
#   The organization of the points will work as 0<=i<nx and 0<=j<ny, but taking into account the hole for the sattelite.
#   x represents 'z' and y represents 'r'.
#Attributes:
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#	+xminsat (double) = Left limit of the satellite (closest to the Sun).
#	+xmaxsat (double) = Right limit of the satellite (farthest from the Sun).
#	+yminsat (double) = Bottom limit of the satellite.
#	+ymaxsat (double) = Top limit of the satellite.
#       +dx (float32) = Distance between adyacent horizontal nodes
#       +dy (float32) = Distance between adyacent vertical nodes
#	+nx (int) = Number of nodes in the x direction.
#	+ny (int) = Number of nodes in the y direction.
#	+nxsat (int) = Number of nodes in the x direction.
#	+nysat (int) = Number of nodes in the y direction.
#   +boundaries ([Boundary]) = It is [Outer_2D_Cylindrical, Inner_2D_Cylindrical].
#   +Mesh class attributes.
#Methods:
#	+Implementation of Mesh methods.
class Mesh_2D_cm_sat (Mesh_2D_cm):
    type = "2D_cm_sat"
    def __init__(self, xmin, xmax, ymin, ymax,\
                 xminsat, xmaxsat, yminsat, ymaxsat,\
                 dx, dy, boundaries):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xminsat = xminsat
        self.xmaxsat = xmaxsat
        self.yminsat = yminsat
        self.ymaxsat = ymaxsat
        self.dx = dx
        self.dy = dy
        self.nx = numpy.rint((xmax-xmin)/dx+1).astype('uint32')
        self.ny = numpy.rint((ymax-ymin)/dy+1).astype('uint32')
        self.nxsat = numpy.rint((xmaxsat-xminsat)/dx+1).astype('uint32')
        self.nysat = numpy.rint((ymaxsat-yminsat)/dy+1).astype('uint32')
        self.sat_i = numpy.rint(((yminsat-ymin)/dy)*self.nx+(self.xminsat-self.xmin)/self.dx).astype('uint32')

        self.boundaries = boundaries
        self.setDomain()

    #       +setDomain() = This function, with the values provided by the boundary files, will create the mesh, by setting up volumes, nPoints and any other subclass variable.
    def setDomain(self):
        self.nPoints = numpy.uint32(self.nx*self.ny)
        y = (numpy.arange(self.nPoints)//self.nx)*self.dy+self.ymin
        if self.ymin == 0.0:
            y[:self.nx] = self.dy/8

        self.volumes = 2*numpy.pi*y*self.dy*self.dx
        self.volumes[:self.nx] /= 1 if self.ymin == 0.0 else 2
        self.volumes[self.nx*(self.ny-1):] /= 2
        self.volumes[self.nx-1::self.nx] /= 2
        self.volumes[:self.nx*self.ny:self.nx] /= 2
        
        #Initializing the rest of the attributes of the outer boundary
        self.boundaries[0].bottom = numpy.arange(0,self.nx, dtype = 'uint32')
        if self.ymin == 0.0:
            b_areas = 2*numpy.pi*y[:self.nx:self.nx-1]*self.dx
            b_directions = numpy.zeros_like(b_areas, dtype = numpy.uint8)
            b_adjacent = numpy.append(self.boundaries[0].bottom[0]+self.nx, self.boundaries[0].bottom[-1]+self.nx)
        else:
            b_areas = 2*numpy.pi*y[:self.nx]*self.dx
            b_directions = numpy.zeros_like(self.boundaries[0].bottom, dtype = numpy.uint8)
            b_adjacent = numpy.append(self.boundaries[0].bottom[1:], self.boundaries[0].bottom[-1]+self.nx)

        self.boundaries[0].left = numpy.arange(0, self.nx*self.ny, self.nx, dtype = 'uint32')
        l_areas = 2*numpy.pi*y[::self.nx]*self.dy
        l_directions = 3*numpy.ones_like(self.boundaries[0].left, dtype = numpy.uint8)
        l_adjacent = numpy.append(self.boundaries[0].left[1:], self.boundaries[0].left[-1]+1)

        self.boundaries[0].right = numpy.arange(self.nx-1, self.nx*self.ny, self.nx, dtype = 'uint32')
        r_areas = copy.copy(l_areas)
        r_directions = numpy.ones_like(self.boundaries[0].right, dtype = numpy.uint8)
        r_adjacent = self.boundaries[0].right+self.nx
        r_adjacent[-1] = r_adjacent[-2]

        self.boundaries[0].top = numpy.arange(self.nx*(self.ny-1), self.nx*self.ny, dtype = 'uint32')
        t_areas = 2*numpy.pi*y[self.nx*(self.ny-1):]*self.dx
        t_directions = 2*numpy.ones_like(self.boundaries[0].top, dtype = numpy.uint8)
        t_adjacent = self.boundaries[0].top+1
        t_adjacent[-1] = t_adjacent[-2]

        if self.ymin == 0.0:
            self.boundaries[0].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].bottom[0::self.nx-1], \
                                                                                       self.boundaries[0].left),\
                                                                                       self.boundaries[0].right),\
                                                                                       self.boundaries[0].top), return_index = True)
        else:
            self.boundaries[0].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[0].bottom, \
                                                                                       self.boundaries[0].left),\
                                                                                       self.boundaries[0].right),\
                                                                                       self.boundaries[0].top), return_index = True)

        ind2 = numpy.argsort(self.boundaries[0].location)
        self.boundaries[0].location = self.boundaries[0].location[ind2]
        self.boundaries[0].areas = numpy.append(numpy.append(numpy.append(b_areas,\
                                                                          l_areas),\
                                                                          r_areas),\
                                                                          t_areas)[ind1][ind2]
        self.boundaries[0].directions = numpy.append(numpy.append(numpy.append(b_directions,\
                                                                               l_directions),\
                                                                               r_directions),\
                                                                               t_directions)[ind1][ind2]

        adjacent_nodes = numpy.append(numpy.append(numpy.append(b_adjacent,\
                                                                l_adjacent),\
                                                                r_adjacent),\
                                                                t_adjacent)[ind1][ind2]

        adjacent_nodes = location_indexes_inv(adjacent_nodes, store = True, location = self.boundaries[0].location)
        self.boundaries[0].adjacent = [{self.boundaries[0].directions[i]:adjacent_nodes[i]} for i in range(len(adjacent_nodes))]
        self.boundaries[0].adjacent[0].update({l_directions[0]: self.nx})
        self.boundaries[0].adjacent[-1].update({2:self.boundaries[0].adjacent[-1][1]})
        #Corners
        corners = location_indexes_inv([self.boundaries[0].bottom[-1], self.boundaries[0].left[-1]], store = False)
        self.boundaries[0].adjacent[corners[0]].update({1: self.boundaries[0].adjacent[corners[0]][0]})
        self.boundaries[0].adjacent[corners[1]].update({2: self.boundaries[0].adjacent[corners[1]][3]})

        #Satellite borders
        topleft = self.sat_i+(self.nysat-1)*self.nx

        self.boundaries[1].bottom = numpy.arange(self.sat_i, self.sat_i+self.nxsat, dtype = 'uint32')
        if self.yminsat == 0.0:
            b_areas = numpy.zeros((0))
            b_directions = numpy.zeros((0), dtype = numpy.uint8)
            b_adjacent = numpy.zeros((0), dtype = 'uint32')
        else:
            b_areas = 2*numpy.pi*y[sat_i:self.nxsat]*self.dx
            b_directions = numpy.zeros_like(self.boundaries[1].bottom, dtype = numpy.uint8)
            b_adjacent = numpy.append(self.boundaries[1].bottom[1:], self.boundaries[1].bottom[-1]+self.nx)

        self.boundaries[1].left = numpy.arange(self.sat_i, topleft+self.nx, self.nx, dtype = 'uint32')
        l_areas = 2*numpy.pi*y[self.sat_i:topleft+self.nx:self.nx]*self.dy
        l_directions = 3*numpy.ones_like(self.boundaries[1].left, dtype = numpy.uint8)
        l_adjacent = numpy.append(self.boundaries[1].left[1:], self.boundaries[1].left[-1]+1)

        self.boundaries[1].right = numpy.arange(self.sat_i+self.nxsat-1, topleft+self.nxsat-1+self.nx, self.nx, dtype = 'uint32')
        r_areas = copy.copy(l_areas)
        r_directions = numpy.ones_like(self.boundaries[1].right, dtype = numpy.uint8)
        r_adjacent = self.boundaries[1].right+self.nx
        r_adjacent[-1] = r_adjacent[-2]

        self.boundaries[1].top = numpy.arange(topleft, topleft+self.nxsat, dtype = 'uint32')
        t_areas = 2*numpy.pi*y[topleft:topleft+self.nxsat]*self.dx
        t_directions = 2*numpy.ones_like(self.boundaries[1].top, dtype = numpy.uint8)
        t_adjacent = self.boundaries[1].top+1
        t_adjacent[-1] = t_adjacent[-2]

        if self.yminsat == 0.0:
            self.boundaries[1].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(numpy.zeros((0), dtype = 'uint32'), \
                                                                                       self.boundaries[1].left),\
                                                                                       self.boundaries[1].right),\
                                                                                       self.boundaries[1].top), return_index = True)
        else:
            self.boundaries[1].location, ind1 = numpy.unique(numpy.append(numpy.append(numpy.append(self.boundaries[1].bottom, \
                                                                                       self.boundaries[1].left),\
                                                                                       self.boundaries[1].right),\
                                                                                       self.boundaries[1].top), return_index = True)
        ind2 = numpy.argsort(self.boundaries[1].location)
        self.boundaries[1].location = self.boundaries[1].location[ind2]
        self.boundaries[1].areas = numpy.append(numpy.append(numpy.append(b_areas,\
                                                                          l_areas),\
                                                                          r_areas),\
                                                                          t_areas)[ind1][ind2]
        self.boundaries[1].directions = numpy.append(numpy.append(numpy.append(b_directions,\
                                                                               l_directions),\
                                                                               r_directions),\
                                                                               t_directions)[ind1][ind2]

        adjacent_nodes = numpy.append(numpy.append(numpy.append(b_adjacent,\
                                                                l_adjacent),\
                                                                r_adjacent),\
                                                                t_adjacent)[ind1][ind2]

        adjacent_nodes = location_indexes_inv(adjacent_nodes, store = True, location = self.boundaries[1].location)
        self.boundaries[1].adjacent = [{self.boundaries[1].directions[i]:adjacent_nodes[i]} for i in range(len(adjacent_nodes))]
        #Corners
        corners = location_indexes_inv([self.boundaries[1].bottom[-1], self.boundaries[1].left[-1]], store = False)
        self.boundaries[1].adjacent[corners[1]].update({2: self.boundaries[1].adjacent[corners[1]][3]})
        if self.yminsat > 0.0:
            self.boundaries[1].adjacent[0].update({l_directions[0]: self.nxsat})
            self.boundaries[1].adjacent[corners[0]].update({1: self.boundaries[1].adjacent[corners[0]][0]})
        self.boundaries[1].adjacent[-1].update({2:self.boundaries[1].adjacent[-1][1]})
        #Little correction to make the bottom-left node account as an impact in the left side of the spacecraft
        self.boundaries[1].directions[0] = int(3)

        if self.yminsat == 0.0:
            self.boundaries[1].ind_inner = numpy.concatenate(tuple(numpy.arange(self.boundaries[1].left[i]+1, self.boundaries[1].right[i])\
                                                              for i in range(0, int(self.nysat-1))))
        else:
            self.boundaries[1].ind_inner = numpy.concatenate(tuple(numpy.arange(self.boundaries[1].left[i]+1, self.boundaries[1].right[i])\
                                                              for i in range(1, int(self.nysat-1))))


        #Volume
        numpy.divide.at(self.volumes , self.boundaries[1].location, 2)
        #Corners of the satellite
        numpy.multiply.at(self.volumes, [self.boundaries[1].top[0],\
                                         self.boundaries[1].top[-1]], 3/2)
        if self.yminsat > 0.0:
            numpy.multiply.at(self.volumes, [self.boundaries[1].bottom[0],\
                                             self.boundaries[1].bottom[-1]], 3/2)

        #Locations of borders in the mesh
        self.location = numpy.append(self.boundaries[0].location, self.boundaries[1].location)
        self.location, ind1 = numpy.unique(self.location, return_index = True)
        ind2 = numpy.argsort(self.location)
        self.location = self.location[ind2]
        #Location of the satellite
        self.location_sat = numpy.zeros((0), dtype ='uint32')
        self.area_sat = numpy.zeros((0), dtype ='float32')
        self.direction_sat = numpy.zeros((0), dtype ='uint8')
        self.adjacent_sat = []
        if self.boundaries[0].material != "space":
            self.location_sat = numpy.append(self.location_sat, self.boundaries[0].location)
            self.area_sat = numpy.append(self.area_sat, self.boundaries[0].areas)
            self.direction_sat = numpy.append(self.direction_sat, self.boundaries[0].directions)
            self.adjacent_sat.extend(self.boundaries[0].adjacent)
        if self.boundaries[1].material != "space":
            self.location_sat = numpy.append(self.location_sat, self.boundaries[1].location)
            self.area_sat = numpy.append(self.area_sat, self.boundaries[1].areas)
            self.direction_sat = numpy.append(self.direction_sat, self.boundaries[1].directions)
            self.adjacent_sat.extend(self.boundaries[1].adjacent)
        if self.boundaries[0].material != "space" and boundaries[1].material != "space":
            self.location_sat = self.location
            self.area_sat = self.area_sat[ind1][ind2]
            self.direction_sat = self.direction_sat[ind1][ind2]
            #TODO: Needs to be fixed later. The values of the dictionaries are wrong.
            self.adjacent_sat = [self.adjacent_sat[i] for i in range(len(ind1))]

        #Setting up potential and location_index_inv for use in the main program
        if len(self.location_sat) > 0:
            test = location_indexes_inv([self.sat_i], store = True, location = self.location_sat)[0]
            assert test == 0, "location_indexes_inv is not correctly set up"


class Mesh_2D_rm_recursive(Mesh_recursive, Mesh_2D_rm):
    def __init__(self, xmin, xmax, ymin, ymax,\
                 dx, dy, depth, boundaries,\
                 n_children, n_root, n_id, n_start_ind):
        super(Mesh_recursive, self).__init__(xmin, xmax, ymin, ymax, dx, dy, depth, boundaries)
        super().__init__(n_children, n_root, n_id, n_start_ind, self.type)
        if self.root:
            self.print()

##NOTE: Both functions only make sense when all elements of array or ind are in the same mesh. As such, there's no need to redefine the function here.
#    def arrayToIndex(self, array):
#    return super(Mesh_recursive, self).arrayToIndex(array)
#        
#    def indexToArray(self, ind):
#    return super(Mesh_recursive, self).indexToArray(ind)

    def getPosition(self, ind):
        return self.executeFunctionByIndex(ind, super(Mesh_recursive, self).getPosition, ind = ind)

    def getIndex(self, pos, recursive = False, seedList = None):
        positions = None
        if recursive:
            if type(pos) == numpy.ndarray:
                pos = super().sortPositionsByMeshes(pos, seedList = positions)
            seedList = []
            positions = pos.pop(0)
        else:
            seedList = []
            positions = pos
        seedList.append(super(Mesh_recursive, self).getIndex(positions))
        if recursive:
            for child in self.children:
                child.getIndex(pos, recursive = recursive, seedList = seedList)
            return seedList
        else:
            return seedList[0]
    
#   +indexToArray_general([int] index) [int] index = This array does the same as indexToArray but returns the index according to the flatIndexation in the whole group of meshes.
#       NOTE: It should only be used with the root of the tree.
    def indexToArray_general(self, index):
        array = super(Mesh_recursive, self).indexToArray(index)
        assert self.root, "This function only works with the root node"
        return self.flatIndexing(self.id, array)

    #       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preloaded density and velocity of that node. This will depend on each type of mesh.
    #           Even though the mesh can have children, it is assigning the number of particles in the domain according to its densiy distribution only. This means that the particles 
    #           will not recreate a density distribution with the precission of NGs. Same for the thermal component and average component of the velocity.
    def loadSpeciesVTK(self, species, pic):
        #Preparing things for numpy functions use
        #Volume
        dv = self.dx*self.dy*self.depth
        particles = (species.mesh_values.density[:self.nPoints]*dv/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        index = numpy.repeat(ind, particles)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(index)
        random = numpy.random.rand(*numpy.shape(pos))
        random += numpy.where(random == 0, 1e-3, 0)
        pos += (random-0.5)*numpy.asarray((self.dx,self.dy)).T
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity, recursive = False)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})


class Mesh_2D_rm_separateBorders_recursive(Mesh_recursive, Mesh_2D_rm_separateBorders):
    def __init__(self, xmin, xmax, ymin, ymax,\
                 dx, dy, depth, boundaries,\
                 n_children, n_root, n_id, n_start_ind):
        super(Mesh_recursive, self).__init__(xmin, xmax, ymin, ymax, dx, dy, depth, boundaries)
        super().__init__(n_children, n_root, n_id, n_start_ind, self.type)
        if self.root:
            self.print()

##NOTE: Both functions only make sense when all elements of array or ind are in the same mesh. As such, there's no need to redefine the function here.
#    def arrayToIndex(self, array):
#    return super(Mesh_recursive, self).arrayToIndex(array)
#        
##TODO:
#    def indexToArray(self, ind):
#    return super(Mesh_recursive, self).indexToArray(ind)

    def getPosition(self, ind):
        return self.executeFunctionByIndex(ind, super(Mesh_recursive, self).getPosition, ind = ind)

    def getIndex(self, pos, recursive = False, seedList = None):
        positions = None
        if recursive:
            if type(pos) == numpy.ndarray:
                pos = super().sortPositionsByMeshes(pos, seedList = positions)
            seedList = []
            positions = pos.pop(0)
        else:
            seedList = []
            positions = pos
        seedList.append(super(Mesh_recursive, self).getIndex(positions))
        if recursive:
            for child in self.children:
                child.getIndex(pos, recursive = recursive, seedList = seedList)
            return seedList
        else:
            return seedList[0]
    
#   +indexToArray_general([int] index) [int] index = This array does the same as indexToArray but returns the index according to the flatIndexation in the whole group of meshes.
#       NOTE: It should only be used with the root of the tree.
    def indexToArray_general(self, index):
        array = super(Mesh_recursive, self).indexToArray(index)
        assert self.root, "This function only works with the root node"
        return self.flatIndexing(self.id, array)

    #       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preloaded density and velocity of that node. This will depend on each type of mesh.
    #           Even though the mesh can have children, it is assigning the number of particles in the domain according to its densiy distribution only. This means that the particles 
    #           will not recreate a density distribution with the precission of NGs. Same for the thermal component and average component of the velocity.
    def loadSpeciesVTK(self, species, pic):
        #Preparing things for numpy functions use
        #Volume
        dv = self.dx*self.dy*self.depth
        particles = (species.mesh_values.density[:self.nPoints]*dv/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        index = numpy.repeat(ind, particles)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(index)
        random = numpy.random.rand(*numpy.shape(pos))
        random += numpy.where(random == 0, 1e-3, 0)
        pos += (random-0.5)*numpy.asarray((self.dx,self.dy)).T
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity, recursive = False)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})


class Mesh_2D_rm_sat_recursive(Mesh_recursive, Mesh_2D_rm_sat):
    def __init__(self, xmin, xmax, ymin, ymax,\
                 xminsat, xmaxsat, yminsat, ymaxsat,\
                 dx, dy, depth, boundaries,\
                 n_children, n_root, n_id, n_start_ind):
        super(Mesh_recursive, self).__init__(xmin, xmax, ymin, ymax, xminsat, xmaxsat, yminsat, ymaxsat, dx, dy, depth, boundaries)
        super().__init__(n_children, n_root, n_id, n_start_ind, self.type)
        if self.root:
            self.print()

##NOTE: Both functions only make sense when all elements of array or ind are in the same mesh. As such, there's no need to redefine the function here.
#    def arrayToIndex(self, array):
#    return super(Mesh_recursive, self).arrayToIndex(array)
#        
#    def indexToArray(self, ind):
#    return super(Mesh_recursive, self).indexToArray(ind)

    def getPosition(self, ind):
        return self.executeFunctionByIndex(ind, super(Mesh_recursive, self).getPosition, ind = ind)

    def getIndex(self, pos, recursive = False, seedList = None):
        positions = None
        if recursive:
            if type(pos) == numpy.ndarray:
                pos = super().sortPositionsByMeshes(pos, seedList = positions)
            seedList = []
            positions = pos.pop(0)
        else:
            seedList = []
            positions = pos
        seedList.append(super(Mesh_recursive, self).getIndex(positions))
        if recursive:
            for child in self.children:
                child.getIndex(pos, recursive = recursive, seedList = seedList)
            return seedList
        else:
            return seedList[0]
    
#   +indexToArray_general([int] index) [int] index = This array does the same as indexToArray but returns the index according to the flatIndexation in the whole group of meshes.
#       NOTE: It should only be used with the root of the tree.
    def indexToArray_general(self, index):
        array = super(Mesh_recursive, self).indexToArray(index)
        assert self.root, "This function only works with the root node"
        return self.flatIndexing(self.id, array)

    #       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preloaded density and velocity of that node. This will depend on each type of mesh.
    #           Even though the mesh can have children, it is assigning the number of particles in the domain according to its densiy distribution only. This means that the particles 
    #           will not recreate a density distribution with the precission of NGs. Same for the thermal component and average component of the velocity.
    def loadSpeciesVTK(self, species, pic):
        #Preparing things for numpy functions use
        #Volume
        dv = self.dx*self.dy*self.depth
        particles = (species.mesh_values.density[:self.nPoints]*dv/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        index = numpy.repeat(ind, particles)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(index)
        random = numpy.random.rand(*numpy.shape(pos))
        random += numpy.where(random == 0, 1e-3, 0)
        pos += (random-0.5)*numpy.asarray((self.dx,self.dy)).T
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity, recursive = False)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})


class Mesh_2D_rm_sat_HET_recursive(Mesh_recursive, Mesh_2D_rm_sat_HET):
    def __init__(self, xmin, xmax, ymin, ymax,\
                 xminsat, xmaxsat, yminsat, ymaxsat,\
                 dx, dy, depth, boundaries,\
                 n_children, n_root, n_id, n_start_ind):
        super(Mesh_recursive, self).__init__(xmin, xmax, ymin, ymax, xminsat, xmaxsat, yminsat, ymaxsat, dx, dy, depth, boundaries)
        super().__init__(n_children, n_root, n_id, n_start_ind, self.type)
        if self.root:
            self.print()

##NOTE: Both functions only make sense when all elements of array or ind are in the same mesh. As such, there's no need to redefine the function here.
#    def arrayToIndex(self, array):
#    return super(Mesh_recursive, self).arrayToIndex(array)
#        
#    def indexToArray(self, ind):
#    return super(Mesh_recursive, self).indexToArray(ind)

    def getPosition(self, ind):
        return self.executeFunctionByIndex(ind, super(Mesh_recursive, self).getPosition, ind = ind)

    def getIndex(self, pos, recursive = False, seedList = None):
        positions = None
        if recursive:
            if type(pos) == numpy.ndarray:
                pos = super().sortPositionsByMeshes(pos, seedList = positions)
            seedList = []
            positions = pos.pop(0)
        else:
            seedList = []
            positions = pos
        seedList.append(super(Mesh_recursive, self).getIndex(positions))
        if recursive:
            for child in self.children:
                child.getIndex(pos, recursive = recursive, seedList = seedList)
            return seedList
        else:
            return seedList[0]
    
#   +indexToArray_general([int] index) [int] index = This array does the same as indexToArray but returns the index according to the flatIndexation in the whole group of meshes.
#       NOTE: It should only be used with the root of the tree.
    def indexToArray_general(self, index):
        array = super(Mesh_recursive, self).indexToArray(index)
        assert self.root, "This function only works with the root node"
        return self.flatIndexing(self.id, array)

    #       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preloaded density and velocity of that node. This will depend on each type of mesh.
    #           Even though the mesh can have children, it is assigning the number of particles in the domain according to its densiy distribution only. This means that the particles 
    #           will not recreate a density distribution with the precission of NGs. Same for the thermal component and average component of the velocity.
    def loadSpeciesVTK(self, species, pic):
        #Preparing things for numpy functions use
        #Volume
        dv = self.dx*self.dy*self.depth
        particles = (species.mesh_values.density[:self.nPoints]*dv/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        index = numpy.repeat(ind, particles)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(index)
        random = numpy.random.rand(*numpy.shape(pos))
        random += numpy.where(random == 0, 1e-3, 0)
        pos += (random-0.5)*numpy.asarray((self.dx,self.dy)).T
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity, recursive = False)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})


class Mesh_2D_cm_recursive(Mesh_recursive, Mesh_2D_cm):
    def __init__(self, xmin, xmax, ymin, ymax,\
                 dx, dy, boundaries,\
                 n_children, n_root, n_id, n_start_ind):
        super(Mesh_recursive, self).__init__(xmin, xmax, ymin, ymax, dx, dy, boundaries)
        super().__init__(n_children, n_root, n_id, n_start_ind, self.type)
        if self.root:
            self.print()

##NOTE: Both functions only make sense when all elements of array or ind are in the same mesh. As such, there's no need to redefine the function here.
#    def arrayToIndex(self, array):
#    return super(Mesh_recursive, self).arrayToIndex(array)
#        
#    def indexToArray(self, ind):
#    return super(Mesh_recursive, self).indexToArray(ind)

    def getPosition(self, ind):
        return self.executeFunctionByIndex(ind, super(Mesh_recursive, self).getPosition, ind = ind)

    def getIndex(self, pos, recursive = False, seedList = None):
        positions = None
        if recursive:
            if type(pos) == numpy.ndarray:
                pos = super().sortPositionsByMeshes(pos, seedList = positions)
            seedList = []
            positions = pos.pop(0)
        else:
            seedList = []
            positions = pos
        seedList.append(super(Mesh_recursive, self).getIndex(positions))
        if recursive:
            for child in self.children:
                child.getIndex(pos, recursive = recursive, seedList = seedList)
            return seedList
        else:
            return seedList[0]
    
#   +indexToArray_general([int] index) [int] index = This array does the same as indexToArray but returns the index according to the flatIndexation in the whole group of meshes.
#       NOTE: It should only be used with the root of the tree.
    def indexToArray_general(self, index):
        array = super(Mesh_recursive, self).indexToArray(index)
        assert self.root, "This function only works with the root node"
        return self.flatIndexing(self.id, array)

    #       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preloaded density and velocity of that node. This will depend on each type of mesh.
    #           Even though the mesh can have children, it is assigning the number of particles in the domain according to its densiy distribution only. This means that the particles 
    #           will not recreate a density distribution with the precission of NGs. Same for the thermal component and average component of the velocity.
    def loadSpeciesVTK(self, species, pic, prec = 10**-c.INDEX_PREC):
        #Number of particles created
        particles = (species.mesh_values.density[:self.nPoints]*self.volumes[:self.nPoints]/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(ind)
        rmin = numpy.where(pos[:,1] == self.ymin, pos[:,1], pos[:,1]-self.dy/2)
        rmax = numpy.where(pos[:,1] == self.ymax, pos[:,1], pos[:,1]+self.dy/2)
        pos = numpy.repeat(pos, particles, axis = 0)
        pos[:,1] = cmt.randomYPositions_2D_cm(particles, rmin, rmax)
        random = numpy.random.rand(numpy.shape(pos)[0])
        shifts = numpy.where((pos[:,0]-self.xmin) < prec, random*self.dx/2, (random-0.5)*self.dx)
        shifts -= numpy.where((pos[:,0]-self.xmax) > -prec, random*self.dx/2, 0)
        pos[:,0] += shifts
        pos[:,1] += numpy.where((pos[:,1] - self.ymin) < prec, prec, 0)
        pos[:,1] += numpy.where((pos[:,1] - self.ymax) > -prec, -prec, 0)
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity, recursive = False)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})


class Mesh_2D_cm_sat_recursive(Mesh_recursive, Mesh_2D_cm_sat):
    def __init__(self, xmin, xmax, ymin, ymax,\
                 xminsat, xmaxsat, yminsat, ymaxsat,\
                 dx, dy, boundaries,\
                 n_children, n_root, n_id, n_start_ind):
        super(Mesh_recursive, self).__init__(xmin, xmax, ymin, ymax, xminsat, xmaxsat, yminsat, ymaxsat, dx, dy, boundaries)
        super().__init__(n_children, n_root, n_id, n_start_ind, self.type)
        if self.root:
            self.print()

##NOTE: Both functions only make sense when all elements of array or ind are in the same mesh. As such, there's no need to redefine the function here.
#    def arrayToIndex(self, array):
#    return super(Mesh_recursive, self).arrayToIndex(array)
#        
#    def indexToArray(self, ind):
#    return super(Mesh_recursive, self).indexToArray(ind)

    def getPosition(self, ind):
        return self.executeFunctionByIndex(ind, super(Mesh_recursive, self).getPosition, ind = ind)

    def getIndex(self, pos, recursive = False, seedList = None):
        positions = None
        if recursive:
            if type(pos) == numpy.ndarray:
                pos = super().sortPositionsByMeshes(pos, seedList = positions)
            seedList = []
            positions = pos.pop(0)
        else:
            seedList = []
            positions = pos
        seedList.append(super(Mesh_recursive, self).getIndex(positions))
        if recursive:
            for child in self.children:
                child.getIndex(pos, recursive = recursive, seedList = seedList)
            return seedList
        else:
            return seedList[0]
    
#   +indexToArray_general([int] index) [int] index = This array does the same as indexToArray but returns the index according to the flatIndexation in the whole group of meshes.
#       NOTE: It should only be used with the root of the tree.
    def indexToArray_general(self, index):
        array = super(Mesh_recursive, self).indexToArray(index)
        assert self.root, "This function only works with the root node"
        return self.flatIndexing(self.id, array)

    #       +loadSpeciesVTK(self, species) = It creates particles around every node that can match the preloaded density and velocity of that node. This will depend on each type of mesh.
    #           Even though the mesh can have children, it is assigning the number of particles in the domain according to its densiy distribution only. This means that the particles 
    #           will not recreate a density distribution with the precission of NGs. Same for the thermal component and average component of the velocity.
    def loadSpeciesVTK(self, species, pic, prec = 10**(-c.INDEX_PREC)):
        #Number of particles created
        particles = (species.mesh_values.density[:self.nPoints]*self.volumes[:self.nPoints]/species.spwt).astype(int)
        ind = numpy.arange(self.nPoints)
        #Setting up positions
        pos = super(Mesh_recursive, self).getPosition(ind)
        rmin = numpy.where(pos[:,1] == self.ymin, pos[:,1], pos[:,1]-self.dy/2)
        rmax = numpy.where(pos[:,1] == self.ymax, pos[:,1], pos[:,1]+self.dy/2)
        pos = numpy.repeat(pos, particles, axis = 0)
        pos[:,1] = cmt.randomYPositions_2D_cm(particles, rmin, rmax)
        random = numpy.random.rand(numpy.shape(pos)[0])
        shifts = numpy.where((pos[:,0]-self.xmin) < prec, random*self.dx/2, (random-0.5)*self.dx)
        shifts -= numpy.where((pos[:,0]-self.xmax) > -prec, random*self.dx/2, 0)
        pos[:,0] += shifts
        pos[:,1] += numpy.where((pos[:,1] - self.ymin) < prec, prec, 0)
        pos[:,1] += numpy.where((pos[:,1] - self.ymax) > -prec, -prec, 0)
        #Adding particles and thermal velocity
        vel = self.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[:self.nPoints], species.m), particles)
        self.boundaries[0].addParticles(species, pos, vel)
        #Clearing particles outside of boundaries
        for boundary in self.boundaries:
            boundary.applyParticleBoundary(species, 'open', old_position = None)
        #Setting up shifted velocities
        np = species.part_values.current_n
        species.part_values.velocity[:np] += pic.gather(species.part_values.position[:np], species.mesh_values.velocity, recursive = False)
        #Update trackers
        self.boundaries[0].updateTrackers(species, species.part_values.current_n)

    def print(self):
        cwd = os.path.split(os.getcwd())[0]
        vtkstring = cwd+'/results/mesh'
        self.saveVTK(vtkstring, {'volume' : self.vtkOrdering(self.volumes)})
