#Data structures that contain PIC numerical methods
import copy
from numba import jit, njit, prange
import numpy
import pdb

import accelerated_functions as af
from Boundaries.inner_2D_rectangular import Inner_2D_Rectangular
import constants as c
from solver import location_indexes_inv
import mesh as m

#PIC (Abstract)(Association between Mesh and PIC):
#
#Definition = Indicate the methods that all PIC classes have to implement. Each PIC concrete class will depend on the type of mesh, as well as the type of PIC algorithm implemented.
#Attributes:
#	+mesh (Mesh) = Instance of the mesh class for later use of getIndex().
#Methods:
#	+scatter([double, double] positions, [double] values, [double] field) = Receives the positions of the particles, and makes scatter procedure, calculating the values of field for each node.
#	+gather([double, double] positions, [double, double] field): [double, double]field_p = Calculates values of the field in particles' positions, returning these values in an array as long as positions,
#           The columns are the (x,y,z) components
#       +scatterDiffSq([double, double] positions, [double] values, [double] array_diff, [double] field) = Makes a PIC averaging over 
#           (values-array_diff)**2 in all the nodes involved for every particle in positions. 
#           Thus, array_diff an field need to have the same dimension, and values have to be broadcastable to the len of positions.
class PIC (object):
    def __init__(self, mesh):
        self.mesh = mesh

    def scatter(self, positions, values, field):
        pass

    def scatterDiffSq(self, positions, values, array_diff, field):
        pass

    def gather(self, positions, field):
        pass


class PIC_recursive(object):
    def __init__(self, n_children, n_root, n_type):
        n_type += " - Recursive"
        self.children = n_children
        self.root = n_root

#   +getTreeSize(): int = This function returns and integer indicating the number of nodes in the tree starting (and including) the current node.
    def getTreeSize():
        count = 0
        for child in self.children:
            count += child.getTreeSize()
        return count + 1


    def scatter(self, positions, values, field):
        pass

    def scatterDiffSq(self, positions, values, weights, array_diff, field):
        pass

    def gather(self, positions, field, recursive = True):
        pass
        


#PIC_2D_rm1o (Inherits from PIC):
#
#Definition = PIC class for rm10 ('r'ectangular 'm'esh, '1'st 'o'rder implementation).
#Attributes:
#	+PIC class attributes.
#Methods:
#	+Implementation of PIC class methods.
#       +scatter_1D([double, double] positions, [double] values, [double] field, [ind] location, [ind] add_info) = This method scatters the 'values' through a set of nodes
#           identified by 'location'. 'position' indicates the positions of the particles in the domain, and the positions must comply to lie in the 1D region delineated by
#           'location'. The scatter process is stored in 'field' which is a ndarray with the same size as 'location' and where its elements refer to the nodes of 'location',
#           ordered increasingly index-based. 'add_info' is additional information that can be used for speeding up the function. In this case, it indicates whether the
#           particle is located in a vertical or horizontal line of the delimited mesh.
#	+scatterDensity (Species) = return densities of that species in every node of the mesh.
#	+scatterVelocity (Species) = return velocities of that species in every node of the mesh.
#	+scatterFlux((Species) species, hit) = return flux of particles of that species into every indicated node (not all the mesh).
#       +scatterOutgoingFlux((Species) species, hit) = it works as scatterFlux but it scatters the particles and velocities as if the particles are going out of the delimited region.
class PIC_2D_rm1o(PIC):
    def __init__(self, mesh):
        super(PIC_2D_rm1o, self).__init__(mesh)
        self.type = "PIC method for a rectangular 2D mesh, making interpolations at first order"

    def scatter_1D(self, positions, values, field, location, add_info, prec = 10**(-c.INDEX_PREC)):
        #Getting mesh coordinates
        mc = self.mesh.getIndex(positions)

        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        array_1D = location_indexes_inv(array, store = False)
        horizontal = numpy.equal(add_info % 2, 0)
        vertical = numpy.logical_not(horizontal)
        dindex = numpy.where(horizontal, mc[:,0]-index[:,0], mc[:,1]-index[:,1])
        #n_index = numpy.where(horizontal, array_1D+1, array_1D+2)
        #n_index = numpy.where(numpy.logical_and(vertical, array_1D == 0), n_index-2+self.mesh.nxsat, n_index)
        #n_index = numpy.where(numpy.logical_and(vertical, array_1D == len(location)-1-self.mesh.nxsat), n_index-2+self.mesh.nxsat, n_index)
        n_index = numpy.asarray([self.mesh.adjacent_sat[array_1D[i]][add_info[i]] for i in range(len(array_1D))], dtype = numpy.uint16)

        filter_i = dindex > prec

        values = values*numpy.ones_like((dindex)) if type(values) != numpy.ndarray else values
        numpy.add.at(field, array_1D, (1-dindex)*values)
        numpy.add.at(field, n_index[filter_i], dindex[filter_i]*values[filter_i])

    def scatter(self, positions, values, field, prec = 10**(-c.INDEX_PREC)):
        #Getting mesh coordinates
        mc = self.mesh.getIndex(positions)
        
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        di = numpy.around(mc[:,0] - index[:,0], decimals = c.INDEX_PREC)
        dj = numpy.around(mc[:,1] - index[:,1], decimals = c.INDEX_PREC)

        filter_i = di > prec 
        filter_j = dj > prec 
        
        numpy.add.at(field, array, (1-di)*(1-dj)*values)
        numpy.add.at(field, array[filter_i]+1, di[filter_i]*(1-dj[filter_i])*values[filter_i])
        numpy.add.at(field, array[filter_j]+self.mesh.nx, (1-di[filter_j])*dj[filter_j]*values[filter_j])
        numpy.add.at(field, array[numpy.logical_and(filter_i, filter_j)]+self.mesh.nx+1, di[numpy.logical_and(filter_i, filter_j)]*dj[numpy.logical_and(filter_i, filter_j)]*values[numpy.logical_and(filter_i, filter_j)])

#       +scatterDiffSq([double, double] positions, [double] values, [double] array_diff, [double] field) = Makes a PIC averaging over 
#           weights*(values-array_diff)**2 in all the nodes involved for every particle in positions. 
#           Thus, array_diff an field need to have the same dimension, and values have to be broadcastable to the len of positions.
    def scatterDiffSq(self, positions, values, weights, array_diff, field, prec = 10**(-c.INDEX_PREC)):
        #Getting mesh coordinates
        mc = self.mesh.getIndex(positions)
        
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        di = numpy.around(mc[:,0] - index[:,0], decimals = c.INDEX_PREC)
        dj = numpy.around(mc[:,1] - index[:,1], decimals = c.INDEX_PREC)

        filter_i = di > prec 
        filter_j = dj > prec 
        filter_ij = numpy.logical_and(filter_i, filter_j)
        
        numpy.add.at(field, array, (1-di)*(1-dj)*(values-array_diff[array])*(values-array_diff[array])*weights)
        numpy.add.at(field, array[filter_i]+1, di[filter_i]*(1-dj[filter_i])*\
                (values[filter_i]-array_diff[array[filter_i]+1])*(values[filter_i]-array_diff[array[filter_i]+1])*weights[filter_i])
        numpy.add.at(field, array[filter_j]+self.mesh.nx, (1-di[filter_j])*dj[filter_j]*\
                (values[filter_j]-array_diff[array[filter_j]+self.mesh.nx])*(values[filter_j]-array_diff[array[filter_j]+self.mesh.nx])*weights[filter_j])
        numpy.add.at(field, array[filter_ij]+self.mesh.nx+1, di[filter_ij]*dj[filter_ij]*\
                (values[filter_ij]-array_diff[array[filter_ij]+self.mesh.nx+1])*(values[filter_ij]-array_diff[array[filter_ij]+self.mesh.nx+1])*weights[filter_ij])

#       +scatterDensity (Species) = return densities of that species in every node of the mesh.
    def scatterDensity(self, species):
        #reset the density
        species.mesh_values.density *= 0
    
        #scatter particles to the mesh
        self.scatter(species.part_values.position[:species.part_values.current_n], species.part_values.spwt[:species.part_values.current_n], species.mesh_values.density)
        
        #divide by cell volume
        species.mesh_values.density /= self.mesh.volumes
    
#       +scatterVelocity (Species) = return velocities of that species in every node of the mesh.
    def scatterSpeed(self, species):
        #reset the velocity
        species.mesh_values.velocity *= 0
        #scatter particles to the mesh
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            self.scatter(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n,dim]*species.part_values.spwt[:species.part_values.current_n], species.mesh_values.velocity[:,dim])
            #Finalizing velocity
            species.mesh_values.velocity[:,dim] *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes)

#       +scatterTemperature(Species) = return temperature of that species in every node of the mesh.
    def scatterTemperature(self, species):
        #Reset temperature
        species.mesh_values.temperature *= 0
        #Scatter temperature
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            self.scatterDiffSq(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n, dim], species.part_values.spwt[:species.part_values.current_n], \
                    species.mesh_values.velocity[:,dim], species.mesh_values.temperature)
        #Finalizing temperature
        species.mesh_values.temperature *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes*species.m/c.K/2)
    
#       +acatterFlux = 
    def scatterFlux(self, species, hit):
        #Prepare fields
        species.mesh_values.flux *= 0
        species.mesh_values.accDensity *= self.mesh.volumes[species.mesh_values.fluxind]
        #Parameter to use
        positions = hit[0][:,:2]
        add_info = hit[0][:,2].astype(int)
        spwts = hit[0][3]
        #Scatter flux
        self.scatter_1D(positions, spwts*species.q, species.mesh_values.flux, species.mesh_values.fluxind, add_info)
        self.scatter_1D(positions, spwts, species.mesh_values.accDensity, species.mesh_values.fluxind, add_info)
        species.mesh_values.flux /= (self.mesh.area_sat)*species.dt
        species.mesh_values.accDensity /= self.mesh.volumes[species.mesh_values.fluxind]

#       +acatterOutgoingFlux = 
    def scatterOutgoingFlux(self, species, hit):
        #Prepare fields
        species.mesh_values.outgoing_flux *= 0
        species.mesh_values.accDensity *= self.mesh.volumes[species.mesh_values.fluxind]
        #Parameter to use
        positions = hit[0][:,:2]
        add_info = hit[0][:,2].astype(int)
        spwts = hit[0][3]
        #Scatter flux
        self.scatter_1D(positions, -spwts*species.q, species.mesh_values.outgoing_flux, species.mesh_values.fluxind, add_info)
        self.scatter_1D(positions, -spwts, species.mesh_values.accDensity, species.mesh_values.fluxind, add_info)
        species.mesh_values.flux /= (self.mesh.area_sat)*species.dt
        species.mesh_values.accDensity /= self.mesh.volumes[species.mesh_values.fluxind]

##	+gather([double, double] positions, [double, double] field): [double, double]field_p = Calculates values of the field in particles' positions, returning these values in an array as long as positions,
##                                                                                               The columns are the (x,y,z) positions
#    def gather(self, positions, field, prec = 10**(-c.INDEX_PREC)):
#        dim = field.shape[1]
#
#        #Getting mesh coordinates
#        mc = self.mesh.getIndex(positions)
#        index = mc.astype(int)
#        array = self.mesh.indexToArray(index)
#
#        values, di, dj, filter_i, filter_j = af.gather_p(mc, index, dim, prec, c.INDEX_PREC)
#
#        #From mesh to particles, summing in every dimension through different columns
#        values += field[array,:]*(1-di)*(1-dj)
#        values[filter_i] += field[array[filter_i]+1,:]*di[filter_i]*(1-dj[filter_i])
#        values[filter_j] += field[array[filter_j]+self.mesh.nx,:]*(1-di[filter_j])*dj[filter_j]
#        values[numpy.logical_and(filter_i, filter_j)] += field[array[numpy.logical_and(filter_i, filter_j)]+self.mesh.nx+1,:]*di[numpy.logical_and(filter_i, filter_j)]*dj[numpy.logical_and(filter_i, filter_j)]
#
#        return values

#	+gather([double, double] positions, [double, double] field): [double, double]field_p = Calculates values of the field in particles' positions, returning these values in an array as long as positions,
#                                                                                               The columns are the (x,y,z) positions
    def gather(self, positions, field, prec = 10**(-c.INDEX_PREC)):

        #Getting mesh coordinates
        mc = self.mesh.getIndex(positions)
        #Creating the array
        values = numpy.zeros((numpy.shape(positions)[0], numpy.shape(field)[1]))
        
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        di = numpy.around(mc[:,0] - index[:,0], decimals = c.INDEX_PREC)
        dj = numpy.around(mc[:,1] - index[:,1], decimals = c.INDEX_PREC)

        #Dealing with nodes
        filter_i = di > prec
        filter_j = dj > prec

        #NOTE: Maybe this can be further optmized later
        di = numpy.repeat(di[:,None], numpy.shape(field)[1], axis = 1)
        dj = numpy.repeat(dj[:,None], numpy.shape(field)[1], axis = 1)

        #From mesh to particles, summing in every dimension through different columns
        values += field[array,:]*(1-di)*(1-dj)
        values[filter_i] += field[array[filter_i]+1,:]*di[filter_i]*(1-dj[filter_i])
        values[filter_j] += field[array[filter_j]+self.mesh.nx,:]*(1-di[filter_j])*dj[filter_j]
        values[numpy.logical_and(filter_i, filter_j)] += field[array[numpy.logical_and(filter_i, filter_j)]+self.mesh.nx+1,:]*di[numpy.logical_and(filter_i, filter_j)]*dj[numpy.logical_and(filter_i, filter_j)]

        return values


#PIC_2D_cm1o (Inherits from PIC_2D_rm1o):
#
#Definition = PIC class for cm1o ('c'ylindrical 'm'esh, '1'st 'o'rder implementation).
#Attributes:
#	+PIC class attributes.
#Methods:
#	+Implementation of PIC class methods.
#       +scatter_1D([double, double] positions, [double] values, [double] field, [ind] location, [ind] add_info) = This method scatters the 'values' through a set of nodes
#           identified by 'location'. 'position' indicates the positions of the particles in the domain, and the positions must comply to lie in the 1D region delineated by
#           'location'. The scatter process is stored in 'field' which is a ndarray with the same size as 'location' and where its elements refer to the nodes of 'location',
#           ordered increasingly index-based. 'add_info' is additional information that can be used for speeding up the function. In this case, it indicates whether the
#           particle is located in a vertical or horizontal line of the delimited mesh.
#	+scatterDensity (Species) = return densities of that species in every node of the mesh.
#	+scatterVelocity (Species) = return velocities of that species in every node of the mesh.
#	+scatterFlux((Species) species, hit) = return flux of particles of that species into every indicated node (not all the mesh).
#       +scatterOutgoingFlux((Species) species, hit) = it works as scatterFlux but it scatters the particles and velocities as if the particles are going out of the delimited region.
class PIC_2D_cm1o(PIC_2D_rm1o):
    def __init__(self, mesh):
        super(PIC_2D_cm1o, self).__init__(mesh)
        self.type = "PIC method for a cylindrical 2D mesh, making interpolations at first order"

#       +scatterTemperature(Species) = return temperature of that species in every node of the mesh.
    def scatterTemperature(self, species):
        #Reset temperature
        species.mesh_values.temperature *= 0
        #Scatter temperature
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            self.scatterDiffSq(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n, dim], species.part_values.spwt[:species.part_values.current_n], \
                    species.mesh_values.velocity[:,dim], species.mesh_values.temperature)
        #Finalizing temperature
        species.mesh_values.temperature *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes*species.m/c.K/3)


#PIC_2D_rm1o_recursive(Inherits from PIC_recursive and PIC_2D_rm1o):
#
#Definition = PIC class for rm10 ('r'ectangular 'm'esh, '1'st 'o'rder implementation), with the tools for recursion.
#Attributes:
#	+PIC class attributes.
#       +PIC_recursive attributes.
#Methods:
#	+Implementation of PIC class methods.
#	+scatterDensity (Species) = return densities of that species in every node of the mesh.
#	+scatterVelocity (Species) = return velocities of that species in every node of the mesh.
#	+scatterFlux = return flux of particles of that species into every indicated node (not all the mesh).
class PIC_2D_rm1o_recursive(PIC_recursive, PIC_2D_rm1o):
    def __init__(self, mesh, n_children, n_root):
        super(PIC_recursive, self).__init__(mesh)
        super().__init__(n_children, n_root, self.type)

    def scatter(self, positions, values, field, indices = None, surface = False):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [], surface = surface)
            values = [values[i] for i in pos_ind]
            indices = self.mesh.sortArrayByMeshes(numpy.arange(len(field)))
        pos = positions.pop(0)
        val = values.pop(0)
        ind = indices.pop(0)
        for child in self.children:
            ind_i = copy.copy(indices[0])
            child.scatter(positions, values, field, indices = indices)
            nonzero = numpy.flatnonzero(field[ind_i])
            pos = numpy.append(pos, child.mesh.getPosition(nonzero), axis = 0)
            val = numpy.append(val, field[ind_i][nonzero])
        temp = field[ind]
        super(PIC_recursive, self).scatter(pos, val, temp)
        field[ind] += temp

    def scatter_1D(self, positions, values, field, location, add_info):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [], surface = True)
            values = [values[i] for i in pos_ind]
            add_info = [add_info[i] for i in pos_ind]
            location = self.mesh.sortIndexByMeshes(location, shift = False)
        pos = positions.pop(0)
        val = values.pop(0)
        info = add_info.pop(0)
        loc = location.pop(0)
        for child in self.children:
            loc_i = copy.copy(location[0])
            child.scatter_1D(positions, values, field, location, add_info)
            array_1D = location_indexes_inv(loc_i, store = False)
            nonzero = numpy.flatnonzero(field[array_1D])
            array_1D = array_1D[nonzero]
            pos = numpy.append(pos, child.mesh.getPosition(child.mesh.location_sat[nonzero]), axis = 0)
            val = numpy.append(val, field[array_1D])
            info = numpy.append(info, child.mesh.direction_sat[nonzero])
        if len(self.children) > 0:
            field[location_indexes_inv(loc, store = False)] *= 0
        self.scatter_1D_aux(pos, val, field, loc, info)

    def scatter_1D_aux(self, positions, values, field, general_location, add_info, prec = 10**(-c.INDEX_PREC)):
        #Creating 1-1 connection between local satellite nodes and general nodes
        dic = {loc_i: gen_i for loc_i, gen_i in zip(self.mesh.location_sat, general_location)}
        #Getting mesh coordinates
        mc = self.mesh.getIndex(positions)
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)

        #Calcualting array indexes with respect to the whole domain
        array_gen = [dic[i] for i in array]
        #Getting the correct indexes for the field
        array_1D = location_indexes_inv(array_gen, store = False)
        horizontal = numpy.equal(add_info % 2, 0)
        vertical = numpy.logical_not(horizontal)
        dindex = numpy.where(horizontal, mc[:,0]-index[:,0], mc[:,1]-index[:,1])

        filter_1D = dindex > prec

        #Adapting horizontal and vertical such that n_index works for inner boundaries as well as outer boundaries
        #mask = numpy.ones_like(add_info, dtype = numpy.bool_)
        #for boundary in self.mesh.boundaries:
        #    if boundary.type == Inner_2D_Rectangular.type:
        #        mask[numpy.flatnonzero(numpy.isin(array, boundary.location))] = False
        #horizontal = numpy.logical_or(horizontal, mask)
        #vertical = numpy.logical_and(vertical, numpy.logical_not(mask))
        #n_index = numpy.where(horizontal, array_1D+1, array_1D+2)
        #n_index = numpy.where(numpy.logical_and(vertical, array_1D == 0), n_index-2+self.mesh.nxsat, n_index)
        #n_index = numpy.where(numpy.logical_and(vertical, array_1D == len(self.mesh.location_sat)-1-self.mesh.nxsat), n_index-2+self.mesh.nxsat, n_index)

        # Finding the other node involved
        temp_location_indexes_inv = {locations: indexes[0] for indexes, locations in numpy.ndenumerate(self.mesh.location_sat)}
        loc_ind = numpy.asarray([temp_location_indexes_inv.get(ind_i) for ind_i in array], dtype = numpy.uint16)
        n_index = [self.mesh.adjacent_sat[loc_ind[i]][add_info[i]] for i in range(len(array_1D))]
        n_index = location_indexes_inv([dic[self.mesh.location_sat[ind]] for ind in n_index], store = False)

        #scatter process
        numpy.add.at(field, array_1D, (1-dindex)*values)
        numpy.add.at(field, n_index[filter_1D], dindex[filter_1D]*values[filter_1D])

#NOTE: I created here a different way of calculating the temperature in the children meshes. Probably is wrong.
#    def scatterDiffSq(self, positions, values, weights, array_diff, field, indices = None):
#        if self.root == True:
#            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [])
#            values = [values[i] for i in pos_ind]
#            weights = [weights[i] for i in pos_ind]
#            indices = self.mesh.sortArrayByMeshes(numpy.arange(len(field)))
#        pos = positions.pop(0)
#        val = values.pop(0)
#        w = weights.pop(0)
#        ind = indices.pop(0)
#        pos_c = numpy.zeros((0,numpy.shape(pos)[1]))
#        val_c = numpy.zeros((0))
#        for child in self.children:
#            ind_i = copy.copy(indices[0])
#            child.scatterDiffSq(positions, values, weights, array_diff, field, indices = indices)
#            nonzero = numpy.flatnonzero(field[ind_i])
#            pos_c = numpy.append(pos_c, child.mesh.getPosition(nonzero), axis = 0)
#            val_c = numpy.append(val_c, field[ind_i][nonzero])
#        temp = field[ind]
#        temp2 = numpy.zeros_like(temp)
#        super(PIC_recursive, self).scatter(pos_c, val_c, temp)
#        super(PIC_recursive, self).scatter(pos_c, numpy.ones_like(pos_c[:,0]), temp2)
#        temp *= numpy.where(temp2 != 0, 1/temp2, 0)
#        super(PIC_recursive, self).scatterDiffSq(pos, val, w, array_diff[ind], temp)
#        field[ind] += temp

    def scatterDiffSq(self, positions, values, weights, array_diff, field, indices = None):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [])
            values = [values[i] for i in pos_ind]
            weights = [weights[i] for i in pos_ind]
            indices = self.mesh.sortArrayByMeshes(numpy.arange(len(field)))
        pos = positions.pop(0)
        val = values.pop(0)
        w = weights.pop(0)
        ind = indices.pop(0)
        pos_c = numpy.zeros((0,numpy.shape(pos)[1]))
        val_c = numpy.zeros((0))
        for child in self.children:
            ind_i = copy.copy(indices[0])
            child.scatterDiffSq(positions, values, weights, array_diff, field, indices = indices)
            nonzero = numpy.flatnonzero(field[ind_i])
            pos_c = numpy.append(pos_c, child.mesh.getPosition(nonzero), axis = 0)
            val_c = numpy.append(val_c, field[ind_i][nonzero])
        temp = field[ind]
        super(PIC_recursive, self).scatter(pos_c, val_c, temp)
        super(PIC_recursive, self).scatterDiffSq(pos, val, w, array_diff[ind], temp)
        field[ind] += temp

    def scatterDensity(self, species):
        #reset the density
        species.mesh_values.density *= 0
    
        #scatter particles to the mesh
        self.scatter(species.part_values.position[:species.part_values.current_n], species.part_values.spwt[:species.part_values.current_n], species.mesh_values.density)
        
        #divide by cell volume and add the accumulated charge from material surfaces in the domain
        species.mesh_values.density /= self.mesh.volumes

    def scatterSpeed(self, species):
        #reset the velocity
        species.mesh_values.velocity *= 0
    
        #scatter particles to the mesh
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            self.scatter(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n,dim]*species.part_values.spwt[:species.part_values.current_n], species.mesh_values.velocity[:,dim])
            #Finalizing velocity
            species.mesh_values.velocity[:,dim] *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes)

    def scatterTemperature(self, species):
        #Reset temperature
        species.mesh_values.temperature *= 0
        #Scatter temperature
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            temp = numpy.zeros_like(species.mesh_values.temperature)
            self.scatterDiffSq(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n, dim], species.part_values.spwt[:species.part_values.current_n], species.mesh_values.velocity[:,dim], temp)
            species.mesh_values.temperature += temp
        #Finalizing temperature
        species.mesh_values.temperature *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes*species.m/c.K/2)

    def scatterFlux(self, species, hit):
        #Prepare fields
        species.mesh_values.flux *= 0
        species.mesh_values.accDensity *= self.mesh.volumes[species.mesh_values.fluxind]
        #Parameter to use
        positions = hit[0][:,:2]
        add_info = hit[0][:,2].astype(int)
        spwts = hit[0][:,3]
        #Scatter flux
        self.scatter_1D(positions, spwts*species.q, species.mesh_values.flux, species.mesh_values.fluxind, add_info)
        self.scatter_1D(positions, spwts, species.mesh_values.accDensity, species.mesh_values.fluxind, add_info)
        species.mesh_values.flux /= (self.mesh.overall_area_sat)*species.dt
        species.mesh_values.accDensity /= self.mesh.volumes[species.mesh_values.fluxind]

    def scatterOutgoingFlux(self, species, hit):
        #Prepare fields
        species.mesh_values.outgoing_flux *= 0
        species.mesh_values.accDensity *= self.mesh.volumes[species.mesh_values.fluxind]
        #Parameter to use
        positions = hit[0][:,:2]
        add_info = hit[0][:,2].astype(int)
        spwts = hit[0][:,3]
        #Scatter flux
        self.scatter_1D(positions, -spwts*species.q, species.mesh_values.outgoing_flux, species.mesh_values.fluxind, add_info)
        self.scatter_1D(positions, -spwts, species.mesh_values.accDensity, species.mesh_values.fluxind, add_info)
        species.mesh_values.outgoing_flux /= self.mesh.overall_area_sat*species.dt
        species.mesh_values.accDensity /= self.mesh.volumes[species.mesh_values.fluxind]

    def gather(self, positions, field, recursive = True, values = None, indices = None):
        if recursive:
            if self.root == True:
                values = []
                val_shape = (numpy.shape(positions)[0], numpy.shape(field)[1])
                positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [])
                indices = self.mesh.sortArrayByMeshes(numpy.arange(numpy.shape(field)[0]))
            pos = positions.pop(0)
            ind = indices.pop(0)
            values.append(super(PIC_recursive, self).gather(pos, field[ind]))

            for child in self.children:
                child.gather(positions, field, recursive = recursive, values = values, indices = indices)
            if self.root == True:
                ordered_values = numpy.zeros(val_shape)
                for ind_i, val_i in zip(pos_ind, values):
                    ordered_values[ind_i] += val_i
                return ordered_values
        else:
            return super(PIC_recursive, self).gather(positions, field)


#PIC_2D_cm1o_recursive(Inherits from PIC_recursive and PIC_2D_cm1o):
#
#Definition = PIC class for cm1o ('c'ylindrical 'm'esh, '1'st 'o'rder implementation), with the tools for recursion.
#Attributes:
#	+PIC class attributes.
#       +PIC_recursive attributes.
#Methods:
#	+Implementation of PIC class methods.
#	+scatterDensity (Species) = return densities of that species in every node of the mesh.
#	+scatterVelocity (Species) = return velocities of that species in every node of the mesh.
#	+scatterFlux = return flux of particles of that species into every indicated node (not all the mesh).
class PIC_2D_cm1o_recursive(PIC_recursive, PIC_2D_cm1o):
    def __init__(self, mesh, n_children, n_root):
        super(PIC_recursive, self).__init__(mesh)
        super().__init__(n_children, n_root, self.type)

    def rEqual0_treatment(self, field, acc = 0):
        field[acc:acc+self.mesh.nx] = field[acc+self.mesh.nx:acc+2*self.mesh.nx]
        acc += self.mesh.nPoints
        for child in self.children:
            child.rEqual0_treatment(field, acc = acc)

    def scatter(self, positions, values, field, indices = None, surface = False):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [], surface = surface)
            values = [values[i] for i in pos_ind]
            indices = self.mesh.sortArrayByMeshes(numpy.arange(len(field)))
        pos = positions.pop(0)
        val = values.pop(0)
        ind = indices.pop(0)
        for child in self.children:
            ind_i = copy.copy(indices[0])
            child.scatter(positions, values, field, indices = indices)
            nonzero = numpy.flatnonzero(field[ind_i])
            pos = numpy.append(pos, child.mesh.getPosition(nonzero), axis = 0)
            val = numpy.append(val, field[ind_i][nonzero])
        temp = field[ind]
        super(PIC_recursive, self).scatter(pos, val, temp)
        field[ind] += temp

    def scatter_1D(self, positions, values, field, location, add_info):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [], surface = True)
            values = [values[i] for i in pos_ind]
            add_info = [add_info[i] for i in pos_ind]
            location = self.mesh.sortIndexByMeshes(location, shift = False)
        pos = positions.pop(0)
        val = values.pop(0)
        info = add_info.pop(0)
        loc = location.pop(0)
        for child in self.children:
            loc_i = copy.copy(location[0])
            child.scatter_1D(positions, values, field, location, add_info)
            array_1D = location_indexes_inv(loc_i, store = False)
            nonzero = numpy.flatnonzero(field[array_1D])
            array_1D = array_1D[nonzero]
            pos = numpy.append(pos, child.mesh.getPosition(child.mesh.location_sat[nonzero]), axis = 0)
            val = numpy.append(val, field[array_1D])
            info = numpy.append(info, child.mesh.direction_sat[nonzero])
        if len(self.children) > 0:
            field[location_indexes_inv(loc, store = False)] *= 0
        self.scatter_1D_aux(pos, val, field, loc, info)

    def scatter_1D_aux(self, positions, values, field, general_location, add_info, prec = 10**(-c.INDEX_PREC)):
        #Creating 1-1 connection between local satellite nodes and general nodes
        dic = {loc_i: gen_i for loc_i, gen_i in zip(self.mesh.location_sat, general_location)}
        #Getting mesh coordinates
        mc = self.mesh.getIndex(positions)
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)

        #Calcualting array indexes with respect to the whole domain
        array_gen = [dic[i] for i in array]
        #Getting the correct indexes for the field
        array_1D = location_indexes_inv(array_gen, store = False)
        horizontal = numpy.equal(add_info % 2, 0)
        vertical = numpy.logical_not(horizontal)
        dindex = numpy.where(horizontal, mc[:,0]-index[:,0], mc[:,1]-index[:,1])

        filter_1D = dindex > prec

        #Adapting horizontal and vertical such that n_index works for inner boundaries as well as outer boundaries
        #mask = numpy.ones_like(add_info, dtype = numpy.bool_)
        #for boundary in self.mesh.boundaries:
        #    if boundary.type == Inner_2D_Rectangular.type:
        #        mask[numpy.flatnonzero(numpy.isin(array, boundary.location))] = False
        #horizontal = numpy.logical_or(horizontal, mask)
        #vertical = numpy.logical_and(vertical, numpy.logical_not(mask))
        #n_index = numpy.where(horizontal, array_1D+1, array_1D+2)
        #n_index = numpy.where(numpy.logical_and(vertical, array_1D == 0), n_index-2+self.mesh.nxsat, n_index)
        #n_index = numpy.where(numpy.logical_and(vertical, array_1D == len(self.mesh.location_sat)-1-self.mesh.nxsat), n_index-2+self.mesh.nxsat, n_index)

        # Finding the other node involved
        temp_location_indexes_inv = {locations: indexes[0] for indexes, locations in numpy.ndenumerate(self.mesh.location_sat)}
        loc_ind = numpy.asarray([temp_location_indexes_inv.get(ind_i) for ind_i in array], dtype = numpy.uint16)
        n_index = [self.mesh.adjacent_sat[loc_ind[i]][add_info[i]] for i in range(len(array_1D))]
        n_index = location_indexes_inv([dic[self.mesh.location_sat[ind]] for ind in n_index], store = False)

        #scatter process
        numpy.add.at(field, array_1D, (1-dindex)*values)
        numpy.add.at(field, n_index[filter_1D], dindex[filter_1D]*values[filter_1D])

    def scatterDiffSq(self, positions, values, weights, array_diff, field, indices = None):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [])
            values = [values[i] for i in pos_ind]
            weights = [weights[i] for i in pos_ind]
            indices = self.mesh.sortArrayByMeshes(numpy.arange(len(field)))
        pos = positions.pop(0)
        val = values.pop(0)
        w = weights.pop(0)
        ind = indices.pop(0)
        pos_c = numpy.zeros((0,numpy.shape(pos)[1]))
        val_c = numpy.zeros((0))
        pseudo_weight_c = numpy.zeros((0))
        for child in self.children:
            ind_i = copy.copy(indices[0])
            child.scatterDiffSq(positions, values, weights, array_diff, field, indices = indices)
            nonzero = numpy.flatnonzero(field[ind_i])
            pos_c = numpy.append(pos_c, child.mesh.getPosition(nonzero), axis = 0)
            val_c = numpy.append(val_c, field[ind_i][nonzero])
            pseudo_weight_c = numpy.append(pseudo_weight_c, numpy.ones_like(nonzero))
        temp = field[ind]
        temp_2 = numpy.zeros_like(field[ind])
        super(PIC_recursive, self).scatter(pos_c, val_c, temp)
        super(PIC_recursive, self).scatter(pos_c, pseudo_weight_c, temp_2)
        temp /= numpy.where(temp_2 != 0, temp_2, 0)
        super(PIC_recursive, self).scatterDiffSq(pos, val, w, array_diff[ind], temp)
        field[ind] += temp

    def scatterDensity(self, species):
        #reset the density
        species.mesh_values.density *= 0
    
        #scatter particles to the mesh
        self.scatter(species.part_values.position[:species.part_values.current_n], species.part_values.spwt[:species.part_values.current_n], species.mesh_values.density)
        
        #divide by cell volume and add the accumulated charge from material surfaces in the domain
        species.mesh_values.density /= self.mesh.volumes
        self.rEqual0_treatment(species.mesh_values.density)

    def scatterSpeed(self, species):
        #reset the velocity
        species.mesh_values.velocity *= 0
    
        #scatter particles to the mesh
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            self.scatter(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n,dim]*species.part_values.spwt[:species.part_values.current_n], species.mesh_values.velocity[:,dim])
            self.rEqual0_treatment(species.mesh_values.velocity[:,dim])
            #Finalizing velocity
            species.mesh_values.velocity[:,dim] *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes)

    def scatterTemperature(self, species):
        #Reset temperature
        species.mesh_values.temperature *= 0
        #Scatter temperature
        for dim in range(numpy.shape(species.part_values.velocity)[1]):
            temp = numpy.zeros_like(species.mesh_values.temperature)
            self.scatterDiffSq(species.part_values.position[:species.part_values.current_n], \
                    species.part_values.velocity[:species.part_values.current_n, dim], species.part_values.spwt[:species.part_values.current_n], species.mesh_values.velocity[:,dim], temp)
            species.mesh_values.temperature += temp
        #Finalizing temperature
        species.mesh_values.temperature *= numpy.where(species.mesh_values.density < 1e-5, 0.0, 1/species.mesh_values.density/self.mesh.volumes*species.m/c.K/3)
        self.rEqual0_treatment(species.mesh_values.temperature)

    def scatterFlux(self, species, hit):
        #Prepare fields
        species.mesh_values.flux *= 0
        species.mesh_values.accDensity *= self.mesh.volumes[species.mesh_values.fluxind]
        #Parameter to use
        positions = hit[0][:,:2]
        add_info = hit[0][:,2].astype(int)
        spwts = hit[0][:,3]
        #Scatter flux
        self.scatter_1D(positions, spwts*species.q, species.mesh_values.flux, species.mesh_values.fluxind, add_info)
        self.scatter_1D(positions, spwts, species.mesh_values.accDensity, species.mesh_values.fluxind, add_info)
        species.mesh_values.flux /= (self.mesh.overall_area_sat)*species.dt
        species.mesh_values.accDensity /= self.mesh.volumes[species.mesh_values.fluxind]

    def scatterOutgoingFlux(self, species, hit):
        #Prepare fields
        species.mesh_values.outgoing_flux *= 0
        species.mesh_values.accDensity *= self.mesh.volumes[species.mesh_values.fluxind]
        #Parameter to use
        positions = hit[0][:,:2]
        add_info = hit[0][:,2].astype(int)
        spwts = hit[0][:,3]
        #Scatter flux
        self.scatter_1D(positions, -spwts*species.q, species.mesh_values.outgoing_flux, species.mesh_values.fluxind, add_info)
        self.scatter_1D(positions, -spwts, species.mesh_values.accDensity, species.mesh_values.fluxind, add_info)
        species.mesh_values.outgoing_flux /= self.mesh.overall_area_sat*species.dt
        species.mesh_values.accDensity /= self.mesh.volumes[species.mesh_values.fluxind]

    def gather(self, positions, field, recursive = True, values = None, indices = None):
        if recursive:
            if self.root == True:
                values = []
                val_shape = (numpy.shape(positions)[0], numpy.shape(field)[1])
                positions, pos_ind = self.mesh.sortPositionsByMeshes(positions, return_ind = [])
                indices = self.mesh.sortArrayByMeshes(numpy.arange(numpy.shape(field)[0]))
            pos = positions.pop(0)
            ind = indices.pop(0)
            values.append(super(PIC_recursive, self).gather(pos, field[ind]))

            for child in self.children:
                child.gather(positions, field, recursive = recursive, values = values, indices = indices)
            if self.root == True:
                ordered_values = numpy.zeros(val_shape)
                for ind_i, val_i in zip(pos_ind, values):
                    ordered_values[ind_i] += val_i
                return ordered_values
        else:
            return super(PIC_recursive, self).gather(positions, field)
