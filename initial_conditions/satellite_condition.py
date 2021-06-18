import numpy
import sys

sys.path.insert(0,'..')

import constants as c


def load_protons_SW(mesh, species):
    ind = numpy.arange(mesh.nPoints)
    ind_2D = mesh.arrayToIndex(ind)
    extremes_array = numpy.asarray([mesh.boundaries[1].location[0], mesh.boundaries[1].location[-1]])
    extremes_2D = mesh.arrayToIndex(extremes_array)
    ind = ind[numpy.flatnonzero(numpy.logical_not(numpy.logical_and(extremes_2D[0,0]<=ind_2D[:,0],\
                                                  numpy.logical_and(extremes_2D[1,0]>=ind_2D[:,0],\
                                                  numpy.logical_and(extremes_2D[0,1]<=ind_2D[:,1],\
                                                                    extremes_2D[1,1]>=ind_2D[:,1])))))]
    species.mesh_values.density[ind] = c.P_N
    species.mesh_values.velocity[ind,0] = c.P_V_SW
    species.mesh_values.velocity[ind,1] = 0
    species.mesh_values.temperature[ind] = c.P_T

def load_electrons_SW(mesh, species):
    ind = numpy.arange(mesh.nPoints)
    ind_2D = mesh.arrayToIndex(ind)
    extremes_array = numpy.asarray([mesh.boundaries[1].location[0], mesh.boundaries[1].location[-1]])
    extremes_2D = mesh.arrayToIndex(extremes_array)
    ind = ind[numpy.flatnonzero(numpy.logical_not(numpy.logical_and(extremes_2D[0,0]<=ind_2D[:,0],\
                                                  numpy.logical_and(extremes_2D[1,0]>=ind_2D[:,0],\
                                                  numpy.logical_and(extremes_2D[0,1]<=ind_2D[:,1],\
                                                                    extremes_2D[1,1]>=ind_2D[:,1])))))]
    species.mesh_values.density[ind] = c.E_N
    species.mesh_values.velocity[ind,0] = c.E_V_SW
    species.mesh_values.velocity[ind,1] = 0
    species.mesh_values.temperature[ind] = c.E_T
