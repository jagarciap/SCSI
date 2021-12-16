import numpy
import sys

sys.path.insert(0,'..')

import constants as c


def load_protons_SW(mesh, species):
    ind = numpy.arange(mesh.nPoints)
    ind_pos = mesh.getPosition(ind)
    mask1 = numpy.logical_not(numpy.logical_and(mesh.boundaries[1].xmin<=ind_pos[:,0],\
                             numpy.logical_and(mesh.boundaries[1].xmax>=ind_pos[:,0],\
                             numpy.logical_and(mesh.boundaries[1].ymin<=ind_pos[:,1],\
                                               mesh.boundaries[1].ymax>=ind_pos[:,1]))))
    #Approximation of a proton wake
    #NOTE: Unique to 2021_10_06 domain and the like
    x1 = 9.2
    mask2 = numpy.logical_not(numpy.logical_and(numpy.logical_and(ind_pos[:,0] >= mesh.boundaries[1].xmax, ind_pos[:,0] <= x1), ind_pos[:,1] <= \
            (mesh.boundaries[1].ymin-mesh.boundaries[1].ymax)/(x1-mesh.boundaries[1].xmax)*(ind_pos[:,0]-mesh.boundaries[1].xmax)+mesh.boundaries[1].ymax))
    ind = ind[numpy.logical_and(mask1,mask2)]

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
