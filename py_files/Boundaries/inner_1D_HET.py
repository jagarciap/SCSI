import copy
import matplotlib.pyplot as plt
import numpy
import pdb

import accelerated_functions as af
import constants as c
from Boundaries.inner_1D_rectangular import Inner_1D_Rectangular
from solver import location_indexes_inv
from Species.species import Species
from timing import Timing


#Inner_1D_HET (Inherits from Inner_1D_Rectangular):
#
#Definition = One-dimensional boundary that represents a HET in a 2D_rectangular mesh 
#Attributes:
#	+type (string) = "Inner - 1D_HET"
#	+xmin (double) = Left limit of the boundary.
#	+xmax (double) = Right limit of the boundary.
#	+ymin (double) = Bottom limit of the boundary.
#	+ymax (double) = Top limit of the boundary.
#       +rmin (double) = Inner radius of the thruster exit
#       +rmax (double) = Outer radius of the thruster exit
#       +exit_area (double) = Area at the thruster exit
#       +exi_nodes ([ind]) = Nodes where the information of Species is stored.
#       +exit_pot (double) = Electric potential at the exit nodes
#       +exit_pot_nodes ([ind]) = Nodes where the potential of the thruster exit is applied
#       +Boundary attributes
#Methods:
#	+Boundary methods.
class Inner_1D_HET(Inner_1D_Rectangular):
    type = "Inner - 1D_HET"
    def __init__(self, x_min, x_max , y_min, y_max, n_material):
        super().__init__(x_min, x_max , y_min, y_max, n_material)
        self.rmin = c.R_MIN
        self.rmax = c.R_MAX
        self.exit_area = c.EXIT_AREA
        self.exit_nodes = None
        self.exit_pot = c.EXIT_POT
        self.exit_pot_nodes = None

#       NOTE: This way of treating the boundary does not take into account the particles that cross the plane defined by the boundary that do not properly cross the boundary.
#           Example: if the boundary is a right boundary, the cases where pos_x > xmax, but also ymax > pos_y or ymin < pos_y.
    def checkPositionInBoundary(self, pos, surface = False, prec = 1e-3):
        return numpy.ones_like(pos[:,0], dtype = numpy.bool_)

    #	+applyElectricBoundary(Electric_Field) = Applies the boundary condition to the electric field passed as argument. 
    #       In this case, some nodes are considered Thruster exit and have a particular potential, the others lie in the
    #       metallic section of the thruster and are kept at potential 0. 
    def applyElectricBoundary(self, e_field):
        values = numpy.where(numpy.isin(self.location, self.exit_pot_nodes), self.exit_pot, 0.0)
        e_field.dirichlet(e_field.potential[self.location]+values, self, e_field.pic.mesh.nx, e_field.pic.mesh.ny, e_field.pic.mesh.dx, e_field.pic.mesh.dy)

#       +createDistributionAtBorder(Motion_Solver part_solver, Species species, [double] delta_n): (([double,double] pos, [int] border), [int] repeats) =
#           The function creates particle positions of 'species' along the region between rmin and rmax, under a uniform distribution with a surface density 'delta_n', where
#           delta_n indicates the density per 'location' node [particle/m^2].
#           Return: 'pos' is the numpy array indicating the positions of the new particles, 'border' indicates in which border they are created, and
#               'repeats' indicates for each position, how many particles are expected to be created.
#               The tuple (pos, border) is reffered as flux in the program.
    @Timing
    def createDistributionHET(self, part_solver, species, delta_n, drift_vel, ind_offset, prec = 1e-5):
        add_rand = numpy.random.rand(len(self.exit_nodes))
        mpf_new = delta_n*(self.exit_area/2*drift_vel[:,1]*species.dt)
        mpf_new = mpf_new/species.spwt+species.mesh_values.residuals[ind_offset+self.exit_nodes]+add_rand
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals[ind_offset+self.exit_nodes] = mpf_new-mp_new

        #Bottom
        if self.directions[0] == 0:
            center = (self.xmax+self.xmin)/2
            pos_1 = numpy.asarray([[center-self.rmax, self.ymin],[center+self.rmin, self.ymin]])
            pos_1 = numpy.repeat(pos_1, mp_new, axis = 0)
            random = numpy.random.rand(numpy.shape(pos_1)[0])
            shifts = random*(self.rmax-self.rmin)
            pos_1[:,0] += shifts
            hit_1 = numpy.zeros_like(pos_1[:,1], dtype = numpy.uint8)[:,None]
        #Left
        elif self.directions[0] == 3:
            center = (self.ymax+self.ymin)/2
            pos_1 = numpy.asarray([[self.xmin, center-self.rmax],[self.xmin, center+self.rmin]])
            pos_1 = numpy.repeat(pos_1, mp_new, axis = 0)
            random = numpy.random.rand(numpy.shape(pos_1)[0])
            shifts = random*(self.rmax-self.rmin)
            pos_1[:,1] += shifts
            hit_1 = 3*numpy.ones_like(pos_1[:,1], dtype = numpy.uint8)[:,None]
        #Right
        elif self.directions[0] == 1:
            center = (self.ymax+self.ymin)/2
            pos_1 = numpy.asarray([[self.xmax, center-self.rmax], [self.xmax, center+self.rmin]])
            pos_1 = numpy.repeat(pos_1, mp_new, axis = 0)
            random = numpy.random.rand(numpy.shape(pos_1)[0])
            shifts = random*(self.rmax-self.rmin)
            pos_1[:,1] += shifts
            hit_1 = numpy.ones_like(pos_1[:,1], dtype = numpy.uint8)[:,None]
        #Top
        else:
            center = (self.xmax+self.xmin)/2
            pos_1 = numpy.asarray([[center-self.rmax, self.ymax],[center+self.rmin, self.ymax]])
            pos_1 = numpy.repeat(pos_1, mp_new, axis = 0)
            random = numpy.random.rand(numpy.shape(pos_1)[0])
            shifts = random*(self.rmax-self.rmin)
            pos_1[:,0] += shifts
            hit_1 = 2*numpy.ones_like(pos_1[:,1], dtype = numpy.uint8)[:,None]

        repeats = numpy.ones(numpy.shape(hit_1)[0], dtype = numpy.uint8)
        return (numpy.append(numpy.append(pos_1, hit_1, axis = 1), species.spwt*numpy.ones_like(hit_1), axis = 1),), repeats
