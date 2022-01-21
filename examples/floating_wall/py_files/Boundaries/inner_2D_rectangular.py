import copy
import numpy
import matplotlib.pyplot as plt
import pdb

import accelerated_functions as af
import constants as c
from Boundaries.boundary import Boundary
from Species.species import Species
from solver import location_indexes_inv
from timing import Timing

#Inner_2D_Rectangular (Inherits from Boundary):
#
#Definition = Inner rectangular boundary for a rectangular mesh
#Attributes:
#	+type (string) = "Inner - 2D_Rectangular"
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#       +bottom ([int]) = array of indices that indicates the represents the bottom of the boundary.
#       +top ([int]) = array of indices that indicates the represents the top of the boundary.
#       +left ([int]) = array of indices that indicates the represents the left of the boundary.
#       +right ([int]) = array of indices that indicates the represents the right of the boundary.
#       +ind_inner([int]) = array of indices that lie inside the object sorrounded by this boundary.
#       +Boundary attributes
#Methods:
#	+Boundary methods.
    #	+applyParticleBoundary(Species) = Applies the boundary condition to the species passed as argument.
    #           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
    #           **kwargs may contain arguments necessary for inner methods.
#       +applyParticleOpenBoundary(Species) = Deletes particles of Species  outside of the boundaries.
#       +applyParticleReflectiveBoundary(Species species, Species old_species) = Reflects the particles back into the domain.
#           old_species refers to the state of species in the previous step.
#       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = for every location,
#           create the dummy boxes outside of the domain with particles in them, using delta_n (density), n_vel (thermal velocity), shift_vel (velocity shared by all particles).
#       +injectParticlesDummyBox([int] location, PIC pic, Field field, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) =
#               Inject the particles in location indices by creating dummy boxes around them, creating particles
#       	inside of them, moving the particles, and then adding the ones that entered into the computational domain.
#       +createDistributionAtBorder([int] location, Motion_Solver part_solver, Species species, [double] delta_n): (([double,double] pos, [int] border), [int] repeats) =
#           The function creates particle positions of 'species' along the region denoted by 'location', under a uniform distribution with a density 'delta_n', where
#           delta_n indicates the density per 'location' node.
#           Return: 'pos' is the numpy array indicating the positions of the new particles, 'border' indicates in which border they are created, and
#               'repeats' indicates for each position, how many particles are expected to be created.
#               The tuple (pos, border) is reffered as flux in the program.
#       +injectParticlesAtPositions('flux', Motion_Solver part_solver, Field field, Species species, [double] delta_n, [double] n_vel, double delta_pos) =
#           The method creates 'delta_n' particles at each entry of 'pos' stored in the parameter 'flux' (See Documentation of 'createDistributionArBorder').
#           The new particles are stored in 'species', shifted 'delta_pos' away from their borders, initiated with 'n_vel' velocities and prepared in time
#           according to the method used 'part_solver' for advancing particles.
class Inner_2D_Rectangular(Boundary):
    type = "Inner - 2D_Rectangular"
    def __init__(self, x_min, x_max , y_min, y_max, n_material):
        self.material = n_material

        self.xmin = x_min
        self.xmax = x_max
        self.ymin = y_min
        self.ymax = y_max

        self.bottom = []
        self.top = []
        self.left = []
        self.right = []
        self.ind_inner = []
        self.location = []

        self.directions = []
        self.areas = []
        self.adjacent = []

    ##	+applyElectricBoundary(Electric_Field) = Applies the boundary condition to the electric field passed as argument. So far a 0V Dirichlet boundary condition is applied.
    ##NOTE: Modifified to add a constant -20V at the right boundary (2020_03_12)
    #    def applyElectricBoundary(self, e_field):
    #        #Location of dirichlet and neumann boundaries
    #        dirichlet_loc_1 = numpy.arange(c.NX-1, c.NX*c.NY, c.NX)
    #        neumann_loc = numpy.delete(self.location, numpy.arange(c.NX-1,c.NX+(c.NY-1)*2, 2))
    #        dirichlet_loc_2 = numpy.arange(0, c.NX*c.NY, c.NX)
    #        #Values
    #        dirichlet_val_1 = -20*numpy.ones_like(dirichlet_loc_1)
    #        dirichlet_val_2 = numpy.zeros_like(dirichlet_loc_2)
    #        neumann_val = numpy.zeros_like(neumann_loc)
    #        #Applying values
    #        e_field.dirichlet(dirichlet_loc_1, dirichlet_val_1)
    #        e_field.dirichlet(dirichlet_loc_2, dirichlet_val_2)
    #        e_field.neumann(neumann_loc, neumann_val)

    def checkPositionInBoundary(self, pos, surface = False):
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        #Inner boundary
        if surface:
            mask2 = af.geq_2D_p(pos, xmin, xmax, ymin, ymax)
        else:
            mask2 = af.g_2D_p(pos, xmin, xmax, ymin, ymax)

        return mask2

    #	+applyElectricBoundary(Electric_Field) = Applies the boundary condition to the electric field passed as argument. So far a 0V Dirichlet boundary condition is applied.
    def applyElectricBoundary(self, e_field):
        values = e_field.potential[self.location]
        e_field.dirichlet(values, self, e_field.pic.mesh.nx, e_field.pic.mesh.ny, e_field.pic.mesh.dx, e_field.pic.mesh.dy)

    #	+applyMagneticBoundary(Magnetic_Field) = Applies the boundary condition to the magnetic field passed as argument.
    #       No magnetic field so far
    def applyMagneticBoundary(self, m_field):
        pass

    #	+applyParticleBoundary(Species) = Applies the boundary condition to the species passed as argument.
    #           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
    #           **kwargs may contain arguments necessary for inner methods.
    def applyParticleBoundary(self, species, type_boundary, albedo = None, **kwargs):
        np = species.part_values.current_n
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        # Finding the particles out of domain
        out_ind = af.l_2D_p(species.part_values.position[:np,:], xmin, xmax, ymin, ymax, prec = 0)
        out_ind = numpy.flatnonzero(out_ind)

        if type_boundary == 'mixed':
            rand = numpy.random.rand(len(out_ind))
            mask_albedo = rand < albedo
            self.applyParticleReflectiveBoundary(species, out_ind[mask_albedo], old_position = kwargs['old_position'])
            return self.applyParticleOpenBoundary(species, out_ind[numpy.logical_not(mask_albedo)], old_position = kwargs['old_position'])
        elif type_boundary == 'open':
            return self.applyParticleOpenBoundary(species, out_ind, old_position = kwargs['old_position'])
        elif type_boundary == 'reflective':
            return self.applyParticleReflectiveBoundary(species, out_ind, old_position = kwargs['old_position'])
        else:
            raise ValueError("Called invalid boundary method")

    #       +applyParticleOpenBoundary(Species) = Deletes particles at or outside of the boundaries. In this case the particles that are to be eliminated are sent in 'ind'.
    def applyParticleOpenBoundary(self, species, ind, old_position = None, prec = 0):
        #Just for convenience in writing
        np = species.part_values.current_n
        coord = None
        vel = None
        tan_vel = None
        cos = None
        if old_position is not None:
            #Bottom
            botind = numpy.nonzero((old_position[ind,1]-self.ymin) <= prec)[0]
            slope = (species.part_values.position[ind[botind],0]-old_position[ind[botind],0])/\
                    (species.part_values.position[ind[botind],1]-old_position[ind[botind],1])
            hit = slope*(self.ymin-old_position[ind[botind],1])+old_position[ind[botind],0]
            hit_ind = numpy.nonzero(numpy.logical_and(self.xmin < hit, hit < self.xmax))[0]
            coord = numpy.append(hit[hit_ind][:,None], numpy.append(self.ymin*numpy.ones_like((hit_ind))[:,None],\
                                                          numpy.zeros_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
            coord = numpy.append(coord, species.part_values.spwt[ind[botind[hit_ind]]][:,None], axis = 1)
            vel = copy.copy(species.part_values.velocity[ind[botind[hit_ind]],1])
            tan_vel = copy.copy(species.part_values.velocity[ind[botind[hit_ind]],0])
            cos = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Left
            leftind = numpy.nonzero((old_position[ind,0]-self.xmin) <= prec)[0]
            slope = (species.part_values.position[ind[leftind],1]-old_position[ind[leftind],1])/\
                    (species.part_values.position[ind[leftind],0]-old_position[ind[leftind],0])
                  
            hit = slope*(self.xmin-old_position[ind[leftind],0])+old_position[ind[leftind],1]
            hit_ind = numpy.nonzero(numpy.logical_and(self.ymin < hit, hit < self.ymax))[0]
            coord_l = numpy.append(self.xmin*numpy.ones_like((hit_ind))[:,None], numpy.append(hit[hit_ind][:,None],\
                                    3*numpy.ones_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
            coord_l = numpy.append(coord_l, species.part_values.spwt[ind[leftind[hit_ind]]][:,None], axis = 1)
            vel_l = species.part_values.velocity[ind[leftind[hit_ind]], 0]
            tan_vel_l = species.part_values.velocity[ind[leftind[hit_ind]], 1]
            cos_l = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Right
            rightind = numpy.nonzero((old_position[ind,0]-self.xmax) >= -prec)[0]
            slope = (species.part_values.position[ind[rightind],1]-old_position[ind[rightind],1])/\
                    (species.part_values.position[ind[rightind],0]-old_position[ind[rightind],0])
            hit = slope*(self.xmax-old_position[ind[rightind],0])+old_position[ind[rightind],1]
            hit_ind = numpy.nonzero(numpy.logical_and(self.ymin < hit, hit < self.ymax))[0]
            coord_r = numpy.append(self.xmax*numpy.ones_like((hit_ind))[:,None], numpy.append(hit[hit_ind][:,None],\
                                    numpy.ones_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
            coord_r = numpy.append(coord_r, species.part_values.spwt[ind[rightind[hit_ind]]][:,None], axis = 1)
            vel_r = -species.part_values.velocity[ind[rightind[hit_ind]], 0]
            tan_vel_r = species.part_values.velocity[ind[rightind[hit_ind]], 1]
            cos_r = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Top
            topind = numpy.nonzero((old_position[ind,1]-self.ymax) >= -prec)[0]
            slope = (species.part_values.position[ind[topind],0]-old_position[ind[topind],0])/\
                    (species.part_values.position[ind[topind],1]-old_position[ind[topind],1])
            hit = slope*(self.ymax-old_position[ind[topind],1])+old_position[ind[topind],0]
            hit_ind = numpy.nonzero(numpy.logical_and(self.xmin < hit, hit < self.xmax))[0]
            coord_t = numpy.append(hit[hit_ind][:,None], numpy.append(self.ymax*numpy.ones_like((hit_ind))[:,None],\
                                                            2*numpy.ones_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
            coord_t = numpy.append(coord_t, species.part_values.spwt[ind[topind[hit_ind]]][:,None], axis = 1)
            vel_t = -species.part_values.velocity[ind[topind[hit_ind]], 1]
            tan_vel_t = species.part_values.velocity[ind[topind[hit_ind]], 0]
            cos_t = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Preparing the arrays that will be returned
            coord = numpy.concatenate((coord, coord_l, coord_r, coord_t), axis = 0)
            vel = numpy.concatenate((vel, vel_l, vel_r, vel_t), axis = 0)
            tan_vel = numpy.concatenate((tan_vel, tan_vel_l, tan_vel_r, tan_vel_t), axis = 0)
            cos = numpy.concatenate((cos, cos_l, cos_r, cos_t), axis = 0)
            #Evaluating that everything goes as expected
            assert len(ind) == numpy.shape(coord)[0], "There should not be particles inside the boundaries prev. to this state, or duplicated particles"
        # Eliminating particles
        self.removeParticles(species,ind)
        count2 = numpy.shape(ind)[0]
        print('Number of {} eliminated - inner:'.format(species.name), count2)
        #Positions of deleted particles for posterior processing of flux
        return {'flux': (coord, vel, tan_vel, cos), 'del_ind': ind}


    #       +applyParticleOpenBoundaryInverse(Species) = This function, as 'applyParticleOpenBoundary', identifies where  and under which parameters the particles cross the border,
    #           but does not eliminate them. The method is used for calculating the Outgoing flux.
    def applyParticleOpenBoundaryInverse(self, species, ind, old_position = None):
        #Just for convenience in writing
        np = species.part_values.current_n
        coord = None
        vel = None
        tan_vel = None
        cos = None
        if old_position is not None:
            #Bottom
            botind = numpy.nonzero(species.part_values.position[ind,1] < self.ymin)[0]
            slope = (species.part_values.position[ind[botind],0]-old_position[ind[botind],0])/\
                    (species.part_values.position[ind[botind],1]-old_position[ind[botind],1])
            hit = slope*(self.ymin-old_position[ind[botind],1])+old_position[ind[botind],0]
            hit_ind = numpy.nonzero(numpy.logical_and(self.xmin < hit, hit < self.xmax))[0]
            coord = numpy.append(hit[hit_ind][:,None], numpy.append(self.ymin*numpy.ones_like((hit_ind))[:,None],\
                                                          numpy.zeros_like((hit_ind)[:,None], dtype = numpy.short), axis = 1), axis = 1)
            coord = numpy.append(coord, species.part_values.spwt[ind[botind[hit_ind]]][:,None], axis = 1)
            vel = -copy.copy(species.part_values.velocity[ind[botind[hit_ind]],1])
            #tan_vel = copy.copy(species.part_values.velocity[ind[botind[hit_ind]],0])
            #cos = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Left
            leftind = numpy.nonzero(species.part_values.position[ind,0] < self.xmin)[0]
            slope = (species.part_values.position[ind[leftind],1]-old_position[ind[leftind],1])/\
                    (species.part_values.position[ind[leftind],0]-old_position[ind[leftind],0])
            hit = slope*(self.xmin-old_position[ind[leftind],0])+old_position[ind[leftind],1]
            hit_ind = numpy.nonzero(numpy.logical_and(self.ymin < hit, hit < self.ymax))[0]
            coord_l = numpy.append(self.xmin*numpy.ones_like((hit_ind))[:,None], numpy.append(hit[hit_ind][:,None],\
                                    3*numpy.ones_like((hit_ind)[:,None], dtype = numpy.short), axis = 1), axis = 1)
            coord_l = numpy.append(coord_l, species.part_values.spwt[ind[leftind[hit_ind]]][:,None], axis = 1)
            vel_l = -species.part_values.velocity[ind[leftind[hit_ind]], 0]
            #tan_vel_l = species.part_values.velocity[ind[leftind[hit_ind]], 1]
            #cos_l = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Right
            rightind = numpy.nonzero(species.part_values.position[ind,0] > self.xmax)[0]
            slope = (species.part_values.position[ind[rightind],1]-old_position[ind[rightind],1])/\
                    (species.part_values.position[ind[rightind],0]-old_position[ind[rightind],0])
            hit = slope*(self.xmax-old_position[ind[rightind],0])+old_position[ind[rightind],1]
            hit_ind = numpy.nonzero(numpy.logical_and(self.ymin < hit, hit < self.ymax))[0]
            coord_r = numpy.append(self.xmax*numpy.ones_like((hit_ind))[:,None], numpy.append(hit[hit_ind][:,None],\
                                    numpy.ones_like((hit_ind)[:,None], dtype = numpy.short), axis = 1), axis = 1)
            coord_r = numpy.append(coord_r, species.part_values.spwt[ind[rightind[hit_ind]]][:,None], axis = 1)
            vel_r = species.part_values.velocity[ind[rightind[hit_ind]], 0]
            #tan_vel_r = species.part_values.velocity[ind[rightind[hit_ind]], 1]
            #cos_r = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Top
            topind = numpy.nonzero(species.part_values.position[ind,1] > self.ymax)[0]
            slope = (species.part_values.position[ind[topind],0]-old_position[ind[topind],0])/\
                    (species.part_values.position[ind[topind],1]-old_position[ind[topind],1])
            hit = slope*(self.ymax-old_position[ind[topind],1])+old_position[ind[topind],0]
            hit_ind = numpy.nonzero(numpy.logical_and(self.xmin < hit, hit < self.xmax))[0]
            coord_t = numpy.append(hit[hit_ind][:,None], numpy.append(self.ymax*numpy.ones_like((hit_ind))[:,None],\
                                                            2*numpy.ones_like((hit_ind)[:,None], dtype = numpy.short), axis = 1), axis = 1)
            coord_t = numpy.append(coord_t, species.part_values.spwt[ind[topind[hit_ind]]][:,None], axis = 1)
            vel_t = species.part_values.velocity[ind[topind[hit_ind]], 1]
            #tan_vel_t = species.part_values.velocity[ind[topind[hit_ind]], 0]
            #cos_t = 1/numpy.sqrt(slope[hit_ind]*slope[hit_ind]+1)
            #Preparing the arrays that will be returned
            coord = numpy.concatenate((coord, coord_l, coord_r, coord_t), axis = 0)
            vel = numpy.concatenate((vel, vel_l, vel_r, vel_t), axis = 0)
            #tan_vel = numpy.concatenate((tan_vel, tan_vel_l, tan_vel_r, tan_vel_t), axis = 0)
            #cos = numpy.concatenate((cos, cos_l, cos_r, cos_t), axis = 0)
            #Evaluating that everything goes as expected
            assert len(ind) == numpy.shape(coord)[0], "There should not be particles inside the boundaries prev. to this state, or duplicated particles"
        #Positions of deleted particles for posterior processing of flux
        return {'flux': (coord, vel, tan_vel, cos)}


    #       +applyParticleReflectiveBoundary(Species species, Species old_species) = Reflects the particles back into the domain.
    #           old_species refers to the state of species in the previous step. ind are the particles that need to be treated.
    def applyParticleReflectiveBoundary(self, species, ind, old_position = None):
        delta = 1e-5
        if old_position is not None:
            #Bottom
            botind = numpy.nonzero(old_position[ind,1] < self.ymin)[0]
            hit = (species.part_values.position[ind[botind],0]-old_position[ind[botind],0])/\
                  (species.part_values.position[ind[botind],1]-old_position[ind[botind],1])*\
                  (self.ymin-old_position[ind[botind],1])+old_position[ind[botind],0]
            hit_ind = numpy.nonzero(numpy.logical_and(self.xmin < hit, hit < self.xmax))[0]
            species.part_values.position[ind[botind[hit_ind]], 1] = 2*self.ymin - species.part_values.position[ind[botind[hit_ind]],1]-delta
            species.part_values.velocity[ind[botind[hit_ind]], 1] *= -1.0
            #Left
            leftind = numpy.nonzero(old_position[ind,0] < self.xmin)[0]
            hit = (species.part_values.position[ind[leftind],1]-old_position[ind[leftind],1])/ \
                  (species.part_values.position[ind[leftind],0]-old_position[ind[leftind],0])* \
                  (self.xmin-old_position[ind[leftind],0])+old_position[ind[leftind],1]
            hit_ind = numpy.nonzero(numpy.logical_and(self.ymin < hit, hit < self.ymax))[0]
            species.part_values.position[ind[leftind[hit_ind]], 0] = 2*self.xmin - species.part_values.position[ind[leftind[hit_ind]],0]-delta
            species.part_values.velocity[ind[leftind[hit_ind]], 0] *= -1.0
            #Right
            rightind = numpy.nonzero(old_position[ind,0] > self.xmax)[0]
            hit = (species.part_values.position[ind[rightind],1]-old_position[ind[rightind],1])/ \
                  (species.part_values.position[ind[rightind],0]-old_position[ind[rightind],0])* \
                  (self.xmax-old_position[ind[rightind],0])+old_position[ind[rightind],1]
            hit_ind = numpy.nonzero(numpy.logical_and(self.ymin < hit, hit < self.ymax))[0]
            species.part_values.position[ind[rightind[hit_ind]], 0] = 2*self.xmax - species.part_values.position[ind[rightind[hit_ind]],0]+delta
            species.part_values.velocity[ind[rightind[hit_ind]], 0] *= -1.0
            #Top
            topind = numpy.nonzero(old_position[ind,1] > self.ymax)[0]
            hit = (species.part_values.position[ind[topind],0]-old_position[ind[topind],0])/ \
                  (species.part_values.position[ind[topind],1]-old_position[ind[topind],1])* \
                  (self.ymax-old_position[ind[topind],1])+old_position[ind[topind],0]
            hit_ind = numpy.nonzero(numpy.logical_and(self.xmin < hit, hit < self.xmax))[0]
            species.part_values.position[ind[topind[hit_ind]], 1] = 2*self.ymax - species.part_values.position[ind[topind[hit_ind]],1]+delta
            species.part_values.velocity[ind[topind[hit_ind]], 1] *= -1.0


 #       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = create the dummy boxes with particles in them.
    def createDummyBox(self, location, pic, species, delta_n, n_vel, shift_vel, prec = 1e-5):
        #Preparing things for numpy functions use
        loc, u_ind = numpy.unique(location, return_index = True)
        add_rand = numpy.random.rand(*numpy.shape(loc))
        dv = numpy.max(pic.mesh.volumes)
        mpf_new = delta_n[u_ind]*(dv-pic.mesh.volumes[loc])/species.spwt+\
                  species.mesh_values.residuals[loc]+add_rand
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals[loc] = mpf_new-mp_new
        ind = numpy.arange(len(loc))
        index = numpy.repeat(ind, mp_new)
        #Setting up positions
        pos = pic.mesh.getPosition(loc)[index]
        random = numpy.random.rand(*numpy.shape(pos))
        random += numpy.where(random == 0, 1e-3, 0)
        shift = numpy.where(numpy.abs(pos[:,0]-self.xmin) < prec, random[:,0]/2*pic.mesh.dx, (random[:,0]-0.5)*pic.mesh.dx)
        pos[:,0] = pos[:,0] + shift - numpy.where(numpy.abs(pos[:,0]-self.xmax) < prec, random[:,0]/2*pic.mesh.dx, 0)
        shift = numpy.where(numpy.abs(pos[:,1]-self.ymin) < prec, random[:,1]/2*pic.mesh.dy, (random[:,1]-0.5)*pic.mesh.dy)
        pos[:,1] = pos[:,1] + shift - numpy.where(numpy.abs(pos[:,1]-self.ymax) < prec, random[:,1]/2*pic.mesh.dy, 0)
        #Setting up velocities
        vel = super().sampleIsotropicVelocity(n_vel[u_ind], mp_new)+shift_vel[index]
        #Adding particles
        super().addParticles(species, pos, vel)


#       +injectParticlesDummyBox([int] location, PIC pic, Field field, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) =
#               Inject the particles in location indices by creating dummy boxes around them, creating particles
#       	inside of them, moving the particles, and then adding the ones that entered into the computational domain.
    @Timing
    def injectParticlesDummyBox(self, location, part_solver, field, species, delta_n, n_vel, shift_vel):
        # Creating temporary species
        ghost = Species("temporary species", species.dt, species.q, species.m, species.debye, species.spwt, \
                        int(species.part_values.max_n/10), species.pos_dim, species.vel_dim, species.mesh_values.nPoints, numpy.asarray([0]))
        ghost.mesh_values.residuals = species.mesh_values.residuals
        self.createDummyBox(location, part_solver.pic, ghost, delta_n, n_vel, shift_vel)
        species.mesh_values.residuals[location] = copy.copy(ghost.mesh_values.residuals[location])
        #Preparing variables
        np = ghost.part_values.current_n
        #Entering particles into the mesh and adjusting them according to motion_solver
        old_position = copy.copy(ghost.part_values.position)
        ghost.part_values.position[:np,:] += ghost.part_values.velocity[:np,:]*ghost.dt
        ind = numpy.flatnonzero(self.checkPositionInBoundary(ghost.part_values.position[:np,:]))

        hit = self.applyParticleOpenBoundaryInverse(ghost, ind, old_position = old_position)['flux']
        ###Test
        #np = ghost.part_values.current_n
        ##Test positioning
        #fig = plt.figure(figsize=(8,8))
        #plt.scatter(ghost.part_values.position[:np, 0], ghost.part_values.position[:np,1], marker = '.')
        #plt.title(self.type+" - "+species.name)
        #plt.show()
        ##Test velocity
        #fig = plt.figure(figsize=(8,8))
        #datamag = plt.hist(numpy.sqrt(ghost.part_values.velocity[:np,0]*ghost.part_values.velocity[:np,0]+ \
        #                              ghost.part_values.velocity[:np,1]*ghost.part_values.velocity[:np,1]), 81, alpha=0.5, label=species.name)
        #plt.title(self.type+" - "+species.name)
        #plt.show()

        #Leap-frog state
        part_solver.initialConfiguration(ghost, field, ind)

        #Adding particles
        self.addParticles(species, ghost.part_values.position[ind,:], ghost.part_values.velocity[ind,:])
        self.updateTrackers(species, len(ind))

        #Calculating outgoing flux
        part_solver.pic.scatterOutgoingFlux(species, hit)

        print("Injected particles: ", len(ind))
        print("Total {}".format(species.name),": ", species.part_values.current_n)


#       +createDistributionAtBorder([int] location, Motion_Solver part_solver, Species species, [double] delta_n): (([double,double] pos, [int] border), [int] repeats) =
#           The function creates particle positions of 'species' along the region denoted by 'location', under a uniform distribution with a surface density 'delta_n', where
#           delta_n indicates the density per 'location' node [particle/m^2].
#           Return: 'pos' is the numpy array indicating the positions of the new particles, 'border' indicates in which border they are created, and
#               'repeats' indicates for each position, how many particles are expected to be created.
#               The tuple (pos, border) is reffered as flux in the program.
    @Timing
    def createDistributionAtBorder(self, location, part_solver, species, delta_n, prec = 1e-5):
        add_rand = numpy.random.rand(len(location))
        #This needs to be generalized later
        #NOTE: Modified(2021/02/14) with no backward compatibility
        local_loc = location_indexes_inv(location, store = False)
        mpf_new = delta_n*self.areas[local_loc]

        #Treating borders
        mpf_new /= numpy.where(numpy.max(part_solver.pic.mesh.volumes)/part_solver.pic.mesh.volumes[location] < 1.5, 2, 1)

        #Computing number of particles created
        mpf_new = mpf_new/species.spwt+species.mesh_values.residuals[location]+add_rand
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals[location] = mpf_new-mp_new

        #Assigning positions
        pos_1 = numpy.repeat(part_solver.pic.mesh.getPosition(location), mp_new, axis = 0)
        random = numpy.random.rand(numpy.shape(pos_1)[0])
        random += numpy.where(random == 0, 1e-3, 0)
        hit_1 = numpy.repeat(self.directions[local_loc], mp_new)
        ind_b = numpy.flatnonzero(hit_1 == 0)
        ind_l = numpy.flatnonzero(hit_1 == 3)
        ind_r = numpy.flatnonzero(hit_1 == 1)
        ind_t = numpy.flatnonzero(hit_1 == 2)

        #Bottom
        shifts = numpy.where(numpy.abs(pos_1[ind_b,0]-self.xmin) < prec, random[ind_b]*part_solver.pic.mesh.dx/2, (random[ind_b]-0.5)*part_solver.pic.mesh.dx)
        shifts -= numpy.where(numpy.abs(pos_1[ind_b,1]-self.xmax) < prec, random[ind_b]*part_solver.pic.mesh.dx/2, 0)
        pos_1[ind_b,0] += shifts
        #Left
        shifts = numpy.where(numpy.abs(pos_1[ind_l,1]-self.ymin) < prec, random[ind_l]*part_solver.pic.mesh.dy/2, (random[ind_l]-0.5)*part_solver.pic.mesh.dy)
        shifts -= numpy.where(numpy.abs(pos_1[ind_l,1]-self.ymax) < prec, random[ind_l]*part_solver.pic.mesh.dy/2, 0)
        pos_1[ind_l,1] += shifts
        #Right
        shifts = numpy.where(numpy.abs(pos_1[ind_r,1]-self.ymin) < prec, random[ind_r]*part_solver.pic.mesh.dy/2, (random[ind_r]-0.5)*part_solver.pic.mesh.dy)
        shifts -= numpy.where(numpy.abs(pos_1[ind_r,1]-self.ymax) < prec, random[ind_r]*part_solver.pic.mesh.dy/2, 0)
        pos_1[ind_r,1] += shifts
        #Top
        shifts = numpy.where(numpy.abs(pos_1[ind_t,0]-self.xmin) < prec, random[ind_t]*part_solver.pic.mesh.dx/2, (random[ind_t]-0.5)*part_solver.pic.mesh.dx)
        shifts -= numpy.where(numpy.abs(pos_1[ind_t,1]-self.xmax) < prec, random[ind_t]*part_solver.pic.mesh.dx/2, 0)
        pos_1[ind_t,0] += shifts

        repeats = numpy.ones(numpy.shape(hit_1)[0], dtype = numpy.uint8)
        return (numpy.append(pos_1, hit_1[:,None], axis = 1),), repeats


#       +injectParticlesAtPositions('flux', Motion_Solver part_solver, Field field, Species species, [double] delta_n, [double] n_vel, double delta_pos) =
#           The method creates 'delta_n' particles at each entry of 'pos' stored in the parameter 'flux' (See Documentation of 'createDistributionArBorder').
#           The new particles are stored in 'species', shifted 'delta_pos' away from their borders, initiated with 'n_vel' velocities and prepared in time
#           according to the method used 'part_solver' for advancing particles.
    @Timing
    def injectParticlesAtPositions(self, hit, part_solver, field, species, delta_n, n_vel, delta_pos = 1e-5):
        sum_particles = numpy.sum(delta_n)
        if sum_particles > 0:
            #Unfolding
            border = numpy.repeat(hit[0][:,2], delta_n)
            pos = numpy.repeat(hit[0][:,:2], delta_n, axis = 0)
            pos_copy = copy.copy(pos)
            #Assigning positions
            #NOTE: This part is not necessary but I am including it to be cleaner. It can be deleted for time efficiency.
            pos[:,1] += numpy.where(border == 0, -delta_pos, 0)
            pos[:,0] += numpy.where(border == 1, delta_pos, 0)
            pos[:,1] += numpy.where(border == 2, delta_pos, 0)
            pos[:,0] += numpy.where(border == 3, -delta_pos, 0)

            #Assigning velocities
            vel = super().sampleIsotropicVelocity(numpy.asarray([n_vel[0]]), sum_particles)
            vel[:,1] *= numpy.where(numpy.logical_and(vel[:,1] > 0, border == 0), -1, 1)
            vel[:,0] *= numpy.where(numpy.logical_and(vel[:,0] < 0, border == 1), -1, 1)
            vel[:,1] *= numpy.where(numpy.logical_and(vel[:,1] < 0, border == 2), -1, 1)
            vel[:,0] *= numpy.where(numpy.logical_and(vel[:,0] > 0, border == 3), -1, 1)

            ##Test
            ##Test positioning
            #fig = plt.figure(figsize=(8,8))
            #plt.scatter(pos[:,0], pos[:,1], marker = '.')
            #plt.title(self.type+" - "+species.name)
            #plt.show()
            #Test velocity
            #fig = plt.figure(figsize=(8,8))
            #datamag = plt.hist(numpy.linalg.norm(vel, axis = 1), 41, alpha=0.5, label=species.name)
            #plt.axvline(x=c.P_V_TH_MP*numpy.sqrt(2/3), label = 'protons', color = 'red')
            #plt.axvline(x=c.E_V_TH_MP*numpy.sqrt(2/3), label = 'electrons', color = 'blue')
            #plt.axvline(x=c.PHE_V_TH_MP*numpy.sqrt(2/3), label = 'photoelectrons', color = 'black')
            #plt.axvline(x=c.SEE_V_TH_MP*numpy.sqrt(2/3), label = 'SEE', color = 'purple')
            #plt.title(self.type+" - "+species.name)
            #plt.legend()
            #plt.show()
            #Test with more precision
            #filename = self.type+"_"+species.name+".txt"
            #with open(filename, 'ab') as f:
            #    array = numpy.append(pos, vel, axis = 1)
            #    numpy.savetxt(f, array)

            #Adding particles
            np = numpy.shape(pos)[0]
            self.addParticles(species, pos, vel)
            part_solver.initialConfiguration(species, field, ind = numpy.arange(species.part_values.current_n-sum_particles, species.part_values.current_n, dtype = numpy.uint32))
            self.updateTrackers(species, np)

            #Calculating outgoing flux
            hit = (numpy.append(numpy.append(pos_copy, border[:,None], axis = 1), species.spwt*numpy.ones_like((border))[:,None],axis = 1),\
                    numpy.where(border%2 == 0, numpy.abs(vel[:,1]), numpy.abs(vel[:,0])))
            part_solver.pic.scatterOutgoingFlux(species, hit)

            print("Injected particles: ", np)
            print("Total {}".format(species.name),": ", species.part_values.current_n)

#       +injectParticlesAtPositions_smooth('flux', Motion_Solver part_solver, Field field, Species species, [double] delta_n, [double] n_vel, double delta_pos) =
#           The method creates 'delta_n' particles at each entry of 'pos' stored in the parameter 'flux' (See Documentation of 'createDistributionArBorder').
#           The new particles are shifted from 'pos' according to the border  they are close to, in random distances, such as to create a smooth flux from timestep to timestep.
#           The new particles are stored in 'species', shifted 'delta_pos' away from their borders, initiated with 'n_vel' velocities and prepared in time
#           according to the method used 'part_solver' for advancing particles.
    @Timing
    def injectParticlesAtPositions_smooth(self, hit, part_solver, field, species, delta_n, n_vel, dt, delta_pos = 1e-5):
        sum_particles = numpy.sum(delta_n)
        if sum_particles > 0:
            #Unfolding
            border = numpy.repeat(hit[0][:,2], delta_n)
            pos = numpy.repeat(hit[0][:,:2], delta_n, axis = 0)
            pos_copy = copy.copy(pos)

            #Assigning positions
            rand = numpy.random.rand(pos.shape[0])
            #Average velocity in one direction
            vx = n_vel[0]/2/1.7724538509055159         #That number is sqrt(pi)
            pos[:,1] += numpy.where(border == 0, -(delta_pos+vx*dt*rand), 0)
            pos[:,0] += numpy.where(border == 1,  (delta_pos+vx*dt*rand), 0)
            pos[:,1] += numpy.where(border == 2,  (delta_pos+vx*dt*rand), 0)
            pos[:,0] += numpy.where(border == 3, -(delta_pos+vx*dt*rand), 0)

            #Assigning velocities
            vel = super().sampleIsotropicVelocity(numpy.asarray([n_vel[0]]), sum_particles)
            vel[:,1] *= numpy.where(numpy.logical_and(vel[:,1] > 0, border == 0), -1, 1)
            vel[:,0] *= numpy.where(numpy.logical_and(vel[:,0] < 0, border == 1), -1, 1)
            vel[:,1] *= numpy.where(numpy.logical_and(vel[:,1] < 0, border == 2), -1, 1)
            vel[:,0] *= numpy.where(numpy.logical_and(vel[:,0] > 0, border == 3), -1, 1)

            ##Test
            ##Test positioning
            #fig = plt.figure(figsize=(8,8))
            #plt.scatter(pos[:,0], pos[:,1], marker = '.')
            #plt.title(self.type+" - "+species.name)
            #plt.show()
            #Test velocity
            #fig = plt.figure(figsize=(8,8))
            #datamag = plt.hist(numpy.linalg.norm(vel, axis = 1), 41, alpha=0.5, label=species.name)
            #plt.axvline(x=c.P_V_TH_MP*numpy.sqrt(2/3), label = 'protons', color = 'red')
            #plt.axvline(x=c.E_V_TH_MP*numpy.sqrt(2/3), label = 'electrons', color = 'blue')
            #plt.axvline(x=c.PHE_V_TH_MP*numpy.sqrt(2/3), label = 'photoelectrons', color = 'black')
            #plt.axvline(x=c.SEE_V_TH_MP*numpy.sqrt(2/3), label = 'SEE', color = 'purple')
            #plt.title(self.type+" - "+species.name)
            #plt.legend()
            #plt.show()
            #Test with more precision
            #filename = self.type+"_"+species.name+".txt"
            #with open(filename, 'ab') as f:
            #    array = numpy.append(pos, vel, axis = 1)
            #    numpy.savetxt(f, array)

            #Adding particles
            np = numpy.shape(pos)[0]
            self.addParticles(species, pos, vel)
            part_solver.initialConfiguration(species, field, ind = numpy.arange(species.part_values.current_n-sum_particles, species.part_values.current_n, dtype = numpy.uint32))
            self.updateTrackers(species, np)

            #Calculating outgoing flux
            hit = (numpy.append(numpy.append(pos_copy, border[:,None], axis = 1), species.spwt*numpy.ones_like(border)[:,None],axis = 1),\
                    numpy.where(border%2 == 0, numpy.abs(vel[:,1]), numpy.abs(vel[:,0])))
            part_solver.pic.scatterOutgoingFlux(species, hit)

            print("Injected particles: ", np)
            print("Total {}".format(species.name),": ", species.part_values.current_n)
