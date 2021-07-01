import copy
import matplotlib.pyplot as plt
import numpy
import pdb

import accelerated_functions as af
import constants as c
from Boundaries.boundary import Boundary
from solver import location_indexes_inv
from Species.species import Species
from timing import Timing


#Inner_1D_rectangular (Inherits from Boundary):
#
#Definition = One-dimensional boundary part of the inner boundary of a rectangular mesh
#Attributes:
#	+type (string) = "Inner - 1D_Rectangular"
#	+xmin (double) = Left limit of the boundary.
#	+xmax (double) = Right limit of the boundary.
#	+ymin (double) = Bottom limit of the boundary.
#	+ymax (double) = Top limit of the boundary.
#       +Boundary attributes
#Methods:
#	+Boundary methods.
class Inner_1D_Rectangular(Boundary):
    type = "Inner - 1D_Rectangular"
    def __init__(self, x_min, x_max , y_min, y_max, n_material):
        self.material = n_material

        self.xmin = x_min
        self.xmax = x_max
        self.ymin = y_min
        self.ymax = y_max

        self.location = []
        self.directions = []
        self.areas = []
        self.adjacent = []

#       NOTE: This way of treating the boundary does not take into account the particles that cross the plane defined by the boundary that do not properly cross the boundary.
#           Example: if the boundary is a right boundary, the cases where pos_x > xmax, but also ymax > pos_y or ymin < pos_y.
    def checkPositionInBoundary(self, pos, surface = False, prec = 1e-3):
        if self.directions[0] == 0:
            diff = self.ymin-pos[:,1]
        elif self.directions[0] == 1:
            diff = pos[:,0]-self.xmax
        elif self.directions[0] == 2:
            diff = pos[:,1]-self.ymax
        elif self.directions[0] == 3:
            diff = self.xmin-pos[:,0]

        if surface:
            return af.geq_1D_p(diff, prec)
        else:
            return af.g_1D_p(diff, prec)

    #	+applyElectricBoundary(Electric_Field) = Applies the boundary condition to the electric field passed as argument. 
    #       So far the a Dirichlet boundary condition, with the same potential already present, is implemented.
    def applyElectricBoundary(self, e_field):
        if self.directions[0] == 0:
            values = e_field.potential[self.location+e_field.pic.mesh.nx]
        elif self.directions[0] == 1:
            values = e_field.potential[self.location]
        elif self.directions[0] == 2:
            values = e_field.potential[self.location-e_field.pic.mesh.nx]
        elif self.directions[0] == 3:
            values = numpy.zeros_like(self.location)
        e_field.dirichlet(values, self, e_field.pic.mesh.nx, e_field.pic.mesh.ny, e_field.pic.mesh.dx, e_field.pic.mesh.dy)

    def applyMagneticBoundary(self, m_field):
        pass

#   def applyParticleBoundary(self, species):
#       raise Exception('This method in Outer_1D_rectangular should not be executed. \n'+\
#           'xmin, xmax, ymin, ymax: {:e}, {:e}, {:e}, {:e}'.format(self.xmin, self.xmax, self.ymin, self.ymax))

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
        in_ind = self.checkPositionInBoundary(species.part_values.position[:np,:])
        out_ind = numpy.flatnonzero(numpy.logical_not(in_ind))

        if type_boundary == 'mixed':
            rand = numpy.random.rand(len(out_ind))
            mask_albedo = rand < albedo
            self.applyParticleReflectiveBoundary(species, out_ind[mask_albedo], old_position = kwargs['old_position'])
            return self.applyParticleOpenBoundary(species, out_ind[numpy.logical_not(mask_albedo)], old_position = kwargs['old_position'])
        elif type_boundary == 'open':
            return self.applyParticleOpenBoundary(species, out_ind, old_position = kwargs['old_position'])
        elif type_boundary == 'reflective':
            return self.applyParticleReflectiveBoundary(species, out_ind, kwargs['old_position'])
        else:
            raise ValueError("Called invalid boundary method")

#       +applyParticleOpenBoundary(Species) = Deletes particles at or outside of the boundaries. In this case the particles that are to be eliminated are sent in 'ind'.
    def applyParticleOpenBoundary(self, species, ind, old_position = None):
        #Just for convenience in writing
        np = species.part_values.current_n
        coord = None
        vel = None
        tan_vel = None
        cos = None
        #if species.name == "Electron - Photoelectron":
        #    pdb.set_trace()
        if old_position is not None:
            #Bottom
            if self.directions[0] == 0:
                slope = (species.part_values.position[ind,0]-old_position[ind,0])/\
                        (species.part_values.position[ind,1]-old_position[ind,1])
                hit = slope*(self.ymin-old_position[ind,1])+old_position[ind,0]
                hit_ind = numpy.nonzero(numpy.logical_and(self.xmin <= hit, hit <= self.xmax))[0]
                coord = numpy.append(hit[hit_ind,None], numpy.append(self.ymin*numpy.ones_like((hit_ind))[:,None],\
                                                              numpy.zeros_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = -copy.copy(species.part_values.velocity[ind[hit_ind],1])
                tan_vel = copy.copy(species.part_values.velocity[ind[hit_ind],0])
                cos = 1/numpy.sqrt(slope*slope+1)[hit_ind]
            #Left
            elif self.directions[0] == 3:
                slope = (species.part_values.position[ind,1]-old_position[ind,1])/\
                        (species.part_values.position[ind,0]-old_position[ind,0])
                hit = slope*(self.xmin-old_position[ind,0])+old_position[ind,1]
                hit_ind = numpy.nonzero(numpy.logical_and(self.ymin <= hit, hit <= self.ymax))[0]
                coord = numpy.append(self.xmin*numpy.ones_like((hit_ind))[:,None], numpy.append(hit[hit_ind,None],\
                                        3*numpy.ones_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = -species.part_values.velocity[ind[hit_ind], 0]
                tan_vel = species.part_values.velocity[ind[hit_ind], 1]
                cos = 1/numpy.sqrt(slope*slope+1)[hit_ind]
            #Right
            elif self.directions[0] == 1:
                slope = (species.part_values.position[ind,1]-old_position[ind,1])/\
                        (species.part_values.position[ind,0]-old_position[ind,0])
                hit = slope*(self.xmax-old_position[ind,0])+old_position[ind,1]
                hit_ind = numpy.nonzero(numpy.logical_and(self.ymin <= hit, hit <= self.ymax))[0]
                coord = numpy.append(self.xmax*numpy.ones_like((hit_ind))[:,None], numpy.append(hit[hit_ind,None],\
                                        numpy.ones_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = species.part_values.velocity[ind[hit_ind], 0]
                tan_vel = species.part_values.velocity[ind[hit_ind], 1]
                cos = 1/numpy.sqrt(slope*slope+1)[hit_ind]
            #Top
            else:
                slope = (species.part_values.position[ind,0]-old_position[ind,0])/\
                        (species.part_values.position[ind,1]-old_position[ind,1])
                hit = slope*(self.ymax-old_position[ind,1])+old_position[ind,0]
                hit_ind = numpy.nonzero(numpy.logical_and(self.xmin <= hit, hit <= self.xmax))[0]
                coord = numpy.append(hit[hit_ind,None], numpy.append(self.ymax*numpy.ones_like((hit_ind))[:,None],\
                                                                2*numpy.ones_like((hit_ind), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = species.part_values.velocity[ind[hit_ind], 1]
                tan_vel = species.part_values.velocity[ind[hit_ind], 0]
                cos = 1/numpy.sqrt(slope*slope+1)[hit_ind]

        # Eliminating particles
        self.removeParticles(species,ind)
        count2 = numpy.shape(ind)[0]
        print('Number of {} eliminated - outer - Direction {:d}:'.format(species.name, self.directions[0]), count2)
        #Positions of deleted particles for posterior processing of flux
        if self.material != "space":
            return {'flux': (coord, vel, tan_vel, cos), 'del_ind': ind}

#       +applyParticleOpenBoundaryInverse(Species) = This function, as 'applyParticleOpenBoundary', identifies where  and under which parameters the particles cross the border,
#           but does not eliminate them. The method is used for calculating the Outgoing flux.
    def applyParticleOpenBoundaryInverse(self, species, ind, old_position = None):
        np = species.part_values.current_n
        if old_position is not None:
            #Bottom
            if self.directions[0] == 0:
                slope = (species.part_values.position[ind,0]-old_position[ind,0])/\
                        (species.part_values.position[ind,1]-old_position[ind,1])
                hit = slope*(self.ymin-old_position[ind,1])+old_position[ind,0]
                coord = numpy.append(hit[:,None], numpy.append(self.ymin*numpy.ones_like((hit))[:,None],\
                                                              numpy.zeros_like((hit), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = copy.copy(species.part_values.velocity[ind,1])
                tan_vel = copy.copy(species.part_values.velocity[ind,0])
                cos = 1/numpy.sqrt(slope*slope+1)
            #Left
            elif self.directions[0] == 3:
                slope = (species.part_values.position[ind,1]-old_position[ind,1])/\
                        (species.part_values.position[ind,0]-old_position[ind,0])
                hit = slope*(self.xmin-old_position[ind,0])+old_position[ind,1]
                coord = numpy.append(self.xmin*numpy.ones_like((hit))[:,None], numpy.append(hit[:,None],\
                                        3*numpy.ones_like((hit), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = species.part_values.velocity[ind, 0]
                tan_vel = species.part_values.velocity[ind, 1]
                cos = 1/numpy.sqrt(slope*slope+1)
            #Right
            elif self.directions[0] == 1:
                slope = (species.part_values.position[ind,1]-old_position[ind,1])/\
                        (species.part_values.position[ind,0]-old_position[ind,0])
                hit = slope*(self.xmax-old_position[ind,0])+old_position[ind,1]
                coord = numpy.append(self.xmax*numpy.ones_like((hit))[:,None], numpy.append(hit[:,None],\
                                        numpy.ones_like((hit), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = -species.part_values.velocity[ind, 0]
                tan_vel = species.part_values.velocity[ind, 1]
                cos = 1/numpy.sqrt(slope*slope+1)
            #Top
            else:
                slope = (species.part_values.position[ind,0]-old_position[ind,0])/\
                        (species.part_values.position[ind,1]-old_position[ind,1])
                hit = slope*(self.ymax-old_position[ind,1])+old_position[ind,0]
                coord = numpy.append(hit[:,None], numpy.append(self.ymax*numpy.ones_like((hit))[:,None],\
                                                                2*numpy.ones_like((hit), dtype = numpy.short)[:,None], axis = 1), axis = 1)
                vel = -species.part_values.velocity[ind, 1]
                tan_vel = species.part_values.velocity[ind, 0]
                cos = 1/numpy.sqrt(slope*slope+1)

        #Positions of deleted particles for posterior processing of flux
        return {'flux': (coord, vel, tan_vel, cos)}

    def applyParticleReflectiveBoundary(self, species, ind, old_position = None):
        delta = 2e-3
        if old_position is not None:
            #Bottom
            if self.directions[0] == 0:
                hit = (species.part_values.position[ind,0]-old_position[ind,0])/\
                      (species.part_values.position[ind,1]-old_position[ind,1])*\
                      (self.ymin-old_position[ind,1])+old_position[ind,0]
                species.part_values.position[ind, 1] = 2*self.ymin - species.part_values.position[ind,1]+delta
                species.part_values.velocity[ind, 1] *= -1.0
            #Left
            elif self.directions[0] == 3:
                hit = (species.part_values.position[ind,1]-old_position[ind,1])/ \
                      (species.part_values.position[ind,0]-old_position[ind,0])* \
                      (self.xmin-old_position[ind,0])+old_position[ind,1]
                species.part_values.position[ind, 0] = 2*self.xmin - species.part_values.position[ind,0]+delta
                species.part_values.velocity[ind, 0] *= -1.0
            #Right
            elif self.directions[0] == 1:
                hit = (species.part_values.position[ind,1]-old_position[ind,1])/ \
                      (species.part_values.position[ind,0]-old_position[ind,0])* \
                      (self.xmax-old_position[ind,0])+old_position[ind,1]
                species.part_values.position[ind, 0] = 2*self.xmax - species.part_values.position[ind,0]-delta
                species.part_values.velocity[ind, 0] *= -1.0
            #Top
            else:
                hit = (species.part_values.position[ind,0]-old_position[ind,0])/ \
                      (species.part_values.position[ind,1]-old_position[ind,1])* \
                      (self.ymax-old_position[ind,1])+old_position[ind,0]
                species.part_values.position[ind, 1] = 2*self.ymax - species.part_values.position[ind,1]-delta
                species.part_values.velocity[ind, 1] *= -1.0

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
        #Bottom
        if self.directions[0] == 0:
            shift = numpy.where(numpy.abs(pos[:,0]-self.xmin) < prec, random[:,0]/2*pic.mesh.dx, (random[:,0]-0.5)*pic.mesh.dx)
            pos[:,0] = pos[:,0] + shift - numpy.where(numpy.abs(pos[:,0]-self.xmax) < prec, random[:,0]/2*pic.mesh.dx, 0)
            pos[:,1] -= random[:,1]/2*pic.mesh.dy
        #Left
        elif self.directions[0] == 3:
            shift = numpy.where(numpy.abs(pos[:,1]-self.ymin) < prec, random[:,1]/2*pic.mesh.dy, (random[:,1]-0.5)*pic.mesh.dy)
            pos[:,1] = pos[:,1] + shift - numpy.where(numpy.abs(pos[:,1]-self.ymax) < prec, random[:,1]/2*pic.mesh.dy, 0)
            pos[:,0] -= random[:,0]/2*pic.mesh.dx

        #Right
        elif self.directions[0] == 1:
            shift = numpy.where(numpy.abs(pos[:,1]-self.ymin) < prec, random[:,1]/2*pic.mesh.dy, (random[:,1]-0.5)*pic.mesh.dy)
            pos[:,1] = pos[:,1] + shift - numpy.where(numpy.abs(pos[:,1]-self.ymax) < prec, random[:,1]/2*pic.mesh.dy, 0)
            pos[:,0] += random[:,0]/2*pic.mesh.dx
        #Top
        elif self.directions[0] == 2:
            shift = numpy.where(numpy.abs(pos[:,0]-self.xmin) < prec, random[:,0]/2*pic.mesh.dx, (random[:,0]-0.5)*pic.mesh.dx)
            pos[:,0] = pos[:,0] + shift - numpy.where(numpy.abs(pos[:,0]-self.xmax) < prec, random[:,0]/2*pic.mesh.dx, 0)
            pos[:,1] += random[:,1]/2*pic.mesh.dy
        else:
            pdb.set_trace()

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
        ind = numpy.flatnonzero(part_solver.pic.mesh.checkPositionInMesh(ghost.part_values.position[:np,:]))

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
        if self.material != "space":
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
        #NOTE: this should be generalized
        if self.material != 'space':
            local_loc = location_indexes_inv(location, store = False)
        else:
            c = 0
            local_loc = []
            for index, loc in numpy.ndenumerate(self.location):
                if loc == location[c]:
                    local_loc.append(index[0])
                    c += 1
            local_loc = numpy.asarray(local_loc, dtype = 'uint32')
        mpf_new = delta_n*self.areas[local_loc]
        mpf_new = mpf_new/species.spwt+species.mesh_values.residuals[location]+add_rand
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals[location] = mpf_new-mp_new

        #Assigning positions
        pos_1 = numpy.repeat(part_solver.pic.mesh.getPosition(location), mp_new, axis = 0)
        random = numpy.random.rand(numpy.shape(pos_1)[0])
        #Bottom
        if self.directions[local_loc[0]] == 0:
            shifts = numpy.where(numpy.abs(pos_1[:,0]-self.xmin) < prec, random*part_solver.pic.mesh.dx/2, (random-0.5)*part_solver.pic.mesh.dx)
            shifts -= numpy.where(numpy.abs(pos_1[:,0]-self.xmax) < prec, random*part_solver.pic.mesh.dx/2, 0)
            pos_1[:,0] += shifts
            hit_1 = numpy.zeros_like(pos_1[:,1], dtype = numpy.uint8)[:,None]
        #Left
        elif self.directions[local_loc[0]] == 3:
            shifts = numpy.where(numpy.abs(pos_1[:,1]-self.ymin) < prec, random*part_solver.pic.mesh.dy/2, (random-0.5)*part_solver.pic.mesh.dy)
            shifts -= numpy.where(numpy.abs(pos_1[:,1]-self.ymax) < prec, random*part_solver.pic.mesh.dy/2, 0)
            pos_1[:,1] += shifts
            hit_1 = 3*numpy.ones_like(pos_1[:,1], dtype = numpy.uint8)[:,None]
        #Right
        elif self.directions[local_loc[0]] == 1:
            shifts = numpy.where(numpy.abs(pos_1[:,1]-self.ymin) < prec, random*part_solver.pic.mesh.dy/2, (random-0.5)*part_solver.pic.mesh.dy)
            shifts -= numpy.where(numpy.abs(pos_1[:,1]-self.ymax) < prec, random*part_solver.pic.mesh.dy/2, 0)
            pos_1[:,1] += shifts
            hit_1 = numpy.ones_like(pos_1[:,1], dtype = numpy.uint8)[:,None]
        #Top
        else:
            shifts = numpy.where(numpy.abs(pos_1[:,0]-self.xmin) < prec, random*part_solver.pic.mesh.dx/2, (random-0.5)*part_solver.pic.mesh.dx)
            shifts -= numpy.where(numpy.abs(pos_1[:,0]-self.xmax) < prec, random*part_solver.pic.mesh.dx/2, 0)
            pos_1[:,0] += shifts
            hit_1 = 2*numpy.ones_like(pos_1[:,1], dtype = numpy.uint8)[:,None]

        repeats = numpy.ones(numpy.shape(hit_1)[0], dtype = numpy.uint8)
        return (numpy.append(pos_1, hit_1, axis = 1),), repeats
        
#       +injectParticlesAtPositions_smooth('flux', Motion_Solver part_solver, Field field, Species species, [double] delta_n, [double] n_vel, double delta_pos) =
#           The method creates 'delta_n' particles at each entry of 'pos' stored in the parameter 'flux' (See Documentation of 'createDistributionArBorder').
#           The new particles are shifted from 'pos' according to the border  they are close to, in random distances, such as to create a smooth flux from timestep to timestep.
#           The new particles are stored in 'species', shifted 'delta_pos' away from their borders, initiated with 'n_vel' velocities and prepared in time
#           according to the method used 'part_solver' for advancing particles.
    @Timing
    def injectParticlesAtPositions_smooth(self, hit, part_solver, field, species, delta_n, n_vel, drift_vel, dt, delta_pos = 1e-3):
        sum_particles = numpy.sum(delta_n)
        if sum_particles > 0:
            #Unfolding
            border = numpy.repeat(hit[0][:,2], delta_n)
            pos = numpy.repeat(hit[0][:,:2], delta_n, axis = 0)
            pos_copy = copy.copy(pos)

            #Assigning positions and velocities
            rand = numpy.random.rand(pos.shape[0])
            #NOTE: this should be updated
            vel = super().sampleIsotropicVelocity(numpy.asarray([n_vel[0]]), sum_particles)+numpy.repeat(numpy.reshape(drift_vel[0,:], (1,2)), sum_particles, axis = 0)
            #Average velocity in one direction
            #TODO: This should be generalized for n_vel being an array of different velocities for each node
            if border[0]%2 == 1:
                vx = n_vel[0]/2/1.7724538509055159+drift_vel[0,0]         #That number is sqrt(pi)
            else:
                vx = n_vel[0]/2/1.7724538509055159+drift_vel[0,1]         #That number is sqrt(pi)
            if border[0] == 0:
                pos[:,1] -= delta_pos+vx*dt*rand
                vel[:,1] *= numpy.where(vel[:,1] > 0, -1, 1)
            elif border[0] == 3:
                pos[:,0] -= delta_pos+vx*dt*rand
                vel[:,0] *= numpy.where(vel[:,0] > 0, -1, 1)
            elif border[0] == 1:
                pos[:,0] += delta_pos+vx*dt*rand
                vel[:,0] *= numpy.where(vel[:,0] < 0, -1, 1)
            elif border[0] == 2:
                pos[:,1] += delta_pos+vx*dt*rand
                vel[:,1] *= numpy.where(vel[:,1] < 0, -1, 1)
                print("Positions in y component: ", pos[:10,1], len(pos[:,1]))
            else:
                pdb.set_trace()

            ##Test
            ##Test positioning
            #fig = plt.figure(figsize=(8,8))
            #plt.scatter(pos[:,0], pos[:,1], marker = '.')
            #plt.title(self.type+" - "+species.name)
            #plt.show()
            ##Test velocity
            #fig = plt.figure(figsize=(8,8))
            #datamag = plt.hist(numpy.linalg.norm(vel, axis = 1), 41, alpha=0.5, label=species.name)
            #plt.axvline(x=c.P_V_TH_MP*numpy.sqrt(2/3), label = 'protons', color = 'red')
            #plt.axvline(x=c.E_V_TH_MP*numpy.sqrt(2/3), label = 'electrons', color = 'blue')
            #plt.axvline(x=c.PHE_V_TH_MP*numpy.sqrt(2/3), label = 'photoelectrons', color = 'black')
            #plt.axvline(x=c.SEE_V_TH_MP*numpy.sqrt(2/3), label = 'SEE', color = 'purple')
            #plt.title(self.type+" - "+species.name)
            #plt.legend()
            #plt.show()
            ###Test with more precision
            ##filename = self.type+"_"+species.name+".txt"
            ##with open(filename, 'ab') as f:
            ##    array = numpy.append(pos, vel, axis = 1)
            ##    numpy.savetxt(f, array)
            #pdb.set_trace()

            #Adding particles
            np = numpy.shape(pos)[0]
            self.addParticles(species, pos, vel)
            part_solver.initialConfiguration(species, field, ind = numpy.arange(species.part_values.current_n-sum_particles, species.part_values.current_n, dtype = numpy.uint32))
            self.updateTrackers(species, np)

            #Calculating outgoing flux
            #NOTE: this should be updated
            pdb.set_trace()
            if self.material != 'space' and self.material != 'HET':
                hit = (numpy.append(pos_copy, border[:,None], axis = 1), numpy.where(border%2 == 0, numpy.abs(vel[:,1]), numpy.abs(vel[:,0])))
                part_solver.pic.scatterOutgoingFlux(species, hit)

            print("Injected particles: ", np)
            print("Total {}".format(species.name),": ", species.part_values.current_n)
