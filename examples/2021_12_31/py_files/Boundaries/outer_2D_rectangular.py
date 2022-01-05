import copy
import numpy
import matplotlib.pyplot as plt
import pdb

import accelerated_functions as af
import constants as c
from Boundaries.boundary import Boundary
from Species.species import Species
from timing import Timing

#Outer_2D_Rectangular (Inherits from Boundary):
#
#Definition = Outer boundary for a rectangular mesh
#Attributes:
#	+type (string) = "Outer - 2D_Rectangular"
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#       +bottom ([int]) = array of indices that indicates the represents the bottom of the boundary.
#       +top ([int]) = array of indices that indicates the represents the top of the boundary.
#       +left ([int]) = array of indices that indicates the represents the left of the boundary.
#       +right ([int]) = array of indices that indicates the represents the right of the boundary.
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
#             create the dummy boxes outside of the domain with particles in them, using delta_n (density), n_vel (thermal velocity), shift_vel (velocity shared by all particles).
#       +injectParticlesDummyBox([int] location, PIC pic, Field field, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) =
#               Inject the particles in location indices by creating dummy boxes around them, creating particles
#       	inside of them, moving the particles, and then adding the ones that entered into the computational domain.
class Outer_2D_Rectangular(Boundary):
    type = "Outer - 2D_Rectangular"
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
        if surface:
            return af.leq_2D_p(pos, xmin, xmax, ymin, ymax)
        else:
            return af.l_2D_p(pos, xmin, xmax, ymin, ymax)

#	+applyElectricBoundary(Electric_Field) = Applies the boundary condition to the electric field passed as argument. So far a 0V Dirichlet boundary condition is applied.
    def applyElectricBoundary(self, e_field):
        values = numpy.zeros((len(self.location)))
        e_field.dirichlet(values, self, e_field.pic.mesh.nx, e_field.pic.mesh.ny, e_field.pic.mesh.dx, e_field.pic.mesh.dy)

#	+applyMagneticBoundary(Magnetic_Field) = Applies the boundary condition to the magnetic field passed as argument.
#       No magnetic field so far
    def applyMagneticBoundary(self, m_field):
        pass

#	+applyParticleBoundary(Species) = Applies the boundary condition to the species passed as argument.
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    def applyParticleBoundary(self, species, type_boundary, **kwargs):
        if type_boundary == 'open':
            return self.applyParticleOpenBoundary(species)
        elif type_boundary == 'reflective':
            return self.applyParticleReflectiveBoundary(species, old_position = kwargs['old_position'])
        else:
            raise ValueError("Called invalid boundary method")

#       +applyParticleOpenBoundary(Species) = Deletes particles at or outside of the boundaries.
    def applyParticleOpenBoundary(self, species):
        #Just for convenience in writing and for use of accelerated functions
        np = species.part_values.current_n
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        #Finding the particles
        ind = numpy.flatnonzero(af.geq_2D_p(species.part_values.position[:np,:], xmin, xmax, ymin, ymax))
        # Eliminating particles
        super().removeParticles(species,ind)
        count2 = numpy.shape(ind)[0]
        print('Number of {} eliminated - outer:'.format(species.name), count2)
        return {'del_ind': ind}

#       +applyParticleReflectiveBoundary(Species species, Species old_species) = Reflects the particles back into the domain.
#           old_species refers to the state of species in the previous step.
#NOTE: This needs update
    def applyParticleReflectiveBoundary(self, species, old_species):
        #For ease of typing
        np = species.part_values.current_n
        delta = 2*10**(-c.INDEX_PREC)
        #Upper wall
        ind = numpy.flatnonzero(species.part_values.position[:np,1] >= self.ymax)
        species.part_values.velocity[ind,1] *= -1
        pos = (self.ymax-delta)*numpy.ones((len(ind), species.pos_dim))
        pos[:,0] = (species.part_values.position[ind,0]-old_species.part_values.position[ind,0])/(species.part_values.position[ind,1]-old_species.part_values.position[ind,1])\
                *(self.ymax-delta-old_species.part_values.position[ind,1])+old_species.part_values.position[ind,0]
        species.part_values.position[ind,:] = pos
        #Left wall
        ind = numpy.flatnonzero(species.part_values.position[:np,0] <= self.xmin)
        ind_v = numpy.flatnonzero(species.part_values.position[ind,1] == self.ymax-delta)
        species.part_values.velocity[ind[ind_v],1] *= numpy.where(species.part_values.position[ind[ind_v],0] == self.xmin+delta, 1, -1)
        species.part_values.velocity[ind,0] *= -1
        pos = (self.xmin+delta)*numpy.ones((len(ind), species.pos_dim))
        pos[:,1] = (species.part_values.position[ind,1]-old_species.part_values.position[ind,1])/(species.part_values.position[ind,0]-old_species.part_values.position[ind,0])\
                *(self.xmin+delta-old_species.part_values.position[ind,0])+old_species.part_values.position[ind,1]
        species.part_values.position[ind,:] = pos
        #Right wall
        ind = numpy.flatnonzero(species.part_values.position[:np,0] >= self.xmax)
        ind_v = numpy.flatnonzero(species.part_values.position[ind,1] == self.ymax-delta)
        species.part_values.velocity[ind[ind_v],1] *= numpy.where(species.part_values.position[ind[ind_v],0] == self.xmax-delta, 1, -1)
        species.part_values.velocity[ind,0] *= -1
        pos = (self.xmax-delta)*numpy.ones((len(ind), species.pos_dim))
        pos[:,1] = (species.part_values.position[ind,1]-old_species.part_values.position[ind,1])/(species.part_values.position[ind,0]-old_species.part_values.position[ind,0])\
                *(self.xmax-delta-old_species.part_values.position[ind,0])+old_species.part_values.position[ind,1]
        species.part_values.position[ind,:] = pos
        #Lower wall
        ind = numpy.flatnonzero(species.part_values.position[:np,1] <= self.ymin)
        ind_v = numpy.flatnonzero(species.part_values.position[ind,0] == self.xmax-delta)
        species.part_values.velocity[ind[ind_v],0] *= numpy.where(species.part_values.position[ind[ind_v],1] == self.ymin+delta, 1, -1)
        ind_v = numpy.flatnonzero(species.part_values.position[ind,0] == self.xmin+delta)
        species.part_values.velocity[ind[ind_v],0] *= numpy.where(species.part_values.position[ind[ind_v],1] == self.ymin+delta, 1, -1)
        species.part_values.velocity[ind,1] *= -1
        pos = (self.ymin+delta)*numpy.ones((len(ind), species.pos_dim))
        pos[:,0] = (species.part_values.position[ind,0]-old_species.part_values.position[ind,0])/(species.part_values.position[ind,1]-old_species.part_values.position[ind,1])\
                *(self.ymin+delta-old_species.part_values.position[ind,1])+old_species.part_values.position[ind,0]
        species.part_values.position[ind,:] = pos

#       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = create the dummy boxes with particles in them.
#NOTE: I am not sure if addParticles is computationally demanding for other reason apart from the costs on numpy operations.
    def createDummyBox(self, location, pic, species, delta_n, n_vel, shift_vel, index=None):
        #New particles to be stored
        mpf_new = numpy.zeros((pic.mesh.nPoints))
        loc = numpy.unique(location)
        random_1 = numpy.random.rand(loc.shape[0])
        numpy.add.at(mpf_new, location, pic.mesh.volumes[location[1]]*numpy.ones_like(location)*delta_n*2/species.spwt)
        mpf_new[loc] += species.mesh_values.residuals[loc] + random_1
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals = mpf_new - mp_new
        #Preparing indexes for numpy usage
        index = numpy.repeat(numpy.arange(len(loc)), mp_new[loc])
        #Setting up positions
        pos = pic.mesh.getPosition(loc)[index]
        random_2 = numpy.random.rand(*numpy.shape(pos))
        random_2 += numpy.where(random_2 == 0, 1e-3, 0)
        pos[:,0] += (random_2[:,0]-0.5)*pic.mesh.dx
        pos[:,1] += (random_2[:,1]-0.5)*pic.mesh.dy
        #Deleting unwanted particles
        mask = numpy.flatnonzero(numpy.logical_or(numpy.logical_or(numpy.logical_or(\
            pos[:,0] < self.xmin, pos[:,0] > self.xmax), pos[:,1] < self.ymin), pos[:,1] > self.ymax))
        pos = pos[mask,:]
        #Setting up velocities
        vel = super().sampleIsotropicVelocity(n_vel[index[mask]], numpy.ones_like(pos[:,0], dtype = 'uint8'))+shift_vel[index[mask],:]
        #Adding particles
        super().addParticles(species,pos,vel)

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
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        #Entering particles into the mesh and adjusting them according to motion_solver
        ghost.part_values.position[:np,:] += ghost.part_values.velocity[:np,:]*ghost.dt
        ind = numpy.flatnonzero(af.geq_2D_p(ghost.part_values.position[:np,:], xmin, xmax, ymin, ymax))

        self.removeParticles(ghost, ind)
        ##Test
        #np = ghost.part_values.current_n
        ##Test positioning
        #fig = plt.figure(figsize=(8,8))
        #plt.scatter(ghost.part_values.position[:np, 0], ghost.part_values.position[:np,1], marker = '.')
        #plt.title(self.type+" - "+species.name)
        #plt.show()
        ##Test velocity
        #fig = plt.figure(figsize=(8,8))
        #datamag = plt.hist(numpy.sqrt((ghost.part_values.velocity[:np,0]-shift_vel[0,0])*(ghost.part_values.velocity[:np,0]-shift_vel[0,0])+ \
        #                              ghost.part_values.velocity[:np,1]*ghost.part_values.velocity[:np,1]), 41, alpha=0.5, label=species.name)
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
        #    np = ghost.part_values.current_n
        #    array = numpy.append(ghost.part_values.position[:np,:], ghost.part_values.velocity[:np,:], axis = 1)
        #    numpy.savetxt(f, array)

        part_solver.initialConfiguration(ghost, field)
        #Adding particles
        self.addParticles(species, ghost.part_values.position[:ghost.part_values.current_n,:], ghost.part_values.velocity[:ghost.part_values.current_n,:])
        self.updateTrackers(species, ghost.part_values.current_n)
        print("Injected particles: ",ghost.part_values.current_n)
        print("Total {}".format(species.name),": ", species.part_values.current_n)
