import copy
import numpy
import matplotlib.pyplot as plt
import pdb

import accelerated_functions as af
import constants as c
import cylindrical_mesh_tools as cmt
from Boundaries.boundary import Boundary
from Boundaries.outer_2D_rectangular import Outer_2D_Rectangular
from Species.species import Species
from timing import Timing

#Outer_2D_Cylindrical (Inherits from Outer_2D_Rectangular):
#
#Definition = Outer boundary for a cylindrical (z-r) mesh
#Attributes:
#	+type (string) = "Outer - 2D_Cylindrical"
#	+xmin (double) = Left limit of the domain (closest to the Sun).
#	+xmax (double) = Right limit of the domain (farthest from the Sun).
#	+ymin (double) = Bottom limit of the domain.
#	+ymax (double) = Top limit of the domain.
#       +bottom ([int]) = array of indices that represents the bottom of the boundary.
#       +top ([int]) = array of indices that represents the top of the boundary.
#       +left ([int]) = array of indices that represents the left of the boundary.
#       +right ([int]) = array of indices that represents the right of the boundary.
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
#Comments:
#       +The boundary behaves differently depending on whether the bottom of the boundary is at r = 0 (cylinder) or if ymin > 0 (cylindrical shell). In the latter case, the boundary behaves
#           very similar to a Outer_2D_Rectangular boundary, whereas if ymin = 0, it is necessary to take into account the proper handling of the bottom part of the boundary.
class Outer_2D_Cylindrical(Outer_2D_Rectangular):
    type = "Outer - 2D_Cylindrical"
    def __init__(self, x_min, x_max , y_min, y_max, n_material):
        super().__init__(x_min, x_max, y_min, y_max, n_material)

#       +sampleIsotropicVelocity([double] vth, [int] num) [double,double,double] = It receives an array of most probable speeds vth = \sqrt{2kT/m} and creates
#           for each speed num 3D velocities with their magnitudes following a Maxwellian distribution.
#       NOTE: This function needs to be revised. random should not spread throughout different cells, and should use the same seed for every function call.
#    @nb.vectorize(signature = nb.double[:], target='cpu')
    def sampleIsotropicVelocity(self, vth, num):
        #Prepare for the handling of different sets of temperature
        total = numpy.sum(num)
        index = numpy.repeat(numpy.arange(len(vth)), num)
        #pick maxwellian velocities
        rand_spread = numpy.random.rand(total,9)
        vm_x = numpy.sqrt(2)*vth[index]*(rand_spread[:,0]+rand_spread[:,1]+rand_spread[:,2]-1.5)
        vm_y = numpy.sqrt(2)*vth[index]*(rand_spread[:,3]+rand_spread[:,4]+rand_spread[:,5]-1.5)
        vm_z = numpy.sqrt(2)*vth[index]*(rand_spread[:,6]+rand_spread[:,7]+rand_spread[:,8]-1.5)
        ##NOTE: Delete later
        #val = plt.hist(vm, 40)
        #length = val[1][1]-val[1][0]
        #integral = length*numpy.sum(val[0])
        #A = val[0].max()
        #x = numpy.linspace( val[1].min(), val[1].max(), num = 50)
        #y = A*numpy.exp(-x*x/vth/vth)
        #plt.plot(x,y)
        #plt.show()
        #pdb.set_trace()
        #3D components of velocity 
        return numpy.append(numpy.append(vm_x[:,None], vm_y[:, None], axis = 1), vm_z[:,None], axis = 1)

#	+applyParticleBoundary(Species) = Applies the boundary condition to the species passed as argument.
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default method, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    def applyParticleBoundary(self, species, type_boundary, **kwargs):
        if self.ymin == 0.0 and kwargs['old_position'] is not None:
            self.applyParticleReflectiveBottomBoundary(species, old_position = kwargs['old_position'])
        if type_boundary == 'open':
            return self.applyParticleOpenBoundary(species)
        elif type_boundary == 'reflective':
            return self.applyParticleReflectiveBoundary(species, old_position = kwargs['old_position'])
        else:
            raise ValueError("Called invalid boundary method")

#       +applyParticleReflectiveBottomBoundary(Species species, Species old_species) = Reflects the particles back into the domain, only for bottom boundary.
#           old_species refers to the state of species in the previous step.
    def applyParticleReflectiveBottomBoundary(self, species, old_position = None):
        #For ease of typing
        np = species.part_values.current_n
        delta = 1e-5
        #Lower wall
        ind = numpy.flatnonzero(species.part_values.position[:np,1] <= self.ymin)
        species.part_values.velocity[ind,1] *= -1
        pos = (self.ymin+delta)*numpy.ones((len(ind), species.pos_dim))
        pos[:,0] = (species.part_values.position[ind,0]-old_position[ind,0])/(species.part_values.position[ind,1]-old_position[ind,1])\
                *(self.ymin+delta-old_position[ind,1])+old_position[ind,0]
        species.part_values.position[ind,:] = pos

#       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = create the dummy boxes with particles in them.
#NOTE: I am not sure if addParticles is computationally demanding for other reason apart from the costs on numpy operations.
    def createDummyBox(self, location, pic, species, delta_n, n_vel, shift_vel, index=None):
        #Volumes without border consideration
        y = (numpy.arange(pic.mesh.nPoints)//pic.mesh.nx)*pic.mesh.dy+pic.mesh.ymin
        if pic.mesh.ymin == 0.0:
            #It is /8 instead of /4 in order to create particles only if r>=0
            y[:pic.mesh.nx] = pic.mesh.dy/8
        dv = 2*numpy.pi*y*pic.mesh.dy*pic.mesh.dx
        #New particles to be stored
        mpf_new = numpy.zeros((pic.mesh.nPoints))
        loc = numpy.unique(location)
        random_1 = numpy.random.rand(loc.shape[0])
        numpy.add.at(mpf_new, location, dv[location]*delta_n/species.spwt)
        mpf_new[loc] += species.mesh_values.residuals[loc] + random_1
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals = mpf_new - mp_new
        #Preparing indexes for numpy usage
        index = numpy.repeat(numpy.arange(len(loc)), mp_new[loc])
        #Setting up positions
        pos = pic.mesh.getPosition(loc)
        rmin = numpy.where(pos[:,1] == 0.0, pos[:,1], pos[:,1]-pic.mesh.dy/2)
        rmax = pos[:,1]+pic.mesh.dy/2
        pos = pos[index]
        pos[:,1] = cmt.randomYPositions_2D_cm(mp_new[loc], rmin, rmax)
        random_2 = numpy.random.rand(numpy.shape(pos)[0])
        pos[:,0] += (random_2-0.5)*pic.mesh.dx
        #Deleting unwanted particles
        if self.ymin == 0.0:
            mask = numpy.flatnonzero(numpy.logical_and(numpy.logical_or(numpy.logical_or(\
                pos[:,0] < self.xmin, pos[:,0] > self.xmax), pos[:,1] > self.ymax), pos[:,1] >= self.ymin))
        else:
            mask = numpy.flatnonzero(numpy.logical_or(numpy.logical_or(numpy.logical_or(\
                pos[:,0] < self.xmin, pos[:,0] > self.xmax), pos[:,1] > self.ymax), pos[:,1] < self.ymin))
        pos = pos[mask,:]
        #Setting up velocities
        vel = self.sampleIsotropicVelocity(n_vel[index[mask]], numpy.ones_like(pos[:,0], dtype = 'uint8'))+shift_vel[index[mask],:]
        #Adding particles
        self.addParticles(species,pos,vel)

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
        ghost.part_values.position[:np,:] += ghost.part_values.velocity[:np,:2]*ghost.dt
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
        #plt.axvline(x=c.P_V_TH_MP, label = 'protons', color = 'red')
        #plt.axvline(x=c.E_V_TH_MP, label = 'electrons', color = 'blue')
        #plt.axvline(x=c.PHE_V_TH_MP, label = 'photoelectrons', color = 'black')
        #plt.axvline(x=c.SEE_V_TH_MP, label = 'SEE', color = 'purple')
        #plt.title(self.type+" - "+species.name)
        #plt.legend()
        #plt.show()
        #pdb.set_trace()
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
