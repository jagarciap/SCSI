#Data structure of the Boundaries
import constants as c
import copy
#Delete later
import matplotlib.pyplot as plt
import numpy

from Species.species import Species
from timing import Timing

#Boundary (Abstract-like: will also have common methods that can be used by sub-classes. In composition with mesh):
#
#Definition = Class that shows the methods and attributes needed for each particular boundary arrangement.
#Attributes:
#	+type (string) = string indicating the type of boundary. It will be of the form "[Inner or Outer] - [Source, e.g. Spacecraft, Component, etc.]".
#       +material (string) = String describing the material which is in contact with the boundary. So far, possible options: "space", "satellite".
#	+location ([int]) = array containing the indices of the nodes that the boundary represents.
#       +directions ([int]) = Number indicating outward direction of the boundary (from domain to outside). 0: Bottom, 1: Right, 2: Top, 3: Left.
#       +areas ([double]) = Areas assigned to the nodes of the boundary.
#       +adjacent [{int:int}] = A list, with one element per node of the boundary, where the element is a dictionary with entries direction = node, where,
#           for each direction that the node handles, the numeration of the adjacent node in that direction is stored.
#Methods:
#       +checkPositionInBoundary([double, double], Boolean surface): [Boolean] = Returns for each pos whether the position is inside or outside the border.
#	+applyElectricBoundary(Electric_Field) = Applies the boundary condition to the electric field passed as argument.
#	+applyMagneticBoundary(Magnetic_Field) = Applies the boundary condition to the magnetic field passed as argument.
#	+applyParticleBoundary(Species) = Applies the boundary condition to the species passed as argument.
#   +addParticles(Species species, [double, double] pos, [double, double] vel) = Add to Species the new particles, each represented by a row in pos and vel.
#   +updateTrackers(Species species, int new_particles) = Add new indexes to species.part_values.trackers, indicating the new particles to track.
#       new_particles indicates the latest amount of particles that were added to species.
#   +removeParticles(Species species, [ind] ind, Boolean tracker) = Removes the particles from species stored at 'ind' positions.
#       tracker indicates whether there is need for handling a Tracker instance.
#   +sampleIsotropicVelocity(double vth, int num) = It receives the most probable speed vth = \sqrt{2kT/m} and creates num random 2D velocities
#       with their magnitudes following a Maxwellian distribution.
#   +injectParticlesFace(self, location, pic, species, delta_n, n_vel, shift_vel): Function that inject particles into the domain.
#   +injectParticlesDummyBox([int] location, PIC pic, Field field, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) =
#       Inject the particles in location indices by creating dummy boxes around them, creating particles inside of them, moving the particles,
#       and then adding the ones that entered into the computational domain.
class Boundary(object):
    def applyElectricBoundary(self, e_field):
        pass
    
    def applyMagneticBoundary(self, m_field):
        pass

    def applyParticleBoundary(self, species):
        pass

    def createDummyBox (self, location, pic, species, delta_n, n_vel, shift_vel):
        pass

    def thermalVelocity(self, temperature, mass):
        return numpy.sqrt(2*c.K*temperature/mass)

#       +sampleIsotropicVelocity([double] vth, [int] num) [double,double] = It receives an array of most probable speeds vth = \sqrt{2kT/m} and creates
#           for each speed num 2D velocities with their magnitudes following a Maxwellian distribution.
#       NOTE: This function needs to be revised. random should not spread throughout different cells, and should use the same seed for every function call.
#    @nb.vectorize(signature = nb.double[:], target='cpu')
    def sampleIsotropicVelocity(self, vth, num):
        #Prepare for the handling of different sets of temperature
        total = numpy.sum(num)
        index = numpy.repeat(numpy.arange(len(vth)), num)
        #pick maxwellian velocities
        rand_spread = numpy.random.rand(total,6)
        vm_x = numpy.sqrt(2)*vth[index]*(rand_spread[:,0]+rand_spread[:,1]+rand_spread[:,2]-1.5)
        vm_y = numpy.sqrt(2)*vth[index]*(rand_spread[:,3]+rand_spread[:,4]+rand_spread[:,5]-1.5)
        ##NOTE: Delete later
        #val = plt.hist(vm, 40)
        #length = val[1][1]-val[1][0]
        #integral = length*numpy.sum(val[0])
        #A = val[0].max()
        #x = numpy.linspace( val[1].min(), val[1].max(), num = 50)
        #y = A*numpy.exp(-x*x/vth/vth)
        #plt.plot(x,y)
        #pdb.set_trace()
        #plt.show()
        #2D components of velocity 
        return numpy.append(vm_x[:,None], vm_y[:, None], axis = 1)

#       +Add new particles to species. pos and vel represent their positions and velocities, respectively.
    def addParticles(self, species, pos, vel, spwt = None):
        n = numpy.shape(pos)[0]
        if species.part_values.current_n + n > species.part_values.max_n:
            raise ValueError ("Too many particles")
        #store position and velocity of this particle
        species.part_values.position[species.part_values.current_n:species.part_values.current_n+n]= pos
        species.part_values.velocity[species.part_values.current_n:species.part_values.current_n+n]= vel
        if spwt is None:
            species.part_values.spwt[species.part_values.current_n:species.part_values.current_n+n]= species.spwt
        else:
            species.part_values.spwt[species.part_values.current_n:species.part_values.current_n+n]= spwt
        #increment particle counter
        species.part_values.current_n += n

#       +Add new indexes to species.part_values.trackers, indicating the new particles to track.
#           new_particles indicates the latest amount of particles that were added to species.
    def updateTrackers(self, species, new_particles):
        #Updating trackers
        if species.part_values.num_tracked != 0:
            ind_new = numpy.flatnonzero(species.part_values.trackers == species.part_values.max_n)
            num_new = len(ind_new)
            if num_new != 0:
                init = species.part_values.current_n-new_particles
                if new_particles < num_new:
                    step = 1
                    end = species.part_values.current_n
                    ind2 = new_particles
                else:
                    step = new_particles//num_new
                    end = init+step*num_new
                    ind2 = num_new
                species.part_values.trackers[ind_new[:ind2]] = numpy.arange(init, end, step, dtype = numpy.uint32)


#       +Eliminates the particles denoted with the indices ind. 
    def removeParticles(self, species, ind):
        #Eliminating particles
        temp = species.part_values.current_n
        species.part_values.current_n -= numpy.shape(ind)[0]
        species.part_values.position[:species.part_values.current_n,:] = numpy.delete(species.part_values.position[:temp,:], ind, axis = 0)
        species.part_values.velocity[:species.part_values.current_n,:] = numpy.delete(species.part_values.velocity[:temp,:], ind, axis = 0)
        species.part_values.spwt[:species.part_values.current_n] = numpy.delete(species.part_values.spwt[:temp], ind, axis = 0)
        #Updating trackers
        if species.part_values.num_tracked != 0:
            used_trackers = numpy.flatnonzero(species.part_values.trackers != species.part_values.max_n)
            if len(used_trackers) != 0:
                #Sorting both trackers and indices to be deleted
                ind1 = numpy.sort(ind)
                ind_trackers = numpy.argsort(species.part_values.trackers[used_trackers])
                trackers1 = species.part_values.trackers[used_trackers][ind_trackers]
                ind_c = 0
                tracker_c = 0
                n = 0
                #Processing the elimination of particles into trackers
                while ind_c != len(ind) and tracker_c != species.part_values.num_tracked:
                    if ind1[ind_c] < trackers1[tracker_c]:
                        n += 1
                        ind_c += 1
                        continue
                    elif ind1[ind_c] > trackers1[tracker_c]:
                        trackers1[tracker_c] -= n
                        tracker_c += 1
                        continue
                    elif ind1[ind_c] == trackers1[tracker_c]:
                        trackers1[tracker_c] = species.part_values.max_n
                        n += 1
                        tracker_c += 1
                        ind_c += 1
                if ind_c == len(ind1) and tracker_c < len(trackers1):
                    trackers1[tracker_c:] -= n
                #Updating trackers
                species.part_values.trackers[used_trackers][ind_trackers] = trackers1

#       +Function that inject particles into the domain.
#       +Parameters: 
#       +location ([ind]) = Nodes indicating the faces where the particles are going to be inserted. Each node represents the volume surrounding it. Location should be ordered increasingly.
#       +pic (PIC) = Instance of PIC for calculations. It also constains mesh, which is used in the function.
#       +species (Species) = Species to be inserted. It contains inside Particles_In_Mesh 'residuals' which is for each node the values of remnants from the insertion at the previous step.
#       +delta_n ([double]) = For each node in location, the density that is going to be inserted at this timestep. The array is ordered with the order of the nodes in mesh.
#       +n_vel ([double,double]) = For each node in location, the thermal velocity (for a the MB distribution) that the inserted particles will represent. Same order as delta_n.
#       +shift_vel ([double, double]) = For each node in location, an added velocity that does not come from temperature origin (like solar wind average speed.
#NOTE: I am doing a huge change in the conception of inject particle. I should give this a second thinking.
#NOTE: This function did not receive the update of checking whether ufunc were being used properly.
    def injectParticlesFace(self, location, pic, species, delta_n, n_vel, shift_vel): 
        # Useful constant
        box_x = shift_vel[:,0]*species.dt
        # Floating point production rate
        mpf_new = delta_n*pic.mesh.volumes[location]*box_x/pic.mesh.dx/species.spwt+species.mesh_values.residuals[location]
        # Truncate down, adding randomness
        mp_new = (mpf_new+numpy.random.rand(len(location))).astype(int)
        # Save fraction part
        species.mesh_values.residuals[location] = mpf_new-mp_new
        # Generate this many particles
        total_new = numpy.sum(mp_new)
        pos = numpy.zeros((total_new, species.pos_dim))
        vel = numpy.zeros((total_new, species.vel_dim))
        # Preparing positions (this part is mesh-dependant [To be depurated later or changed each time].
        # Preparating velocities. This part is mesh-dependant trough sampleIsotropicVelocity.
        #NOTE: Now I can do this for through numpy since now sampleIsotropicVelocity can receive many vths.
        phys_loc = pic.mesh.getPosition(location)
        c = 0
        for i in range(len(location)):
            vel[c:c+mp_new[i],:] += self.sampleIsotropicVelocity([n_vel[i]], mp_new[i])+shift_vel[i,:]
            pos[c:c+mp_new[i],0] += phys_loc[i,0] + numpy.random.rand(mp_new[i])*box_x[i]
            #pos[c:c+mp_new[i],0] += phys_loc[i,0] + (numpy.random.rand(mp_new[i])-0.5)*pic.mesh.dx
            pos[c:c+mp_new[i],1] += phys_loc[i,1] + (numpy.random.rand(mp_new[i])-0.5)*pic.mesh.dy
            c += mp_new[i]

        # Adjusting boundaries
        pos[:,1] += numpy.where(pos[:,1] <= pic.mesh.ymin, pic.mesh.dy/1.99, 0.0)
        pos[:,1] -= numpy.where(pos[:,1] >= pic.mesh.ymax, pic.mesh.dy/1.99, 0.0)
        pos[:,0] += numpy.where(pos[:,0] <= pic.mesh.xmin, pic.mesh.dx*0.001, 0.0)
        #pos[:,0] -= numpy.where(pos[:,0] >= pic.mesh.xmax, pic.mesh.dx/1.99, 0.0)
        #vel[:,0] = numpy.abs(vel[:,0])
        #Adding particles
        self.addParticles(species,pos,vel)
        print("Injected particles: ",total_new)
        print("Total{}".format(species.type), species.part_values.current_n)

#       +injectParticlesDummyBox([int] location, PIC pic, Field field, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = 
#               Inject the particles in location indices by creating dummy boxes around them, creating particles
#       	inside of them, moving the particles, and then adding the ones that entered into the computational domain.
    @Timing
    def injectParticlesDummyBox(self, location, part_solver, field, species, delta_n, n_vel, shift_vel):
        # Creating temporary species
        print("InjectParticlesDummyBox of boundary.py")
        ghost = Species("temporary species", species.dt, species.q, species.m, species.debye, species.spwt,\
                        int(species.part_values.max_n/10), species.pos_dim, species.vel_dim, species.mesh_values.nPoints, numpy.asarray([0]))
        ghost.mesh_values.residuals = species.mesh_values.residuals
        self.createDummyBox(location, part_solver.pic, ghost, delta_n, n_vel, shift_vel)
        species.mesh_values.residuals[location] = copy.copy(ghost.mesh_values.residuals[location])
        np = ghost.part_values.current_n
        #Entering particles into the mesh and adjusting them according to motion_solver
        ghost.part_values.position[:np,:] += ghost.part_values.velocity[:np,:]*ghost.dt
        ind = numpy.flatnonzero(numpy.logical_not(numpy.logical_and(ghost.part_values.position[:np,0] > part_solver.pic.mesh.xmin,\
                                numpy.logical_and(ghost.part_values.position[:np,0] < part_solver.pic.mesh.xmax,\
                                numpy.logical_and(ghost.part_values.position[:np,1] > part_solver.pic.mesh.ymin, ghost.part_values.position[:np,1] < part_solver.pic.mesh.ymax)))))
        self.removeParticles(ghost, ind)
        ###Test
        #np = ghost.part_values.current_n
        ##Test positioning
        #fig = plt.figure(figsize=(8,8))
        #plt.scatter(ghost.part_values.position[:np, 0], ghost.part_values.position[:np,1], marker = '.')
        #plt.title(species.name)
        #plt.show()
        ##Test velocity
        #fig = plt.figure(figsize=(8,8))
        #datamag = plt.hist(numpy.sqrt(species.part_values.velocity[:np,0]*species.part_values.velocity[:np,0]+\
        #                              species.part_values.velocity[:np,1]*species.part_values.velocity[:np,1]), 81, alpha=0.5, label=species.name)

        part_solver.initialConfiguration(ghost, field)
        #Adding particles
        self.addParticles(species, ghost.part_values.position[:ghost.part_values.current_n,:], ghost.part_values.velocity[:ghost.part_values.current_n,:])
        self.updateTrackers(species, ghost.part_values.current_n)
        print("Injected particles: ",ghost.part_values.current_n)
        print("Total {}".format(species.name),": ", species.part_values.current_n)
