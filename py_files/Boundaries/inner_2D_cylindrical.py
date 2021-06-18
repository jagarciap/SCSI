import copy
import numpy
import matplotlib.pyplot as plt
import pdb

import accelerated_functions as af
import constants as c
import cylindrical_mesh_tools as cmt
from Boundaries.boundary import Boundary
from Boundaries.inner_2D_rectangular import Inner_2D_Rectangular
from Species.species import Species
from solver import location_indexes_inv
from timing import Timing

#Inner_2D_Cylindrical (Inherits from Inner_2D_Rectangular):
#
#Definition = Inner rectangular boundary for a cylindrical (z-r) mesh
#Attributes:
#	+type (string) = "Inner - 2D_Cylindrical"
#	+xmin (double) = Left limit of the cavity (closest to the Sun).
#	+xmax (double) = Right limit of the cavity (farthest from the Sun).
#	+ymin (double) = Bottom limit of the cavity.
#	+ymax (double) = Top limit of the cavity.
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
#Comments:
#       +The boundary behaves differently depending on whether the bottom of the boundary is at r = 0 (cylinder) or if ymin > 0 (cylindrical shell). In the latter case, the boundary behaves
#           very similar to a Outer_2D_Rectangular boundary, whereas if ymin = 0, it is necessary to take into account the proper handling of the bottom part of the boundary.
class Inner_2D_Cylindrical(Inner_2D_Rectangular):
    type = "Inner - 2D_Cylindrical"
    def __init__(self, x_min, x_max , y_min, y_max, n_material):
        super().__init__(x_min, x_max , y_min, y_max, n_material)


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


 #       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = create the dummy boxes with particles in them.
    def createDummyBox(self, location, pic, species, delta_n, n_vel, shift_vel, prec = 1e-5):
        #Preparing things for numpy functions use
        loc, u_ind = numpy.unique(location, return_index = True)
        add_rand = numpy.random.rand(*numpy.shape(loc))
        #Volumes for the inner sections of the boundary cells
        y = (numpy.arange(pic.mesh.nPoints)//pic.mesh.nx)*pic.mesh.dy+pic.mesh.ymin
        if pic.mesh.ymin == 0.0:
            y[:pic.mesh.nx] = pic.mesh.dy/4
        dv = 2*numpy.pi*y*pic.mesh.dy*pic.mesh.dx
        dv = dv[loc]-pic.mesh.volumes[loc]
        mpf_new = delta_n[u_ind]*dv/species.spwt+\
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
        vel = self.sampleIsotropicVelocity(n_vel[u_ind], mp_new)+shift_vel[index]
        #Adding particles
        self.addParticles(species, pos, vel)


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
        ghost.part_values.position[:np,:] += ghost.part_values.velocity[:np,:2]*ghost.dt
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
        y = (numpy.arange(part_solver.pic.mesh.nPoints)//part_solver.pic.mesh.nx)*part_solver.pic.mesh.dy+part_solver.pic.mesh.ymin
        if part_solver.pic.mesh.ymin == 0.0:
            y[:part_solver.pic.mesh.nx] = part_solver.pic.mesh.dy/4
        dv = 2*numpy.pi*y*part_solver.pic.mesh.dy*part_solver.pic.mesh.dx
        mpf_new /= numpy.where(numpy.abs(dv[location]/part_solver.pic.mesh.volumes[location]-2) > 1e-3, 2, 1)

        #Computing number of particles created
        mpf_new = mpf_new/species.spwt+species.mesh_values.residuals[location]+add_rand
        mp_new = mpf_new.astype(int)
        species.mesh_values.residuals[location] = mpf_new-mp_new

        #Assigning positions
        pos = part_solver.pic.mesh.getPosition(location)
        pos_1 = numpy.repeat(pos, mp_new, axis = 0)
        hit_1 = self.directions[local_loc]
        ind_b = hit_1 == 0
        ind_l = hit_1 == 3
        ind_r = hit_1 == 1
        ind_t = hit_1 == 2

        #Bottom
        random = numpy.random.rand(numpy.sum(mp_new[ind_b]))
        random += numpy.where(random == 0, 1e-3, 0)
        ind = numpy.repeat(ind_b, mp_new)
        shifts = numpy.where(numpy.abs(pos_1[ind,0]-self.xmin) < prec, random*part_solver.pic.mesh.dx/2, (random-0.5)*part_solver.pic.mesh.dx)
        shifts -= numpy.where(numpy.abs(pos_1[ind,0]-self.xmax) < prec, random*part_solver.pic.mesh.dx/2, 0)
        pos_1[ind,0] += shifts
        #Left
        rmin = numpy.where(pos[ind_l,1] == self.ymin, pos[ind_l,1], pos[ind_l,1]-pic.mesh.dy/2)
        rmax = numpy.where(pos[ind_l,1] == self.ymax, pos[ind_l,1], pos[ind_l,1]+pic.mesh.dy/2)
        pos_1[numpy.repeat(ind_l, mp_new),1] = cmt.randomYPositions_2D_cm(pos[ind_l,:], rmin, rmax)
        #Right
        rmin = numpy.where(pos[ind_r,1] == self.ymin, pos[ind_r,1], pos[ind_r,1]-pic.mesh.dy/2)
        rmax = numpy.where(pos[ind_r,1] == self.ymax, pos[ind_r,1], pos[ind_r,1]+pic.mesh.dy/2)
        pos_1[numpy.repeat(ind_r, mp_new),1] = cmt.randomYPositions_2D_cm(pos[ind_r,:], rmin, rmax)
        #Top
        random = numpy.random.rand(numpy.sum(mp_new[ind_t]))
        random += numpy.where(random == 0, 1e-3, 0)
        ind = numpy.repeat(ind_t, mp_new)
        shifts = numpy.where(numpy.abs(pos_1[ind,0]-self.xmin) < prec, random*part_solver.pic.mesh.dx/2, (random-0.5)*part_solver.pic.mesh.dx)
        shifts -= numpy.where(numpy.abs(pos_1[ind,0]-self.xmax) < prec, random*part_solver.pic.mesh.dx/2, 0)
        pos_1[ind,0] += shifts

        hit_1 = numpy.repeat(hit_1, mp_new)
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
            vel = self.sampleIsotropicVelocity(numpy.asarray([n_vel[0]]), sum_particles)
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
            hit = (numpy.append(pos_copy, border[:,None], axis = 1), numpy.where(border%2 == 0, numpy.abs(vel[:,1]), numpy.abs(vel[:,0])))
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
            vel = self.sampleIsotropicVelocity(numpy.asarray([n_vel[0]]), sum_particles)
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
            hit = (numpy.append(pos_copy, border[:,None], axis = 1), numpy.where(border%2 == 0, numpy.abs(vel[:,1]), numpy.abs(vel[:,0])))
            part_solver.pic.scatterOutgoingFlux(species, hit)

            print("Injected particles: ", np)
            print("Total {}".format(species.name),": ", species.part_values.current_n)
