

#---------------------------------------------------------------------------------------------------------------------------------------
# #1: I was trying to write the function using numpy instead of for
#---------------------------------------------------------------------------------------------------------------------------------------
#       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = create the dummy boxes with particles in them.
    def createDummyBox(self, location, pic, species, delta_n, n_vel, shift_vel):
        # Locating borders
        phys_loc = pic.mesh.getPosition(location)
        left = numpy.equal(phys_loc[:,0], pic.mesh.xmin)
        right = numpy.equal(phys_loc[:,0], pic.mesh.xmax)
        bottom = numpy.equal(phys_loc[:,1], pic.mesh.ymin)
        top = numpy.equal(phys_loc[:,1], pic.mesh.ymax)
        # Corners
        lb = numpy.logical_and(left, bottom) 
        lt = numpy.logical_and(left, top)
        rb = numpy.logical_and(right, bottom)
        rt = numpy.logical_and(right, top)
        # Margins
        corners = numpy.logical_or(lb, numpy.logical_or(lt, numpy.logical_or(rb, rt)))
        nc = numpy.logical_not(corners)
        l = numpy.logical_and(left, nc)
        r = numpy.logical_and(right, nc)
        t = numpy.logical_and(top, nc)
        b = numpy.logical_and(bottom,nc)
        # Creation of particles (using the short version of adding randomness)
        mpf_new = delta_n*pic.mesh.volumes[location]/species.spwt
        mpf_new = numpy.where(corners, mp_new*3)
        mp_new = mpf_new+numpy.random.rand(len(location))).astype(int)
        # Generate this many particles
        total_new = numpy.sum(mp_new)
        pos = numpy.zeros((total_new, species.pos_dim))
        vel = numpy.zeros((total_new, species.vel_dim))
        # Setting up velocities and positions
        c = 0
        for i in range(len(location)):
            vel[c:c+mp_new[i],:] += self.sampleIsotropicVelocity(n_vel[i], mp_new[i])+shift_vel[i,:]
            if phys_loc[i,0] == mesh.xmin:
                if phys_loc[i,1] == mesh.ymin:
                    pos[c:c+mp_new[i],0] += phys_loc[i,0] + numpy.random.rand(mp_new[i])*box_x[i]
            #pos[c:c+mp_new[i],0] += phys_loc[i,0] + (numpy.random.rand(mp_new[i])-0.5)*pic.mesh.dx
            pos[c:c+mp_new[i],1] += phys_loc[i,1] + (numpy.random.rand(mp_new[i])-0.5)*pic.mesh.dy
            c += mp_new[i]

# Here it comes, another version of create Dummy box, which was outdated after some changes were made to the class outer_2D_rectangular

            #       +createDummyBox([ind]location, PIC pic, Species species, [double] delta_n, [double] n_vel, [double] shift_vel) = create the dummy boxes with particles in them.
            # NOTE: I am not sure if addParticles is computationally demanding for other reason apart from the costs on numpy operations.
        def createDummyBox(self, location, pic, species, delta_n, n_vel, shift_vel):
            phys_loc = pic.mesh.getPosition(location)
            c = 0
            # Node by node
            for i in range(len(location)):
                # Amount of particles
                # Setting up position and velocities
                if phys_loc[i, 0] == self.xmin:
                    if phys_loc[i, 1] == self.ymin:
                        mpf_new = delta_n[i] * 4 * pic.mesh.volumes[location[i]] / species.spwt + \
                                  species.mesh_values.residuals[location[i]] + numpy.random.rand()
                        mp_new = mpf_new.astype(int)
                        species.mesh_values.residuals[location[i]] = mpf_new - mp_new
                        pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                        pos[:, 0] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dx
                        pos[:, 1] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dy
                        pos = numpy.delete(pos, numpy.flatnonzero(
                            numpy.logical_and(pos[:, 0] >= self.xmin, pos[:, 1] >= self.ymin)), axis=0)
                    elif phys_loc[i, 1] == self.ymax:
                        mpf_new = delta_n[i] * 4 * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                        mp_new = mpf_new.astype(int)
                        pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                        pos[:, 0] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dx
                        pos[:, 1] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dy
                        pos = numpy.delete(pos, numpy.flatnonzero(
                            numpy.logical_and(pos[:, 0] >= self.xmin, pos[:, 1] <= self.ymax)), axis=0)
                    else:
                        mpf_new = delta_n[i] * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                        mp_new = mpf_new.astype(int)
                        pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                        pos[:, 0] -= numpy.random.rand(mp_new) * pic.mesh.dx / 2
                        pos[:, 1] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dy
                elif phys_loc[i, 0] == self.xmax:
                    if phys_loc[i, 1] == self.ymin:
                        mpf_new = delta_n[i] * 4 * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                        mp_new = mpf_new.astype(int)
                        pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                        pos[:, 0] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dx
                        pos[:, 1] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dy
                        pos = numpy.delete(pos, numpy.flatnonzero(
                            numpy.logical_and(pos[:, 0] <= self.xmax, pos[:, 1] >= self.ymin)), axis=0)
                    elif phys_loc[i, 1] == self.ymax:
                        mpf_new = delta_n[i] * 4 * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                        mp_new = mpf_new.astype(int)
                        pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                        pos[:, 0] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dx
                        pos[:, 1] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dy
                        pos = numpy.delete(pos, numpy.flatnonzero(
                            numpy.logical_and(pos[:, 0] <= self.xmax, pos[:, 1] <= self.ymax)), axis=0)
                    else:
                        mpf_new = delta_n[i] * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                        mp_new = mpf_new.astype(int)
                        pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                        pos[:, 0] += numpy.random.rand(mp_new) * pic.mesh.dx / 2
                        pos[:, 1] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dy
                elif phys_loc[i, 1] == self.ymin:
                    mpf_new = delta_n[i] * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                    mp_new = mpf_new.astype(int)
                    pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                    pos[:, 0] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dx
                    pos[:, 1] -= numpy.random.rand(mp_new) * pic.mesh.dy / 2
                else:
                    mpf_new = delta_n[i] * pic.mesh.volumes[location[i]] / species.spwt + numpy.random.rand()
                    mp_new = mpf_new.astype(int)
                    pos = numpy.ones((mp_new, species.pos_dim)) * phys_loc[i].T
                    pos[:, 0] += (numpy.random.rand(mp_new) - 0.5) * pic.mesh.dx
                    pos[:, 1] += numpy.random.rand(mp_new) * pic.mesh.dy / 2

                vel = super().sampleIsotropicVelocity(numpy.asarray([n_vel[i]]), numpy.shape(pos)[0]) + shift_vel[i, :]
                self.addParticles(species, pos, vel)


#File that handles the classes for tracking particles
import numpy

#Tracker:
#
#Defintion = Class that initializes and stores the information for the tool of tracking particles after execution.
#Attributes:
#	+trackers ([Species_Tracker]) = List of instances of Species_Tracker.
#	+num_tracked (int) = Number of particles to be tracked.
#Methods:
#	+print() = Creates the files and print the information of the particles. Is basically the function to be called for output of information.
class Tracker(object):
    def __init__(self, num_tracked, *args):
        self.trackers = []
        self.num_tracked = numpy.uint8(num_tracked)
        for species in args:
            species_tracker = Species_Tracker(species, num_tracked)
            self.trackers.append(species_tracker)

    def print(self):
        # Checking tracking method
        for spc in self.trackers:
            if numpy.any(numpy.isnan(spc.indices)):
                raise ValueError("There should not be any nan values")


        narray = numpy.zeros((self.num_tracked, numpy.shape(self.trackers[0].position)[1]*len(self.trackers)))
        nHeader = ''
        for i in len(self.trackers):
            narray[:,2*i:2*(i+1)] = self.trackers[i].position[self.trackers[i].indices,:]
            nHeader += self.trackers[i].identifier + '\t'

        cwd = os.path.split(os.getcwd())[0]
        workfile = cwd+'/particle_tracker/ts={:05d}.dat'.format(ts)
        nHeader = 'No. of particles = {:d} \n'.format(self.num_tracked)+nHeader+'\n' 
        numpy.savetxt(workfile, narray , fmt = '%.5e', delimiter = '\t', header = nHeader)

#Species_Tracker (Composition with Tracker):
#
#Definition = Class that store the indices of the particles being tracked for a certain species.
#Attributes:
#	+identifier (String) = Same species.type
#	+indices (int) = array of size num_tracked that store the indices of the particles as stored in Particles class.
#       +position ([double,double]) = reference to the list of positions of the species being tracked.
class Species_Tracker(object):
    def __init__(self, species, num_tracked):
        self.identifier = species.type
        self.indices = numpy.nan*numpy.ones((num_tracked), dtype = numpy.uint32)
        self.position = species.part_values.position

# Functions created to reproduce the motion of a particle in constant E and constant B.
def x_vs_t(t, *args):
    return args[0]*numpy.cos(args[1]*t+args[2])-args[3]

def y_vs_t(t, *args):
    return args[0]*t+args[1]*numpy.sin(args[2]*t+args[3])-args[4]

#mean_with_error([double] values, [double] errors) double, double = Receives an array of values with its correspondant array of erros, and compute the mean with its error
def mean_with_error(values, errors):
    return numpy.average(values), numpy.linalg.norm(errors)/len(errors)



#An old version of derivatives by Pade on the outer_2D_rectangular boundary

#       +Pade derivation for the nodes in the boundaries. Normal 2nd order derivation when the boundary is perpendicular to the direction of the derivative.
#       +Arguments:
#       +location ([ind]) = location of the boundary nodes to be treated.
#       +mesh (Outer_2D_Rectangular) = mesh with the information to make the finite difference.
#       +potential ([double]) = scalar to be derivated.
#       +Return: [double, double] two-component derivation of potential, with every row being one node of location.
def derive_2D_rm_boundaries(location, mesh, potential):
    #Creating temporary field
    field = numpy.zeros((len(location),2))
    for ind in range(len(location)):
        #Handling corners
        if location[ind] == 0:
            field[ind,0] = (-3*potential[location[ind]]+4*potential[location[ind]+1]-potential[location[ind]+2])/(2*mesh.dx)
            field[ind,1] = (-3*potential[location[ind]]+4*potential[location[ind]+mesh.nx]-potential[location[ind]+2*mesh.nx])/(2*mesh.dy)
        elif location[ind] == mesh.nx:
            field[ind,0] = (3*potential[location[ind]]-4*potential[location[ind]-1]+potential[location[ind]-2])/(2*mesh.dx)
            field[ind,1] = (-3*potential[location[ind]]+4*potential[location[ind]+mesh.nx]-potential[location[ind]+2*mesh.nx])/(2*mesh.dy)
        elif location[ind] == mesh.nx*(mesh.ny-1):
            field[ind,0] = (-3*potential[location[ind]]+4*potential[location[ind]+1]-potential[location[ind]+2])/(2*mesh.dx)
            field[ind,1] = (3*potential[location[ind]]-4*potential[location[ind]-mesh.nx]+potential[location[ind]-2*mesh.nx])/(2*mesh.dy)
        elif location[ind] == mesh.nx*mesh.ny-1:
            field[ind,0] = (3*potential[location[ind]]-4*potential[location[ind]-1]+potential[location[ind]-2])/(2*mesh.dx)
            field[ind,1] = (3*potential[location[ind]]-4*potential[location[ind]-mesh.nx]+potential[location[ind]-2*mesh.nx])/(2*mesh.dy)
        #Handling non-corner borders
        elif location[ind] < mesh.nx:
            field[ind,0] = (potential[location[ind]+1]-potential[location[ind]-1])/(2*mesh.dx)
            field[ind,1] = (-3*potential[location[ind]]+4*potential[location[ind]+mesh.nx]-potential[location[ind]+2*mesh.nx])/(2*mesh.dy)
        elif location[ind]%mesh.nx == 0:
            field[ind,0] = (-3*potential[location[ind]]+4*potential[location[ind]+1]-potential[location[ind]+2])/(2*mesh.dx)
            field[ind,1] = (potential[location[ind]+mesh.nx]-potential[location[ind]-mesh.nx])/(2*mesh.dy)
        elif location[ind]%mesh.nx == mesh.nx-1:
            field[ind,0] = (3*potential[location[ind]]-4*potential[location[ind]-1]+potential[location[ind]-2])/(2*mesh.dx)
            field[ind,1] = (potential[location[ind]+mesh.nx]-potential[location[ind]-mesh.nx])/(2*mesh.dy)
        else:
            field[ind,0] = (potential[location[ind]+1]-potential[location[ind]-1])/(2*mesh.dx)
            field[ind,1] = (3*potential[location[ind]]-4*potential[location[ind]-mesh.nx]+potential[location[ind]-2*mesh.nx])/(2*mesh.dy)
    return field


#A new version of inner_2D_rectangular.py function applyParticleOpenBoundary is created to include the counting of particles going through
# the boundaries to account for the flux and accumulated charge. The version without this is included below.

    #       +applyParticleOpenBoundary(Species) = Deletes particles at or outside of the boundaries.
    def applyParticleOpenBoundary(self, species):
        #Just for convenience in writing
        np = species.part_values.current_n
        #Finding the particles
        ind = numpy.flatnonzero(numpy.logical_and(numpy.logical_and(numpy.logical_and(species.part_values.position[:np,0] >= self.xmin, \
                                                                                      species.part_values.position[:np,0] <= self.xmax), \
                                                                    species.part_values.position[:np,1] <= self.ymax), \
                                                  species.part_values.position[:np,1] >= self.ymin))

        # Eliminating particles
        super().removeParticles(species,ind)
        count2 = numpy.shape(ind)[0]
        print('Number of {} eliminated - inner:'.format(species.name), count2)

#A think the code below is wrong (really, incredibly wrong) so I will change it in the simulation. However, it might be that I am overlooking
# reasons of why the code bewlo is like that, so I want to maintain a copy of it.

#       +Pade derivation for the nodes in the boundaries. Normal 2nd order derivation when the boundary is perpendicular to the direction of the derivative.
#       +Arguments:
#       +location ([ind]) = location of the boundary nodes to be treated.
#       +mesh (Outer_2D_Rectangular) = mesh with the information to make the finite difference.
#       +potential ([double]) = scalar to be derivated.
#       +Return: [double, double] two-component derivation of potential, with every row being one node of location.
def derive_2D_rm_boundaries(potential, boundary, nx, ny, dx, dy):
    #Creating temporary field
    location = numpy.unique(boundary.location)
    tot = len(location)
    field = numpy.zeros((tot,2))
    #Creating markers and checking type of boundary
    b = numpy.isin(location, boundary.bottom)
    l = numpy.isin(location, boundary.left)
    r = numpy.isin(location, boundary.right)
    t = numpy.isin(location, boundary.top)
    inner = True if boundary.type.split(sep= "-")[0] == "Inner " else False
    outer = True if boundary.type.split(sep= "-")[0] == "Outer " else False
    #Derivative
    field[:,1] += numpy.where(numpy.logical_or(numpy.logical_and(outer,b),numpy.logical_and(inner,t)),\
                              (-3*potential[location]+4*potential[(location+nx)%tot]-potential[(location+2*nx)%tot])/(2*dy), 0)
    field[:,1] += numpy.where(numpy.logical_or(numpy.logical_and(outer,t),numpy.logical_and(inner,b)),\
                              (3*potential[location]-4*potential[(location-nx)%tot]+potential[(location-2*nx)%tot])/(2*dy), 0)
    field[:,1] += numpy.where(numpy.logical_not(numpy.logical_or(t,b)),(potential[(location+nx)%tot]-potential[(location-nx)%tot])/(2*dy), 0)
    field[:,0] += numpy.where(numpy.logical_or(numpy.logical_and(outer,l),numpy.logical_and(inner,r)),\
                              (-3*potential[location]+4*potential[(location+1)%tot]-potential[(location+2)%tot])/(2*dx), 0)
    field[:,0] += numpy.where(numpy.logical_or(numpy.logical_and(outer,r),numpy.logical_and(inner,l)),\
                              (3*potential[location]-4*potential[(location-1)%tot]+potential[(location-2)%tot])/(2*dx), 0)
    field[:,0] = numpy.where(numpy.logical_not(numpy.logical_or(l,r)),(potential[(location+1)%tot]-potential[(location-1)%tot])/(2*dx), 0)
    return field
