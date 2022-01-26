import copy
from functools import reduce
import numpy
import pdb

#Particle_reformer (Abstract class):
#
#Definition = Class that takes particles from a species, evaluates in each cell whether there are too many or too few particles, and 
#             then eliminates and create new particles to adjust this number while maintaining the same statistical results.
#Attributes:
#       +mesh (Mesh) = root PIC-mesh of the domain.
#       +partmin (int) = Minimum amount of particles allowed per cell.
#       +partmax (int) = Maximum amount of particles allowed per cell.
#       +minfraction (int) = minimum fraction of the nominal SPWT of a species allowed in a single particle.
#Methods:
#       +sortByCell(Species species): [int] particles, [[int]] indices = The function receives a Species and calculates for each node of the mesh, the amount of particles located in the cell 
#           represented by the node. If 'indices = True' the function also returns a list of lists, where for each node the indices of the particles contained in the respective cell are stored.
#       +redistributeSPWTs(Species species, [[int]] indices): [[int]] SPWTs = This function receives a species, and a list of lists indicating, per cell, the indices of the particles that
#           are inside. Then, it counts the amount of real particles (multipling by SPWT of each particle) and computes, per cell, an array of length 'partoptimum' with SPWTs more or less
#           equal such that the number of real particles in the cell remains the same or very similar.
#       +computeParticleReform(Species species) = This method performs the particles processing. First, by running 'sortByCell' identifies which cells need to be reformed, 
#           then it selects those cells, eliminate the particles involved, and adds the new particles to Species.
#       +saveVTK(Mesh mesh): dictionary = Return the number of particles in each cell for all the species, in order to be printed in the VTK file.
class Particle_reformer(object):
    def __init__(self, n_mesh, n_partmin, n_partmax, n_minfraction, n_vtk):
        self.mesh = n_mesh
        self.partmin = n_partmin
        self.partmax =  n_partmax
        self.minfraction = n_minfraction
        self.vtk = n_vtk
        if self.vtk == True:
            self.species = {}

    def sortByCell(self, pos, pos_ind, indices = True):
        #Getting mesh coordinates
        mc = self.mesh.getIndex(pos)
        
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        
        field = numpy.zeros((self.mesh.nPoints))
        numpy.add.at(field, array, 1)

        if indices == True:
            lists = [[] for i in range(len(field))]
            #NOTE: This line could be changed for indices = mask for further speed-up in the recursion case, but does not work for non-recursive case.
            for node, part in zip(array, pos_ind):
                lists[node].append(int(part))
            return field, lists
        return field

    def redistributeSPWTs(self, species, indices):
        pass

    def computeParticleReform(self, species):
        pass

    def saveVTK(self, mesh):
        temp = {}
        for key, value in self.species.items():
            temp[key+'_SPs'] = mesh.vtkOrdering(value)
        return temp


#Particle_reformer_recursive (Abstract class):
#
#Definition = Class that enforces the recursiveness of 'Particle_Reformer' classes
#Attributes:
#	+root (Boolean) = Identifies whether the object is the root of the tree or not.
#       +children ([Mesh]) = List of all the children of the Particle_Reformer instance.
#Methods:
#       +None.
class Particle_reformer_recursive(object):
    def __init__(self, n_children, n_root, n_type):
        n_type += " - Recursive"
        self.root = n_root
        self.children = n_children


#Particle_reformer_uniform (Particle_reformer):
#
#Definition = This Particle reformer creates all the particles in a cell with the same SPWT, except by the first particle, which also takes the leftover real particles.
#Attributes:
#       +Particle_reformer attributes.
#Methods:
#       +Particle_reformer methods.
class Particle_reformer_uniform(Particle_reformer):
    type = "Particle reformer uniform"
    def __init__(self, n_mesh, n_partmin, n_partmax, n_partoptimum, n_minfraction, n_vtk):
        super().__init__(n_mesh, n_partmin, n_partmax, n_minfraction, n_vtk)
        self.partoptimum = n_partoptimum

    def redistributeSPWTs(self, species, parts, indices):
        real_part = [numpy.sum(species.part_values.spwt[parts[i]]) for i in indices]
        spwts_c, remainder = numpy.divmod(real_part, self.partoptimum)
        spwts = numpy.ones((len(spwts_c)*self.partoptimum))
        iterator = numpy.arange(len(spwts_c), dtype = int)
        for i in iterator:
            spwts[i*self.partoptimum:(i+1)*self.partoptimum] *= spwts_c[i]
        spwts[iterator*self.partoptimum] += remainder[iterator]
        return spwts

    def computeParticleReform(self, species):
        #Checking particles in each cell
        count, parts = self.sortByCell(species.part_values.position[:species.part_values.current_n], numpy.arange(species.part_values.current_n, dtype = numpy.uint64))
        #Selecting cells to process
        indices = numpy.arange(self.mesh.nPoints)
        filter_1 = self.mesh.particleReformNodes()
        filter_2 = count > self.partmax
        filter_3 = numpy.logical_and(count < self.partmin, count > 5)
        indices = indices[numpy.logical_and(numpy.logical_or(filter_2, filter_3), filter_1)]
        spwts = self.redistributeSPWTs(species, parts, indices)
        filter_4 = []
        filter_4_ = numpy.zeros((0))
        for i in range(len(indices)):
            if spwts[i*self.partoptimum+1] < species.spwt*self.minfraction:
                filter_4.append(i)
                filter_4_ = numpy.append(filter_4_, numpy.arange(i*self.partoptimum, (i+1)*self.partoptimum))
        indices = numpy.delete(indices, filter_4)
        spwts = numpy.delete(spwts, filter_4_)
        #Constructing the new particles and eliminating old ones
        num = self.partoptimum*numpy.ones_like(indices)
        new_pos = self.mesh.createDistributionInCell(indices, num)
        new_vel = self.mesh.boundaries[0].sampleIsotropicVelocity(self.boundaries[0].thermalVelocity(species.mesh_values.temperature[indices], species.m), num)
        del_ind = []
        for i in range(len(indices)):
            new_vel[i*self.partoptimum:(i+1)*self.partoptimum,:] += species.mesh_values.velocity[indices[i],:]
            del_ind.extend(parts[indices[i]])
        self.mesh.boundaries[0].removeParticles(species, numpy.asarray(del_ind))
        self.mesh.boundaries[0].addParticles(species, new_pos, new_vel, spwt = spwts)
        print("Mesh reformed. Species: {}".format(species.name))
        print("{:d} indices were affected, {:d} particles were added, {:d} particles were eliminated".format(len(indices), len(spwts[0]), len(del_ind[0])))
        if self.vtk == True:
            self.species[species.name] = self.sortByCell(species.part_values.position[:species.part_values.current_n],\
                    numpy.arange(species.part_values.current_n, dtype = numpy.uint64), indices = False)


#Particle_reformer_uniform_recursive (Particle_reformer_recursive, Particle_reformer_uniform):
#
#Definition = This Particle reformer creates all the particles in a cell with the same SPWT, except by the first particle, which also takes the leftover real particles.
#Attributes:
#       +Particle_reformer attributes.
#Methods:
#       +Particle_reformer methods.
class Particle_reformer_uniform_recursive(Particle_reformer_recursive, Particle_reformer_uniform):
    def __init__(self, n_mesh, n_partmin, n_partmax, n_partoptimum, n_minfraction, n_vtk, n_children, n_root):
        super(Particle_reformer_recursive, self).__init__(n_mesh, n_partmin, n_partmax, n_partoptimum, n_minfraction, n_vtk)
        super().__init__(n_children, n_root, self.type)

    def computeParticleReform(self, species, positions = None, acc_ind = None, pos_ind = None, new_positions = None, new_velocities = None, new_spwts = None, del_indices = None):
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(species.part_values.position[:species.part_values.current_n], return_ind = [], surface = True)
            acc_ind = [0]
            new_positions = [numpy.zeros((0,species.pos_dim))]
            new_velocities = [numpy.zeros((0,species.vel_dim))]
            new_spwts = [numpy.zeros((0))]
            del_indices = [numpy.zeros((0), dtype = numpy.int)]
            #new_positions = numpy.zeros((0,species.pos_dim))
            #new_velocities = numpy.zeros((0,species.vel_dim))
            #new_spwts = numpy.zeros((0))
            #del_indices = numpy.zeros((0), dtype = numpy.int)
        pos = positions.pop(0)
        part = pos_ind.pop(0)
        start_ind = acc_ind[0]
        acc_ind[0] += self.mesh.nPoints
        for child in self.children:
            child.computeParticleReform(species, positions = positions, acc_ind = acc_ind, pos_ind = pos_ind,\
                    new_positions = new_positions, new_velocities = new_velocities, new_spwts = new_spwts, del_indices = del_indices)
        count, parts = self.sortByCell(pos, part)
        #Selecting cells to process
        indices = numpy.arange(self.mesh.nPoints)
        filter_1 = self.mesh.particleReformNodes()
        filter_2 = count > self.partmax
        #filter_species
        if (species.name == "Electron - Photoelectron" or species.name == "Electron - SEE") and self.mesh.id == '0-0-0':
            filter_2[:] = False
        filter_3 = numpy.logical_and(count < self.partmin, count > 5)
        #filter_3 = count < self.partmin
        indices = indices[numpy.logical_and(numpy.logical_or(filter_2, filter_3), filter_1)]
        spwts = self.redistributeSPWTs(species, parts, indices)
        filter_4 = []
        filter_4_ = numpy.zeros((0), dtype = numpy.uint64)
        #Filtering to avoid infinite division
        for i in range(len(indices)):
            if spwts[i*self.partoptimum+1] < species.spwt*self.minfraction:
                filter_4.append(i)
                filter_4_ = numpy.append(filter_4_, numpy.arange(i*self.partoptimum, (i+1)*self.partoptimum, dtype = numpy.uint64))
        if species.name == "Electron - Solar wind":
            pdb.set_trace()
        #Adding the parameters necessary for eliminating and adding particles
        indices = numpy.delete(indices, filter_4)
        spwts = numpy.delete(spwts, filter_4_)
        new_spwts[0] = numpy.append(new_spwts[0], spwts)
        num = self.partoptimum*numpy.ones_like(indices)
        new_positions[0] = numpy.append(new_positions[0], self.mesh.createDistributionInCell(indices, num), axis = 0)
        new_vel = self.mesh.boundaries[0].sampleIsotropicVelocity(self.mesh.boundaries[0].thermalVelocity(species.mesh_values.temperature[start_ind+indices], species.m), num)
        for i in range(len(indices)):
            new_vel[i*self.partoptimum:(i+1)*self.partoptimum,:] += species.mesh_values.velocity[start_ind+indices[i],:]
            del_indices[0] = numpy.append(del_indices[0], numpy.asarray(parts[indices[i]]))
        if species.name == "Electron - Solar wind":
            pdb.set_trace()
        new_velocities[0] = numpy.append(new_velocities[0], new_vel, axis = 0)
        #Constructing the new particles and eliminating old ones
        if self.root == True:
            self.mesh.boundaries[0].removeParticles(species, del_indices[0])
            self.mesh.boundaries[0].addParticles(species, new_positions[0], new_velocities[0], spwt = new_spwts[0])
            print("Mesh reformed. Species: {}".format(species.name))
            print("{:d} indices were affected, {:d} particles were added, {:d} particles were eliminated".format(len(indices), len(new_spwts[0]), len(del_indices[0])))
            if self.vtk == True:
                acc_ind = [0]
                part_per_node = numpy.zeros((self.mesh.accPoints))
                positions, pos_ind = self.mesh.sortPositionsByMeshes(species.part_values.position[:species.part_values.current_n], return_ind = [], surface = False)
                self.vtk_recursion(positions, part_per_node, acc_ind)
                self.species[species.name] = part_per_node

    def vtk_recursion(self, positions, field, acc_ind):
        temp = acc_ind[0]
        acc_ind[0] += self.mesh.nPoints
        pos = positions.pop(0)
        field[temp:acc_ind[0]] = self.sortByCell(pos, None, indices = False)
        for child in self.children:
            child.vtk_recursion(positions, field, acc_ind)


#Particle_reformer_particle (Particle_reformer):
#
#Definition = This Particle reformer creates all the new particles as divisions of the old particles, dividing each of the parents' SPWTs among the new particles.
#Attributes:
#       +Particle_reformer attributes.
#       +partoptimum(int) = Ideal amount of particles per cell.
#Methods:
#       +Particle_reformer methods.
class Particle_reformer_particle(Particle_reformer):
    type = "Particle reformer per particle"
    def __init__(self, n_mesh, n_partmin, n_partmax, n_partoptimum, n_minfraction, n_vtk):
        super().__init__(n_mesh, n_partmin, n_partmax, n_minfraction, n_vtk)
        self.partoptimum = n_partoptimum


##       +redistributeSPWTs(Species species, [[int]] indices): [[int]] SPWTs = This function receives a species, and a list of lists indicating, per cell, the indices of the particles that
##           are inside. Then, it counts the amount of real particles per cell and computes the SPWTs of the new particles.
##           The function returns new_spwts, an array the size of the total amount of new particles, with the SPWT of each, and it returns,
##           part_per_part, the number of new particles created per old particle, ordered per cell and particle per cell.
##           +Parameters: 
#    def redistributeSPWTs(self, species, parts, indices):
#        part_per_cell = numpy.asarray([len(parts[i]) for i in indices]).astype(numpy.int)
#        part_per_part_cell = self.partoptimum/part_per_cell
#        fewer = numpy.flatnonzero(part_per_part_cell < 1)
#        more = numpy.flatnonzero(part_per_part_cell > 1)
#        new_spwts = numpy.zeros((0))
#        part_per_part = numpy.zeros((0))
#
#        #Merging of particles
#        ind_sel = indices[fewer]
#        part_per_part_cell_fewer = numpy.ceil(1/part_per_part_cell[fewer]).astype(numpy.uint8)
#        for i in range(len(ind_sel)):
#            spwts_c = numpy.zeros((self.partoptimum))
#            part_c = numpy.zeros((self.partoptimum))
#            for j in range(self.partoptimum):
#                spwts_c[j] = numpy.sum(species.part_values.spwt[parts[ind_sel[i]]][j::self.partoptimum])
#                part_c[j] = len(parts[ind_sel[i]][j::self.partoptimum])
#            new_spwts = numpy.append(new_spwts, spwts_c)
#            part_per_part = numpy.append(part_per_part, part_c)
##        for i in range(len(ind_sel)):
##            ind = numpy.arange(0, part_per_cell[fewer][i], part_per_part_cell_fewer[i])
##            spwts_c = numpy.zeros((len(ind)))
##            part_c = numpy.zeros((len(ind)))
##            for j in range(len(ind)-1):
##                spwts_c[j] = numpy.sum(species.part_values.spwt[parts[ind_sel[i]]][ind[j]:ind[j]+part_per_part_cell_fewer[i]])
##            part_c[:-1] = part_per_part_cell_fewer[i]
##            spwts_c[-1] = numpy.sum(species.part_values.spwt[parts[ind_sel[i]]][ind[-1]:])
##            part_c[-1] = part_per_cell[fewer][i]-ind[-1]
##            new_spwts = numpy.append(new_spwts, spwts_c)
##            part_per_part = numpy.append(part_per_part, part_c)
#
#        #Splitting of particles
#        ind_sel = indices[more]
#        part_per_part_cell_more = numpy.ceil(part_per_part_cell[more]).astype(numpy.uint8)
#        part_per_part = numpy.append(part_per_part, numpy.repeat(part_per_part_cell_more, part_per_cell[more]))
#        for i in range(len(ind_sel)):
#            spwts_c, remainder = numpy.divmod(species.part_values.spwt[parts[ind_sel[i]]], part_per_part_cell_more[i])
#            new_spwts_per_cell = numpy.repeat(spwts_c, part_per_part_cell_more[i])
#            new_spwts_per_cell[0::part_per_part_cell_more[i]] += remainder
#            new_spwts = numpy.append(new_spwts, new_spwts_per_cell)
#
#        #Returning both
#        return new_spwts, part_per_part.astype(int)


#       +redistributeSPWTs(Species species, [[int]] indices): [[int]] SPWTs = This function receives a species, and a list of lists indicating, per cell, the indices of the particles that
#           are inside. Then, it counts the amount of real particles per cell and computes the SPWTs of the new particles.
#           The function returns new_spwts, an array the size of the total amount of new particles, with the SPWT of each, and it returns,
#           part_per_part, the number of new particles created per old particle, ordered per cell and particle per cell.
#           +Parameters: 
#           +direction = None. direction can be 'more' or 'fewer'. If 'more' it will redistribute SPWT as if more particles are needed. If 'fewer' it will redistribute as if less particles
#               are needed.
    def redistributeSPWTs(self, species, parts, indices, direction = None):
        part_per_cell = numpy.asarray([len(parts[i]) for i in indices]).astype(numpy.int)
        part_per_part_cell = self.partoptimum/part_per_cell
        new_spwts = numpy.zeros((0))
        part_per_part = numpy.zeros((0))

        #Merging of particles
        if direction == 'fewer':
            part_per_part_cell_fewer = numpy.ceil(1/part_per_part_cell).astype(numpy.uint8)
            for i in range(len(indices)):
                spwts_c = numpy.zeros((self.partoptimum))
                part_c = numpy.zeros((self.partoptimum))
                for j in range(self.partoptimum):
                    spwts_c[j] = numpy.sum(species.part_values.spwt[parts[indices[i]]][j::self.partoptimum])
                    part_c[j] = len(parts[indices[i]][j::self.partoptimum])
                new_spwts = numpy.append(new_spwts, spwts_c)
                part_per_part = numpy.append(part_per_part, part_c)

        #Splitting of particles
        elif direction == 'more':
            part_per_part_cell_more = numpy.ceil(part_per_part_cell).astype(numpy.uint8)
            part_per_part = numpy.append(part_per_part, numpy.repeat(part_per_part_cell_more, part_per_cell))
            for i in range(len(indices)):
                spwts_c, remainder = numpy.divmod(species.part_values.spwt[parts[indices[i]]], part_per_part_cell_more[i])
                new_spwts_per_cell = numpy.repeat(spwts_c, part_per_part_cell_more[i])
                new_spwts_per_cell[0::part_per_part_cell_more[i]] += remainder
                new_spwts = numpy.append(new_spwts, new_spwts_per_cell)

        else:
            raise Error ("more or fewer should be passed as value of direction kwarg")

        #Returning
        return new_spwts, part_per_part.astype(int)

#NOTE: This method has to include the change of RedistributeSPWT made on 2022_01_10.
    def computeParticleReform(self, species):
        #Checking particles in each cell
        count, parts = self.sortByCell(species.part_values.position[:species.part_values.current_n], numpy.arange(species.part_values.current_n, dtype = numpy.uint64))
        #Selecting cells to process
        indices = numpy.arange(self.mesh.nPoints)
        filter_1 = self.mesh.particleReformNodes()
        filter_2 = count > self.partmax
        filter_3 = numpy.logical_and(count < self.partmin, count > 3)
        indices_f = indices[numpy.logical_and(filter_1, filter_2)]
        indices_m = indices[numpy.logical_and(filter_1, filter_3)]
        #Creation of new particles - more
        filter_4 = []
        part_ind_m = []
        for i in range(len(indices_m)):
            if numpy.any(species.part_values.spwt[parts[indices_m[i]]] < species.spwt*self.minfraction):
                filter_4.append(i)
            else:
                part_ind_m.extend(parts[indices_m[i]])
        indices_m = numpy.delete(indices_m, filter_4)
        new_spwt_m, part_per_part_m = self.redistributeSPWTs(species, parts, indices_m)
        new_pos_m = numpy.repeat(species.part_values.position[part_ind_m,:], part_per_part_m, axis = 0)
        new_vel_m = numpy.repeat(species.part_values.velocity[part_ind_m,:], part_per_part_m, axis = 0)
        #Adding a random 0.1*|vel| isotropic velocity so that the particles split in time
        speed = numpy.linalg.norm(species.part_values.velocity[part_ind_m,:], axis = 1)
        new_speeds = numpy.repeat(speed, part_per_part_m)*0.1
        angle = 2*numpy.pi*numpy.random.rand(len(new_spwt_m))
        new_vel_m[:,0] += numpy.cos(angle)*new_speeds
        new_vel_m[:,1] += numpy.sin(angle)*new_speeds
        #Creation of new particles - fewer
        part_ind_f = numpy.zeros((0), dtype = numpy.int)
        part_ind_f = reduce(lambda acc, ind: numpy.append(acc, parts[ind]), indices_f, part_ind_f)
        new_spwt_f, part_per_part_f = self.redistributeSPWTs(species, parts, indices_f)
        new_pos_f = numpy.zeros((len(new_spwt_f), species.pos_dim))
        new_vel_f = numpy.zeros((len(new_spwt_f), species.vel_dim))
        base_ind = numpy.zeros_like(new_spwt_f, dtype = numpy.uint)
        base_ind[1:] = numpy.cumsum(part_per_part_f[0:-1], dtype = numpy.uint)
        for i in range(len(base_ind)):
            new_pos_f[i,:] = numpy.average(species.part_values.position[base_ind[i]:base_ind[i]+part_per_part_f[i],:], weights = species.part_values.spwt[base_ind[i]:base_ind[i]+part_per_part_f[i]], axis = 0)
            new_vel_f[i,:] = numpy.average(species.part_values.velocity[base_ind[i]:base_ind[i]+part_per_part_f[i],:], weights = species.part_values.spwt[base_ind[i]:base_ind[i]+part_per_part_f[i]], axis = 0)
        #Adding and removing particles
        self.mesh.boundaries[0].removeParticles(species, numpy.append(part_ind_m, part_ind_f))
        self.mesh.boundaries[0].addParticles(species, numpy.append(new_pos_m, new_pos_f, axis = 0),\
                numpy.append(new_vel_m, new_vel_f, axis = 0), spwt = numpy.append(new_spwt_m, new_spwt_f))
        print("Mesh reformed. Species: {}".format(species.name))
        print("{:d} indices were affected, {:d} particles were added, {:d} particles were eliminated".format(len(indiices_m)+len(indices_f), len(new_spwt_m)+len(new_spwt_f), len(part_ind_m)+len(part_ind_f)))
        if self.vtk == True:
            self.species[species.name] = self.sortByCell(species.part_values.position[:species.part_values.current_n],\
                    numpy.arange(species.part_values.current_n, dtype = numpy.uint64), indices = False)


#Particle_reformer_particle_recursive (Particle_reformer_recursive, Particle_reformer_particle):
#
#Definition = This Particle reformer creates all the new particles as divisions of the old particles, dividing each of the parents' SPWTs among the new particles. Recursive version.
#Attributes:
#       +Particle_reformer_particle attributes.
#       +Particle_reformer_recursive attributes.
#Methods:
#       +Particle_reformer methods.
class Particle_reformer_particle_recursive(Particle_reformer_recursive, Particle_reformer_particle):
    def __init__(self, n_mesh, n_partmin, n_partmax, n_partoptimum, n_minfraction, n_vtk, n_children, n_root):
        super(Particle_reformer_recursive, self).__init__(n_mesh, n_partmin, n_partmax, n_partoptimum, n_minfraction, n_vtk)
        super().__init__(n_children, n_root, self.type)

    def computeParticleReform(self, species, positions = None, acc_ind = None, pos_ind = None, new_positions = None, new_velocities = None, new_spwts = None, del_indices = None):
        #Tree structure section
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(species.part_values.position[:species.part_values.current_n], return_ind = [], surface = True)
            acc_ind = [0]
            new_positions = [numpy.zeros((0,species.pos_dim))]
            new_velocities = [numpy.zeros((0,species.vel_dim))]
            new_spwts = [numpy.zeros((0))]
            del_indices = [numpy.zeros((0), dtype = numpy.int)]
        pos = positions.pop(0)
        part = pos_ind.pop(0)
        acc_ind[0] += self.mesh.nPoints
        for child in self.children:
            child.computeParticleReform(species, positions = positions, acc_ind = acc_ind, pos_ind = pos_ind,\
                    new_positions = new_positions, new_velocities = new_velocities, new_spwts = new_spwts, del_indices = del_indices)
        #Splitting particles with high SPWT
        new_spwt_h, new_pos_h, new_vel_h, part_ind_h, new_pos, new_part = self.partitionHighSPWTParticles(species, pos, part)
        #Checking particles in each cell
        count, parts = self.sortByCell(new_pos, new_part)
        #Selecting cells to process
        indices = numpy.arange(self.mesh.nPoints)
        filter_1 = self.mesh.particleReformNodes()
        filter_2 = count > self.partmax
        #filter_species
        if (species.name == "Electron - Photoelectron" or species.name == "Electron - SEE") and self.mesh.id == '0-0-0':
            filter_2[:] = False
        filter_3 = numpy.logical_and(count < self.partmin, count > 3)
        indices_f = indices[numpy.logical_and(filter_1, filter_2)]
        indices_m = indices[numpy.logical_and(filter_1, filter_3)]

        #Creation of new particles - more
        filter_4 = []
        part_ind_m = []
        for i in range(len(indices_m)):
            if numpy.any(species.part_values.spwt[parts[indices_m[i]]] < species.spwt*self.minfraction):
                filter_4.append(i)
            else:
                part_ind_m.extend(parts[indices_m[i]])
        indices_m = numpy.delete(indices_m, filter_4)
        new_spwt_m, new_pos_m, new_vel_m = self.createMoreParticles(species, parts, indices_m, part_ind_m)

        #Creation of new particles - fewer
        part_ind_f = numpy.zeros((0), dtype = numpy.int)
        part_ind_f = reduce(lambda acc, ind: numpy.append(acc, parts[ind]), indices_f, part_ind_f)
        new_spwt_f, new_pos_f, new_vel_f = self.createFewerParticles(species, parts, indices_f, part_ind_f)

        #Adding to variables in recursion
        #new_spwts[0] = numpy.append(new_spwts[0], numpy.append(numpy.append(new_spwt_m, new_spwt_f), new_spwt_h))
        #new_positions[0] = numpy.append(new_positions[0], numpy.append(numpy.append(new_pos_m, new_pos_f, axis = 0), new_pos_h, axis = 0), axis = 0)
        #new_velocities[0] = numpy.append(new_velocities[0], numpy.append(numpy.append(new_vel_m, new_vel_f, axis = 0), new_vel_h, axis = 0), axis = 0)
        #del_indices[0] = numpy.append(del_indices[0], numpy.append(numpy.append(part_ind_m, part_ind_f), part_ind_h).astype(numpy.int))
        new_spwts[0] = numpy.append(new_spwts[0], numpy.append(new_spwt_m, new_spwt_f))
        new_positions[0] = numpy.append(new_positions[0], numpy.append(new_pos_m, new_pos_f, axis = 0), axis = 0)
        new_velocities[0] = numpy.append(new_velocities[0], numpy.append(new_vel_m, new_vel_f, axis = 0), axis = 0)
        del_indices[0] = numpy.append(del_indices[0], numpy.append(part_ind_m, part_ind_f).astype(numpy.int))
        #Constructing the new particles and eliminating old ones
        if self.root == True:
            self.mesh.boundaries[0].removeParticles(species, del_indices[0])
            self.mesh.boundaries[0].addParticles(species, new_positions[0], new_velocities[0], spwt = new_spwts[0])
            print("Mesh reformed. Species: {}".format(species.name))
            print("{:d} indices were affected, {:d} particles were added, {:d} particles were eliminated".format(len(indices), len(new_spwts[0]), len(del_indices[0])))
            if self.vtk == True:
                acc_ind = [0]
                part_per_node = numpy.zeros((self.mesh.accPoints))
                positions, pos_ind = self.mesh.sortPositionsByMeshes(species.part_values.position[:species.part_values.current_n], return_ind = [], surface = False)
                self.vtk_recursion(positions, part_per_node, acc_ind)
                self.species[species.name] = part_per_node
            self.computeParticleReform_HighSPWTParticles(species)

    def computeParticleReform_HighSPWTParticles(self, species, positions = None, acc_ind = None, pos_ind = None, new_positions = None, new_velocities = None, new_spwts = None, del_indices = None):
        #Tree structure section
        if self.root == True:
            positions, pos_ind = self.mesh.sortPositionsByMeshes(species.part_values.position[:species.part_values.current_n], return_ind = [], surface = True)
            acc_ind = [0]
            new_positions = [numpy.zeros((0,species.pos_dim))]
            new_velocities = [numpy.zeros((0,species.vel_dim))]
            new_spwts = [numpy.zeros((0))]
            del_indices = [numpy.zeros((0), dtype = numpy.int)]
        pos = positions.pop(0)
        part = pos_ind.pop(0)
        acc_ind[0] += self.mesh.nPoints
        for child in self.children:
            child.computeParticleReform_HighSPWTParticles(species, positions = positions, acc_ind = acc_ind, pos_ind = pos_ind,\
                    new_positions = new_positions, new_velocities = new_velocities, new_spwts = new_spwts, del_indices = del_indices)
        #Splitting particles with high SPWT
        #Filtering particles close to borders
        mc = self.mesh.getIndex(pos)
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        filter_1 = self.mesh.particleReformNodes()
        filter_part = numpy.isin(array, filter_1)

        new_spwt_h, new_pos_h, new_vel_h, part_ind_h, new_pos, new_part = self.partitionHighSPWTParticles(species, pos[filter_part,:], part[filter_part])

        #Adding to variables in recursion
        new_spwts[0] = numpy.append(new_spwts[0], new_spwt_h)
        new_positions[0] = numpy.append(new_positions[0],  new_pos_h, axis = 0)
        new_velocities[0] = numpy.append(new_velocities[0], new_vel_h, axis = 0)
        del_indices[0] = numpy.append(del_indices[0], part_ind_h.astype(numpy.int))
        #Constructing the new particles and eliminating old ones
        if self.root == True:
            self.mesh.boundaries[0].removeParticles(species, del_indices[0])
            self.mesh.boundaries[0].addParticles(species, new_positions[0], new_velocities[0], spwt = new_spwts[0])
            print("Mesh reformed. Species: {}".format(species.name))
            print("{:d} particles were added, {:d} particles were eliminated".format(len(new_spwts[0]), len(del_indices[0])))
            if self.vtk == True:
                acc_ind = [0]
                part_per_node = numpy.zeros((self.mesh.accPoints))
                positions, pos_ind = self.mesh.sortPositionsByMeshes(species.part_values.position[:species.part_values.current_n], return_ind = [], surface = False)
                self.vtk_recursion(positions, part_per_node, acc_ind)
                self.species[species.name] = part_per_node

    def partitionHighSPWTParticles(self, species, pos, part, threshold = int(5)):
        #Selecting which particles to partition
        avg = numpy.mean(species.part_values.spwt[part])
        ind = numpy.flatnonzero(species.part_values.spwt[part] > threshold*avg)
        #Taking the partitioned particles out of the list of particles
        new_pos = numpy.delete(pos, ind, axis = 0)
        new_part = numpy.delete(part, ind)
        part_ind_h = part[ind]
        #Preparing the new particles
        new_spwt_h, remainder = numpy.divmod(species.part_values.spwt[part_ind_h], threshold)
        new_spwt_h = numpy.repeat(new_spwt_h, threshold)
        new_spwt_h[0::threshold] += remainder
        #New positions and velocities
        new_pos_h = numpy.repeat(species.part_values.position[part_ind_h,:], threshold, axis = 0)
        new_vel_h = numpy.repeat(species.part_values.velocity[part_ind_h,:], threshold, axis = 0)
        #Adding a random 0.05*|vel| isotropic velocity so that the particles separate over time, and distributing them in space too
        speed = numpy.linalg.norm(species.part_values.velocity[part_ind_h,:], axis = 1)
        new_speeds = numpy.repeat(speed, threshold)*0.1
        randg = numpy.random.rand(len(new_spwt_h),3)
        angle = 2*numpy.pi*randg[:,0]
        new_vel_h[:,0] += numpy.cos(angle)*new_speeds
        new_vel_h[:,1] += numpy.sin(angle)*new_speeds
        #NOTE: This is not mesh independent
        new_pos_h[:,0] += (randg[:,1]*self.mesh.dx-self.mesh.dx*0.5)
        new_pos_h[:,1] += (randg[:,2]*self.mesh.dy-self.mesh.dy*0.5)
        #Return new values
        return new_spwt_h, new_pos_h, new_vel_h, part_ind_h, new_pos, new_part


    def createMoreParticles(self, species, parts, indices_m, part_ind_m):
        #New SPWTs
        new_spwt_m, part_per_part_m = self.redistributeSPWTs(species, parts, indices_m, direction = 'more')
        #New positions and velocities
        new_pos_m = numpy.repeat(species.part_values.position[part_ind_m,:], part_per_part_m, axis = 0)
        new_vel_m = numpy.repeat(species.part_values.velocity[part_ind_m,:], part_per_part_m, axis = 0)
        #Adding a random 0.05*|vel| isotropic velocity so that the particles separate over time, and distributing them in space too
        speed = numpy.linalg.norm(species.part_values.velocity[part_ind_m,:], axis = 1)
        new_speeds = numpy.repeat(speed, part_per_part_m)*0.1
        randg = numpy.random.rand(len(new_spwt_m),3)
        angle = 2*numpy.pi*randg[:,0]
        new_vel_m[:,0] += numpy.cos(angle)*new_speeds
        new_vel_m[:,1] += numpy.sin(angle)*new_speeds
        #NOTE: This is not mesh independent
        new_pos_m[:,0] += (randg[:,1]*self.mesh.dx-self.mesh.dx*0.5)
        new_pos_m[:,1] += (randg[:,2]*self.mesh.dy-self.mesh.dy*0.5)
        #Return new values
        return new_spwt_m, new_pos_m, new_vel_m

    def createFewerParticles(self, species, parts, indices_f, part_ind_f):
        #New SPWTs
        new_spwt_f, part_per_part_f = self.redistributeSPWTs(species, parts, indices_f, direction = 'fewer')
        #New positions and velocities
        new_pos_f = numpy.zeros((len(new_spwt_f), species.pos_dim))
        new_vel_f = numpy.zeros((len(new_spwt_f), species.vel_dim))
        for i in range(len(indices_f)):
            for j in range(self.partoptimum):
                new_pos_f[i*self.partoptimum+j,:] = numpy.average(species.part_values.position[parts[indices_f[i]][j::self.partoptimum],:],\
                        weights = species.part_values.spwt[parts[indices_f[i]][j::self.partoptimum]], axis = 0)
                new_vel_f[i*self.partoptimum+j,:] = numpy.average(species.part_values.velocity[parts[indices_f[i]][j::self.partoptimum],:],\
                        weights = species.part_values.spwt[parts[indices_f[i]][j::self.partoptimum]], axis = 0)
        #Return new vales
        return new_spwt_f, new_pos_f, new_vel_f

    def vtk_recursion(self, positions, field, acc_ind):
        temp = acc_ind[0]
        acc_ind[0] += self.mesh.nPoints
        pos = positions.pop(0)
        field[temp:acc_ind[0]] = self.sortByCell(pos, None, indices = False)
        for child in self.children:
            child.vtk_recursion(positions, field, acc_ind)
