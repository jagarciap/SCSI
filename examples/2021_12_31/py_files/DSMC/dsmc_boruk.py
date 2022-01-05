#Data structure of the Boundaries
import constants as c
import copy
#Delete later
import matplotlib.pyplot as plt
import numpy

from Species.species import Species
from timing import Timing

#DSMC_Boruk (Abstract-like: will also have common methods that can be used by sub-classes):
#
#Definition = Class that contains the essential methods and procedures for appling the Boruk DSMC method.
# Source: 3-D Simulation of Ion Thruster Plumes Using Octree Adaptive Mesh Refinement (Burak Korkut, Zheng Li, and Deborah A. Levin)
#Attributes:
#       +type (string) = Type of collision represented. The name pattern is: "MEX/CEX-species_in[0],...,species_in[i]->species_out[0],...,species_out[j]".
#	+species_in ([Species]) = List containing references to the species used as input for the reaction. I am using the protocol of species_in[0] is the fastest and species_in[n] is the slowest species.
#	+species_out ([Species]) = List containing references to the species used as output of the reaction.
#       +mesh (Mesh) = Mesh or Mesh tree used for this type of collision.
#       +nmax ([int]) = Numpy array of ints N_{max} describing the max. amounf of collisions per cell. The size of the array is the same size as
#           accPoints/nPoints for the mesh, using Nmax = 0 for the nodes that do not represent a cell.
#       +mfpath (double) = mean free path of the collision.
#       +nu (double) = collision frequency.
#       +fnum (int) = nominal specific weight of the slowest particles.
#       +dt (double) = Timestep used for this collision.
#Methods:
#       +sortByCell(Species species): [int] particles, [[int]] indices = The function receives a Species and calculates for each node of the mesh, the amount of particles located in the cell 
#           represented by the node. If 'indices = True' the function also returns a list of lists, where for each node the indices of the particles contained in the respective cell are stored.
#       +computeSigma_pairs(*args1, *args2): [double] = This function receives all the information necessary for computing the cross section of a collision. Each argument it receives is a
#           list, so that the function computes a list of crossSections as well, one per each row in the argument lists.
#       +computeSigma([int] part1, [int] part2) [double, double] = calculates the cross section for each combination of pairs, one from 'part1' and one from 'part2'. 
#           The colliding particles are passed through part1 and part2, which refer to the particles denoted by indices in the arrays stored in Particle_In_Mesh objects. 
#           The returned object is a matrix such that matrix[ind1,ind2] = sigma(part1[ind1, part2[ind2]).
#       +computeDiffVel([double,...,double] vel1, [double,...,double] vel2): [len1, len2, double,..., double] velm = The function receives two lists of velocities and performs the
#           substraction of every pair of velocities vel1-vel2, storing it in the position [ind1, ind2] for ind1 the index of vel1 in the first list and ind2 the index of vel2 in the
#           second list. The array returned has 3 dimensions, being the relative velocity stored in the 3 dimension.
#       +computeEllasticCollisionVelocities(Species species1, Species species2, [int] ind1, [int] ind2, v1_minus_v2 = [len1, len2, double,..., double] velm): 
#           [double,...,double] v1_p, [double,...,double] v2_p = The method receives the species1 and species2 which collide, the indices ind1 and ind2 
#           that indicate the respective collision pairs (so len(ind1) must be equal to len(ind2)), and the code returns the arrays v1_p and v2_p of the new velocites 
#           for species1 and species2, respectively. The code can also receive v1_minus_v2 which gives \vec{v1}-\vec{v2} for v1_minus_v2[ind1,ind2].
#       +computeDSMC() = This method performs all the DSMC method. It makes the calculations necessary for Nmax, updates Nmax as an attribute, and performs the Nmax cycle, with its
#           necessary modifications over velocities for MEX collisions and/or creation of new particles for CEX.
#           NOTES: So far, I am doing all the procedure in one function in order to re-use byproducts of calculations in some steps for later steps. However, it might be good to consider
#           Breaking it down. Also, always consider that this might me parallelized very soon, as the method is higly parallelizable. Also, do not forget, different particles might have
#           different SPWTs.
class DMSC_Boruk(object):
    def __init__(self, n_type, n_species_in, n_species_out, n_mesh, n_mfpath, n_nu, n_fnum, n_dt):
        self.type = n_type
        self.species_in = n_species_in
        self.species_out = n_species_out
        self.mesh = n_mesh
        self.nmax = numpy.zeros((self.mesh.nPoints), dtype = numpy.int64)
        self.mfpath = n_mfpath
        self.nu = n_nu
        self.fnum = n_fnum
        self.dt = n_dt

    def sortByCell(species, indices = True):
        #Getting mesh coordinates
        mc = self.mesh.getIndex(self.species.part_values.position[:species.part_values.current_n, :])
        
        index = mc.astype(int)
        array = self.mesh.indexToArray(index)
        
        field = numpy.zeros_like(self.nmax)
        numpy.add.at(field, array, 1)

        if indices == True:
            lists = [[] for i in range(len(field))]
            indices = numpy.arange(species.part_values.current_n, dtype = numpy.uint64)
            for node, part in zip(array, indices):
                lists[node].append(part)
            return field, lists
        return field

    def computeSigma_pairs():
        pass

    def computeSigma(part1, part2): 
        pass
        
    def computeDiffVel(vel1, vel2):
        len1 = len(vel1[:,0])
        len2 = len(vel2[:,0])
        vel1_ = numpy.repeat(vel1, len2, axis = 0)
        vel2_ = numpy.repeat(vel2[numpy.newaxis,...], len1, axis = 0).reshape((len1*len2, numpy.shape(vel2)[1]))
        diff = (vel1-vel2).reshape((len1, len2, numpy.shape(vel2)[1]), order = 'C')
        return diff

    def computeEllasticCollisionVelocities(species1, species2, ind1, ind2, v1_minus_v2 = None): 
        if v1_minus_v2 is None:
            v1_minus_v2 = species1.part_values.velocity[ind1]-species2.part_values.velocity[ind2]
        x1_minus_x2 = species.part_values.position[ind1]-species2.part_values.position[ind2]
        v1_p = species1.part_values.velocity[ind1]-2*species2.m*species2.part_values.spwt[ind2]/(species1.m*species1.part_values.spwt[ind1]+species2.m*species2.part_values.spwt[ind2])*\
                numpy.inner(v1_minus_v2, x1_minus_x2)/numpy.sum(x1_minus_x2*x1_minus_x2, axis = 1)*x1_minux_x2
        v2_p = species2.part_values.velocity[ind2]+2*species1.m*species1.part_values.spwt[ind1]/(species1.m*species1.part_values.spwt[ind1]+species2.m*species2.part_values.spwt[ind2])*\
                numpy.inner(v1_minus_v2, x1_minus_x2)/numpy.sum(x1_minus_x2*x1_minus_x2, axis = 1)*x1_minux_x2
        return v1_p, v2_p

    @Timing
    def computeDSMC():
        pass


#DSMC_Boruk_MEX_diff (inherits from DSMC_Boruk):
#
#Definition = Boruk DSMC method applied to Momentum Exchange collisions (MEX) between two different species.
#Attributes:
#       +type (string) = Type of collision represented. The name pattern is: "MEX-species_in[0],...,species_in[i]->species_in[0],...,species_in[i]".
#	+species_in ([Species]) = List containing references to the species used as input for the reaction. I am using the protocol of species_in[0] is the fastest and species_in[n] is the slowest species.
#	+species_out ([Species]) = List containing references to the species used as output of the reaction. For this type of reaction, species_out == species_in.
#       +Same as DSMC_Boruk
#Methods:
#       +Same as DSMC_Boruk
class DMSC_Boruk_MEX_diff(DSMC_Boruk):
    def __init__(self, n_type, n_species_in, n_mesh, n_mfpath, n_nu):
        super().__init__(n_type, n_species_in, n_species_in, n_mesh, n_mfpath, n_nu, species_in[1].spwt, species_in[1].dt)

    def computeSigma_pairs():
        pass

    def computeSigma(part1, part2): 
        vel1 = species_in[0].part_values.velocity[part1,:]
        vel2 = species_in[1].part_values.velocity[part2,:]
        len1 = len(vel1[:,0])
        len2 = len(vel2[:,0])
        vel1_ = numpy.repeat(vel1, len2, axis = 0)
        vel2_ = numpy.repeat(vel2[numpy.newaxis,...], len1, axis = 0).reshape((len1*len2, numpy.shape(vel2)[1]))
        sigma = self.computeSigma_pairs(vel1_, vel2_).reshape((len(vel1), len(vel2)), order = 'C')
        return sigma
        
#    @Timing
#    def computeNmax(index, count1, ind1, count2, int2):
#        crossSection = self.computeSigma(ind1[index], ind2[index])
#        relVel = self.computeDiffVel(self.species_in[0].part_values.velocity[ind1[index],:], self.species_in[1].part_values.velocity[ind2[index],:])
#        relVelMag = numpy.linalg.norm(relVel, axis = 2)
#        sigmavel_max = numpy.max(relvelMag*crossSection)
#        maxSPWT1 = numpy.max(self.species_in[0].part_values.spwt[ind1[index]])
#        maxSPWT2 = numpy.max(self.species_in[1].part_values.spwt[ind2[index]])
#        maxSPWT = maxSWPT1 if maxSWPT1 > maxSPWT2 else maxSPWT2
#        self.nmax[index] = count1[index]*count2[index]*maxSPWT*sigmavel_max*self.fnum*self.dt/self.mesh.volumes[index]
#
#        return relVel, crossSection

    @Timing
    def computeDSMC():
        count1, ind1 = self.sortByCell(self.species_in[0])
        count2, ind2 = self.sortByCell(self.species_in[1])
        maxdt = species_in[0].dt if species_in[0].dt > species_in[1].dt else species_in[1].dt
        for i in range(self.mesh.nPoints):
            if len(ind1[i]) > 0 and len(ind2[i]) > 0:
                #Compute Nmax
                crossSection = self.computeSigma(ind1[i], ind2[i])
                relVel = self.computeDiffVel(self.species_in[0].part_values.velocity[ind1[i],:], self.species_in[1].part_values.velocity[ind2[i],:])
                relVelMag = numpy.linalg.norm(relVel, axis = 2)
                sigmaVel_max = numpy.max(relvelMag*crossSection)
                maxSPWT1 = numpy.max(self.species_in[0].part_values.spwt[ind1[i]])
                maxSPWT2 = numpy.max(self.species_in[1].part_values.spwt[ind2[i]])
                maxSPWT = maxSWPT1 if maxSWPT1 > maxSPWT2 else maxSPWT2
                self.nmax[i] = count1[i]*count2[i]*maxSPWT*sigmaVel_max*self.fnum*self.dt/self.mesh.volumes[i]

                #Iterating over Nmax pairs
                r = numpy.random.rand(self.nmax[i], 3)
                elec1 = ind1[(r[:,0]*count1).astype(int)]
                elec2 = ind2[(r[:,1]*count2).astype(int)]
                prob = relVelMag[elec1, elec2]*crossSection[elec1, elec2]/sigmaVel_max
                colls = numpy.flatnonzero(r[:,2] < prob)

                #Replacing velocities for MEX
                v1_p, v2_p = self.computeEllasticCollisionVelocities(self.species_in[0], self.species_in[1], elec1[colls], elec2[colls], v1_minus_v2 = relVel[elec1[colls], elec2[colls]])
                maxSPWT = numpy.where(species_in[0].part_values.spwt[elec1[colls]] > species_in[1].part_values.spwt[elec2[colls]],\
                                      species_in[0].part_values.spwt[elec1[colls]], species_in[1].part_values.spwt[elec2[colls]])
                probVel1 = species_in[1].part_values.spwt[elec2[colls]]/maxSPWT*maxdt/species_in[1].dt
                probVel2 = species_in[0].part_values.spwt[elec1[colls]]/maxSPWT*maxdt/species_in[0].dt
                r2 = numpy.random.rand(colls, 2)
                repl1 = r2[:,0] < probVel1
                repl2 = r2[:,0] < probVel2
                #Replacement
                species_out[0].part_values.velocity[elec1[colls[repl1]], :] = v1_p[repl1]
                species_out[1].part_values.velocity[elec2[colls[repl2]], :] = v2_p[repl2]


#DSMC_Boruk_MEX_same (inherits from DSMC_Boruk_diff):
#
#Definition = Boruk DSMC method applied to Momentum Exchange collisions (MEX) between the same species.
#Attributes:
#       +type (string) = Type of collision represented. The name pattern is: "MEX-species_in[0],...,species_in[i]->species_in[0],...,species_in[i]".
#	+species_in ([Species]) = List containing references to the species used as input for the reaction. In this case species_in[0] == species_in[1].
#	+species_out ([Species]) = List containing references to the species used as output of the reaction. For this type of reaction, species_out == species_in.
#       +Same as DSMC_Boruk
#Methods:
#       +Same as DSMC_Boruk
class DMSC_Boruk_MEX_same(DSMC_Boruk_MEX_diff):
    def __init__(self, n_type, n_species_in, n_mesh, n_mfpath, n_nu):
        super().__init__(self, n_type, n_species_in, n_mesh, n_mfpath, n_nu)

    def computeSigma_pairs():
        pass

#    @Timing
#    def computeNmax(index, count1, ind1, count2, int2):
#        crossSection = self.computeSigma(ind1[index], ind2[index])
#        relVel = self.computeDiffVel(self.species_in[0].part_values.velocity[ind1[index],:], self.species_in[1].part_values.velocity[ind2[index],:])
#        relVelMag = numpy.linalg.norm(relVel, axis = 2)
#        sigmavel_max = numpy.max(relvelMag*crossSection)
#        maxSPWT1 = numpy.max(self.species_in[0].part_values.spwt[ind1[index]])
#        maxSPWT2 = numpy.max(self.species_in[1].part_values.spwt[ind2[index]])
#        maxSPWT = maxSWPT1 if maxSWPT1 > maxSPWT2 else maxSPWT2
#        self.nmax[index] = count1[index]*count2[index]*maxSPWT*sigmavel_max*self.fnum*self.dt/self.mesh.volumes[index]
#
#        return relVel, crossSection

    @Timing
    def computeDSMC():
        count1, ind1 = self.sortByCell(self.species_in[0])
        for i in range(self.mesh.nPoints):
            if len(ind1[i]) > 1:
                #Compute Nmax
                crossSection = self.computeSigma(ind1[i], ind1[i])
                relVel = self.computeDiffVel(self.species_in[0].part_values.velocity[ind1[i],:], self.species_in[1].part_values.velocity[ind1[i],:])
                for j in range(len(ind1[i])):
                    crossSection[j,:j+1] = 0
                    relVel[j,:j+1] = 0
                relVelMag = numpy.linalg.norm(relVel, axis = 2)
                sigmaVel_max = numpy.max(relvelMag*crossSection)
                maxSPWT = numpy.max(self.species_in[0].part_values.spwt[ind1[i]])
                self.nmax[i] = count1[i]*(count1[i]-1)/2*maxSPWT*sigmaVel_max*self.fnum*self.dt/self.mesh.volumes[i]

                #Iterating over Nmax pairs
                r = numpy.random.rand(self.nmax[i], 3)
                elec1 = ind1[(r[:,0]*count1).astype(int)]
                elec2 = ind2[(r[:,1]*count1).astype(int)]
                elec2 += numpy.where(elec2 == elec1, 1, 0)
                elec2 %= count1
                prob = relVelMag[elec1, elec2]*crossSection[elec1, elec2]/sigmaVel_max
                colls = numpy.flatnonzero(r[:,2] < prob)

                #Replacing velocities for MEX
                v1_p, v2_p = self.computeEllasticCollisionVelocities(self.species_in[0], self.species_in[1], elec1[colls], elec2[colls], v1_minus_v2 = relVel[elec1[colls], elec2[colls]])
                maxSPWT = numpy.where(species_in[0].part_values.spwt[elec1[colls]] > species_in[1].part_values.spwt[elec2[colls]],\
                                      species_in[0].part_values.spwt[elec1[colls]], species_in[1].part_values.spwt[elec2[colls]])
                probVel1 = species_in[1].part_values.spwt[elec2[colls]]/maxSPWT
                probVel2 = species_in[0].part_values.spwt[elec1[colls]]/maxSPWT
                r2 = numpy.random.rand(colls, 2)
                repl1 = r2[:,0] < probVel1
                repl2 = r2[:,0] < probVel2
                #Replacement
                species_out[0].part_values.velocity[elec1[colls[repl1]], :] = v1_p[repl1]
                species_out[1].part_values.velocity[elec2[colls[repl2]], :] = v2_p[repl2]
