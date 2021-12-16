import copy
from functools import reduce
from itertools import zip_longest
import numpy
import pdb

import accelerated_functions as af
from field import Field
from timing import Timing

#Motion_Solver(Abstract-like: As with Boundary, some methods can be applied to various motion solvers, so they will be stored here):
#
#Definition = Class that shows which methods and attributes need to have all motion solvers.
#Attributes:
#	+type (string) = string indicating the type of scheme.
#       +pic (PIC) = PIC solver.
#Methods:
#       +initialConfiguration(Species, [Field]) = Make necessary adjustment to the initial configuration of species so as to begin the advancement in time.
#	+advance(Species, [Field]) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#	+updateMeshValues(Species) = Update the attributes of Particles_In_Mesh.
#       +updateParticles(Species, [Field]) = Particle advance in time.
class Motion_Solver(object):
    def __init__(self, pic_slv):
        self.pic = pic_slv

    def initialConfiguration(self, species, fields):
        pass

    def advance(self, species, fields):
        pass
    
    def updateMeshValues(self, species):
        pass

    def updateParticles(self, species, fields):
        pass


#Leap_Frog(Inherits from Motion_Solver):
#
#Definition = Implementation of Leap_Frog method. This method, as being more specific, is more dependant on the current situation of each simulation. As such it has to
#               be updated or new classes has to be created if the situation varies.
#
#Attributes: 
#	+type (string) = "Leap Frog"
#       +pic (PIC) = PIC solver. For this class specific methods not provided by PIC super class are used.
#       +vel_dic {String : [double,double]} = Is a dictionary containing the new values of velocities for every species. The key 'String' is the actual name of the species.
#Methods:
#       +initialConfiguration(Species, Field) = Make necessary adjustment to the initial configuration of species so as to begin the advancement in time.
#           So far just E, so [Field]->Field.
#       +rewindVelocity(species, field) = Take the velocity of particles half a step back in time for 'field' electric field.
#	+advance(Species, [Field]) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#           Extent is a kwarg for updateMeshValues().
#	+updateMeshValues(Species) = Update the attributes of Particles_In_Mesh. Particular for each species, so it needs to be updated with every new species.
#           Extent can be '0' indicating that every attribute of mesh_values is updated, '1' indicating that only necessary attributes are updated, and 
#           '2' indicating that the 'non-necessary' attributes are the only ones updated. Notice that the criteria between 'necessary' and 'non-necessary'
#           attributes will change with the type of phenomena being included.
#       +updateParticles(Species, Field) = Particle advance in time. So far only E, so [Field]->Field in argument.
#       +motionTreatment(Species species) = It takes the velocities array in the species.part_values atribute and average it with the velocities stored in vel_dic for the particular species.
#           That is, to synchronize velocity with position.
#	+Motion_Solver methods.
class Leap_Frog(Motion_Solver):
    def __init__(self, pic_slv, species_names, max_n, vel_dim):
        super().__init__(pic_slv)
        self.type = "Leap Frog"
        self.vel_dic = {}
        for i in range(len(species_names)):
            self.vel_dic[species_names[i]] = numpy.zeros((max_n[i], vel_dim[i]))

#       +initialConfiguration(Species, Field) = Make necessary adjustment to the initial configuration of species so as to begin the advancement in time.
#           So far just E, so [Field]->Field.
    def initialConfiguration(self, species, field, ind = None):
        if type(ind) == type(None):
            ind = numpy.arange(species.part_values.current_n, dtype = numpy.uint32)
        self.rewindVelocity(species, field, ind)

#       +rewindVelocity(species, field) = Take the velocity of particles half a step back in time for 'field' electric field.
    def rewindVelocity(self, species, field, ind):
        species.part_values.velocity[ind,:] -= species.q_over_m*species.dt/2*field.fieldAtParticles(species.part_values.position[ind,:])

#	+advance(Species species, [Field] e_fields, [Field] m_fields) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#           extent is a kwarg for updateMeshValues().
#           update_dic is a kwarg for updateParticles(). 
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    @Timing
    def advance(self, species, e_fields, m_fields, extent = 0, update_dic = 1, types_boundary = ['open'], **kwargs):
        result_dic = self.updateParticles(species, e_fields, m_fields, update_dic, types_boundary, **kwargs)
        self.updateMeshValues(species, extent = extent, **result_dic)
        return result_dic

#	+updateMeshValues(Species) = Update the attributes of Particles_In_Mesh. Particular for each species, so it needs to be updated with every new species.
#           Extent can be '0' indicating that every attribute of mesh_values is updated, '1' indicating that only necessary attributes are updated, and 
#           '2' indicating that the 'non-necessary' attributes are the only ones updated. Notice that the criteria between 'necessary' and 'non-necessary'
#           attributes will change with the type of phenomena being included.
    @Timing
    def updateMeshValues(self, species, extent = 0, scatter_flux = 1, **kwargs):
        if extent == 0:
            if scatter_flux == 1:
                self.pic.scatterFlux(species, kwargs['flux'])
            self.pic.scatterDensity(species)
            #self.motionTreatment(species)
            self.pic.scatterSpeed(species)
            self.pic.scatterTemperature(species)
        elif extent == 1:
            if scatter_flux == 1:
                self.pic.scatterFlux(species, kwargs['flux'])
            self.pic.scatterDensity(species)
        elif extent == 2:
            self.motionTreatment(species)
            self.pic.scatterSpeed(species)
            self.pic.scatterTemperature(species)

#       +updateParticles(Species, Field, int) = Particle advance in time. So far only E, so [Field]->Field in argument.
#           If update_dic == 1, the vel_dic entry for the species is updated, otherwise it remains untouched.
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    @Timing
    def updateParticles(self, species, field, update_dic, type_boundary, **kwargs):
        np = species.part_values.current_n
        species.part_values.velocity[:np,:] += species.q_over_m*species.dt*field.fieldAtParticles(species.part_values.position[:np,:])
        species.part_values.position[:np,:] += species.part_values.velocity[:np,:]*species.dt
        if update_dic == 1:
            self.vel_dic[species.name] = copy.copy(species.part_values.velocity)
        for boundary in self.pic.mesh.boundaries:
            boundary.applyParticleBoundary(species, type_boundary, **kwargs)

#       +motionTreatment(Species species) = It takes the velocities array in the species.part_values atribute and average it with the velocities stored in vel_dic for the particular species.
#           That is, to synchronize velocity with position.
    def motionTreatment(self, species):
        np = species.part_values.current_n
        species.part_values.velocity[:np,:] = af.motionTreatment_p(species.part_values.velocity[:np,:], self.vel_dic[species.name][:np,:])


#Boris_Push(Inherits from Motion_Solver):
#
#Definition = Implementation of  Boris algorithm for time integration. This method also creates desynchronization between v and x.
#
#Attributes: 
#	+type (string) = "Boris Push"
#       +pic (PIC) = PIC solver. For this class specific methods not provided by PIC super class are used.
#       +vel_dic {String : [double,double]} = Is a dictionary containing the new values of velocities for every species. The key 'String' is the actual name of the species.
#Methods:
#       +initialConfiguration(Species, Field) = Make necessary adjustment to the initial configuration of species so as to begin the advancement in time.
#           So far just E, so [Field]->Field.
#       +rewindVelocity(species, field) = Take the velocity of particles half a step back in time for 'field' electric field.
#	+advance(Species species, [Field] e_fields, [Field] m_fields) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#           extent is a kwarg for updateMeshValues().
#           update_dic is a kwarg for updateParticles(). 
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
#	+updateMeshValues(Species) = Update the attributes of Particles_In_Mesh. Particular for each species, so it needs to be updated with every new species.
#           Extent can be '0' indicating that every attribute of mesh_values is updated, '1' indicating that only necessary attributes are updated, and 
#           '2' indicating that the 'non-necessary' attributes are the only ones updated. Notice that the criteria between 'necessary' and 'non-necessary'
#           attributes will change with the type of phenomena being included.
#       +updateParticles(Species species, [Field] e_fields, [Field] m_fields, int) = Particle advance in time. The function can receive multiple electric and magnetic fields.
#           If update_dic == 1, the vel_dic entry for the species is updated, otherwise it remains untouched.
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default method, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
#       +electricAdvance([double,double] e_field, Species species, double dt) = Updates the velocity one 'dt' forward.
#           Boris (1970) advancement in time. Is the usual Leap-Frog advancement in time.
#       +magneticRotation (Field B, Species species, double dt) = Update velocity applying magnetic rotation.
#           Buneman (1973) rotation.
#       +motionTreatment(Species species) = It takes the velocities array in the species.part_values atribute and average it with the velocities stored in vel_dic for the particular species.
#           That is, to synchronize velocity with position.
#	+Motion_Solver methods.
class Boris_Push(Motion_Solver):
    def __init__(self, pic_slv, species_names, max_n, vel_dim):
        super().__init__(pic_slv)
        self.type = "Leap Frog - Boris Push"
        self.vel_dic = {}
        for i in range(len(species_names)):
            self.vel_dic[species_names[i]] = numpy.zeros((max_n[i], vel_dim[i]))

#       +initialConfiguration(Species, Field) = Make necessary adjustment to the initial configuration of species so as to begin the advancement in time.
#           So far just E, so [Field]->Field.
    def initialConfiguration(self, species, field, ind = None):
        if type(ind) == type(None):
            ind = numpy.arange(species.part_values.current_n, dtype = numpy.uint32)
        self.rewindVelocity(species, field, ind)


#       +rewindVelocity(species, field) = Take the velocity of particles half a step back in time for 'field' electric field.
    def rewindVelocity(self, species, field, ind):
        pos = species.part_values.position[ind,:]
        vel = species.part_values.velocity[ind,:]
        field_ar = field.fieldAtParticles(pos)
        species.part_values.velocity[ind,:] = af.rewindVelocity_p(vel, field_ar, species.q_over_m, species.dt)

#	+advance(Species species, [Field] e_fields, [Field] m_fields) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#           extent is a kwarg for updateMeshValues().
#           update_dic is a kwarg for updateParticles(). 
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    @Timing
    def advance(self, species, e_fields, m_fields, extent = 0, update_dic = 1, types_boundary = ['open'], **kwargs):
        result_dic = self.updateParticles(species, e_fields, m_fields, update_dic, types_boundary, **kwargs)
        self.updateMeshValues(species, extent = extent, **result_dic)
        return result_dic

#	+updateMeshValues(Species) = Update the attributes of Particles_In_Mesh. Particular for each species, so it needs to be updated with every new species.
#           Extent can be '0' indicating that every attribute of mesh_values is updated, '1' indicating that only necessary attributes are updated, and 
#           '2' indicating that the 'non-necessary' attributes are the only ones updated. Notice that the criteria between 'necessary' and 'non-necessary'
#           attributes will change with the type of phenomena being included.
    @Timing
    def updateMeshValues(self, species, extent = 1, scatter_flux = 1, **kwargs):
        #if species.name == "Electron - Solar wind" or species.name == "Proton - Solar wind":
        if extent == 0:
            if scatter_flux == 0:
                self.pic.scatterFlux(species, kwargs['flux'])
            self.pic.scatterDensity(species)
            #NOTE: Be careful, the concept behind scatter_flux == 0 has changed, which is why here motion treatment is commented
            #self.motionTreatment(species)
            self.pic.scatterSpeed(species)
            self.pic.scatterTemperature(species)
        elif extent == 1:
            if scatter_flux == 1:
                self.pic.scatterFlux(species, kwargs['flux'])
            self.pic.scatterDensity(species)
        elif extent == 2:
            self.motionTreatment(species)
            self.pic.scatterSpeed(species)
            self.pic.scatterTemperature(species)

#       +updateParticles(Species species, [Field] e_fields, [Field] m_fields, int) = Particle advance in time. The function can receive multiple electric and magnetic fields.
#           If update_dic == 1, the vel_dic entry for the species is updated, otherwise it remains untouched.
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default method, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    @Timing
    def updateParticles(self, species, e_fields, m_fields, update_dic, types_boundary, **kwargs):
        result_dic = {}
        #Summing over the different fields
        e_total = copy.deepcopy(e_fields[0])
        m_total = copy.deepcopy(m_fields[0])
        e_total = reduce(lambda x,y: x+y, e_fields[1:], e_total)
        m_total = reduce(lambda x,y: x+y, m_fields[1:], m_total)
        #Fields at positions
        np = species.part_values.current_n
        e_field = e_total.fieldAtParticles(species.part_values.position[:np,:])
        m_field = m_total.fieldAtParticles(species.part_values.position[:np,:])
        #Updating velocity
        species.part_values.velocity[:np,:] = self.electricHalfAdvance(e_field, species, np)
        species.part_values.velocity[:np,:] = self.magneticRotation(m_field, species, np)
        species.part_values.velocity[:np,:] = self.electricHalfAdvance(e_field, species, np)
        #Updating position
        old_position = copy.copy(species.part_values.position)
        #assert len(numpy.nonzero(numpy.linalg.norm(species.part_values.velocity[:np,:]*species.dt, axis = 1) > self.pic.mesh.dx)[0]) < 200, "It moved more than dx"
        species.part_values.position[:np,:] = af.updateParticles_p(species.part_values.position[:np,:], species.part_values.velocity[:np,:], species.dt)

        if update_dic == 1:
            self.vel_dic[species.name] = copy.copy(species.part_values.velocity)
        for boundary, type_boundary in zip_longest(self.pic.mesh.boundaries, types_boundary, fillvalue = types_boundary[0]):
            result_boundary = boundary.applyParticleBoundary(species, type_boundary, old_position = old_position, **kwargs)
            if result_boundary is not None:
                if 'del_ind' in result_boundary.keys():
                    old_position = numpy.delete(old_position, result_boundary['del_ind'], axis = 0)
                    result_boundary.pop('del_ind')

                if 'flux' in result_boundary.keys() and 'flux' in result_dic.keys():
                    temp = []
                    for val_old, val_new in zip(result_dic['flux'],result_boundary['flux']):
                       temp.append(numpy.append(val_old, val_new, axis = 0))
                    result_dic['flux'] = tuple(temp)
                    del result_boundary['flux']

                result_dic.update(result_boundary)

        return result_dic

#       +electricAdvance([double,double] e_field, Species species, double dt) = Updates the velocity one 'dt' forward.
#           Boris (1970) advancement in time. Is the usual Leap-Frog advancement in time.
    def electricHalfAdvance(self, e_field, species, current_n):
        velocity = species.part_values.velocity[:species.part_values.current_n,:]
        return af.electricAdvance_p(velocity, e_field, species.q_over_m, species.dt/2)


#       +magneticRotation (Field B, Species species, double dt) = Update velocity applying magnetic rotation.
#           Buneman (1973) rotation.
    def magneticRotation (self, B, species, current_n):
        vx = species.part_values.velocity[:species.part_values.current_n,0]
        vy = species.part_values.velocity[:species.part_values.current_n,1]
        B = B[:,0]
        newx, newy = af.magneticRotation_p(vx, vy, B, species.q_over_m, species.dt)
        return numpy.append(newx[:,None], newy[:,None], axis = 1)

#       +motionTreatment(Species species) = It takes the velocities array in the species.part_values atribute and average it with the velocities stored in vel_dic for the particular species.
#           That is, to synchronize velocity with position.
    def motionTreatment(self, species):
        np = species.part_values.current_n
        species.part_values.velocity[:np,:] = af.motionTreatment_p(species.part_values.velocity[:np,:], self.vel_dic[species.name][:np,:])


#Leap_Frog_2D3Dcm(Inherits from Leap_Frog):
#
#Definition = Implementation of Leap_Frog method for a Hybrid (z-r) cylindrical mesh where positions of particles are in 2D and velocities in 3D.
#
#Attributes: 
#	+type (string) = "Leap Frog - 2D-3D-Cylindrical"
#       +pic (PIC) = PIC solver. For this class specific methods not provided by PIC super class are used.
#       +vel_dic {String : [double,double,double]} = Is a dictionary containing the new values of velocities for every species. The key 'String' is the actual name of the species.
#Methods:
#       +initialConfiguration(Species, Field) = Make necessary adjustment to the initial configuration of species so as to begin the advancement in time.
#           So far just E, so [Field]->Field.
#       +rewindVelocity(species, field) = Take the velocity of particles half a step back in time for 'field' electric field.
#	+advance(Species, [Field]) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#           Extent is a kwarg for updateMeshValues().
#	+updateMeshValues(Species) = Update the attributes of Particles_In_Mesh. Particular for each species, so it needs to be updated with every new species.
#           Extent can be '0' indicating that every attribute of mesh_values is updated, '1' indicating that only necessary attributes are updated, and 
#           '2' indicating that the 'non-necessary' attributes are the only ones updated. Notice that the criteria between 'necessary' and 'non-necessary'
#           attributes will change with the type of phenomena being included.
#       +updateParticles(Species, Field) = Particle advance in time. So far only E, so [Field]->Field in argument.
#       +motionTreatment(Species species) = It takes the velocities array in the species.part_values atribute and average it with the velocities stored in vel_dic for the particular species.
#           That is, to synchronize velocity with position.
#	+Motion_Solver methods.
class Leap_Frog_2D3Dcm(Leap_Frog):
    def __init__(self, pic_slv, species_names, max_n, vel_dim):
        super().__init__(pic_slv, species_names, max_n, vel_dim)
        self.type += " - 2D-3D-Cylindrical"

#       +rewindVelocity(species, field) = Take the velocity of particles half a step back in time for 'field' electric field.
    def rewindVelocity(self, species, field, ind):
        pos = species.part_values.position[ind,:]
        vel = species.part_values.velocity[ind,:2]
        field_ar = field.fieldAtParticles(pos)
        species.part_values.velocity[ind,:2] = af.rewindVelocity_p(vel, field_ar, species.q_over_m, species.dt)

#	+advance(Species species, [Field] e_fields, [Field] m_fields) = Advance the particles in time. It will treat the particles as well as update the mesh_values.
#           extent is a kwarg for updateMeshValues().
#           update_dic is a kwarg for updateParticles(). 
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default mthod, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    @Timing
    def advance(self, species, e_fields, m_fields, extent = 0, update_dic = 1, types_boundary = ['open'], **kwargs):
        result_dic = self.updateParticles(species, e_fields, m_fields, update_dic, types_boundary, **kwargs)
        self.updateMeshValues(species, extent = extent, **result_dic)
        return result_dic

#       +updateParticles(Species species, [Field] e_fields, [Field] m_fields, int) = Particle advance in time. The function can receive multiple electric and magnetic fields.
#           If update_dic == 1, the vel_dic entry for the species is updated, otherwise it remains untouched.
#           type_boundary indicates the type of boundary method to apply to particles. 'open', the default method, deletes them. 'reflective' reflects them back to the dominion.
#           **kwargs may contain arguments necessary for inner methods.
    @Timing
    def updateParticles(self, species, e_fields, m_fields, update_dic, types_boundary, **kwargs):
        result_dic = {}
        #Summing over the different fields
        e_total = copy.deepcopy(e_fields[0])
        m_total = copy.deepcopy(m_fields[0])
        e_total = reduce(lambda x,y: x+y, e_fields[1:], e_total)
        m_total = reduce(lambda x,y: x+y, m_fields[1:], m_total)
        #Fields at positions
        np = species.part_values.current_n
        e_field = e_total.fieldAtParticles(species.part_values.position[:np])
        m_field = m_total.fieldAtParticles(species.part_values.position[:np])
        #Some temporal variables
        temp_vel = copy.copy(species.part_values.velocity[:np])
        temp_pos = copy.copy(species.part_values.position[:np])
        temp_pos = numpy.append(temp_pos, numpy.zeros((np,1)), axis = 1)
        #Updating velocity
        temp_vel[:,:2] = self.electricAdvance(e_field, species, np)
        #Updating position
        old_position = copy.copy(species.part_values.position)
        #assert len(numpy.nonzero(numpy.linalg.norm(species.part_values.velocity[:np,:]*species.dt, axis = 1) > self.pic.mesh.dx)[0]) < 200, "It moved more than dx"
        temp_pos = af.updateParticles_p(temp_pos, temp_vel, species.dt)

        ##Rotate back to (z-r) plane
        species.part_values.position[:np,0] = temp_pos[:,0]
        species.part_values.position[:np,1] = numpy.linalg.norm(temp_pos[:,1:], axis = 1)
        sin_theta = temp_pos[:,2]/species.part_values.position[:np,1]
        cos_theta = numpy.sqrt(1-sin_theta*sin_theta)
        #Rotate velocity
        species.part_values.velocity[:np,0] = temp_vel[:,0]
        species.part_values.velocity[:np,1] = cos_theta*temp_vel[:,1]+sin_theta*temp_vel[:,2]
        species.part_values.velocity[:np,2] = -sin_theta*temp_vel[:,1]+cos_theta*temp_vel[:,2]

        if update_dic == 1:
            self.vel_dic[species.name] = copy.copy(species.part_values.velocity)
        for boundary, type_boundary in zip_longest(self.pic.mesh.boundaries, types_boundary, fillvalue = types_boundary[0]):
            result_boundary = boundary.applyParticleBoundary(species, type_boundary, old_position = old_position, **kwargs)
            if result_boundary is not None:
                if 'del_ind' in result_boundary.keys():
                    old_position = numpy.delete(old_position, result_boundary['del_ind'], axis = 0)
                    result_boundary.pop('del_ind')

                if 'flux' in result_boundary.keys() and 'flux' in result_dic.keys():
                    temp = []
                    for val_old, val_new in zip(result_dic['flux'],result_boundary['flux']):
                       temp.append(numpy.append(val_old, val_new, axis = 0))
                    result_dic['flux'] = tuple(temp)
                    del result_boundary['flux']

                result_dic.update(result_boundary)

        return result_dic

#       +electricAdvance([double,double] e_field, Species species, double dt) = Updates the velocity one 'dt' forward.
#           Is the usual Leap-Frog advancement in time.
    def electricAdvance(self, e_field, species, current_n):
        velocity = species.part_values.velocity[:species.part_values.current_n,:2]
        return af.electricAdvance_p(velocity, e_field, species.q_over_m, species.dt)
