#Data structures that contain the fields of the system
from functools import reduce

import copy
import numpy
import pdb
import sys
from vtk.util.numpy_support import vtk_to_numpy

import accelerated_functions as af
import constants as c
from mesh import Mesh_recursive
from pic import PIC_recursive
import solver as slv
from timing import Timing

#Field (Abstract):
#
#Definition = Indicate the attributes and methods that all fields have to implement. The fields obtain the functions to compute the fields from 'solver.py'
#Attributes:
#	+type (string) = some string that describes the source and type of the field (created by the interaction plasma-spacecraft, user-defined, constant, etc.)
#	+pic (PIC) = Class that contains PIC methods.
#	+boundaries ([Boundary]) = Class that represents the methods which apply boundaries to the fields.
#	+nPoints (int) = number of nodes in the mesh (Same as mesh class).
#	+field ([double, double]) = components (columns) of field at each node.
#Methods:
#	+__init__(...) = This function, for each subclass, will take care of the initial condition of the field.
#	+computeField([Species] species) = Computes the updated field values.
#	+fieldAtParticles([double,double] position) [double,double] = return an array indicating by component (columns) and particles (rows) the field at every position.
#       +saveVTK(Mesh mesh): dictionary = Return the attributes of field to be printed in the VTK file.
#           The process is handled inside each particular field, and the final result can be constructed from the output of different subclasses.
#       +loadVTK(Mesh mesh, output) = Takes information of the field from a VTK file through 'output' and stores it in the corresponding attributes.
#           The process is handled inside each particular field, and the final result can be constructed from the output of different subclasses.
class Field(object):
    def __init__(self, n_pic, field_dim, n_type):
        self.type = n_type
        self.pic = n_pic
        self.field = numpy.zeros((self.pic.mesh.nPoints, field_dim))
        self.ind_calc = self.getIndexCalculation(self.pic.mesh)

    def __add__(self, obj2):
        try:
            self.field += obj2.field
            return self
        except ValueError:
            print("Both Field objects must have the same array size")
            print(sys.exc_info())
            raise

    def getIndexCalculation(self, mesh):
        pass

    def computeField(self, species):
        pass

    def fieldAtParticles(self, position):
        return self.pic.gather(position, self.field)

    def saveVTK(self, mesh):
        return {self.type+"-field" : mesh.vtkOrdering(self.field)}

    def loadVTK(self, mesh, output):
        self.field = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(self.type+"-field")))


#Field_recursive(Abstract, Inherits from Field):
#
#Definition = Abstract class that gives to its children the blueprint for including recursive behavior. It should be added as the first parent of its children classes, so that its methods
#             are chosen by default when invoking super().
#Attributes:
#       +children ([Field]) = Objects that are the children of this instance.
#       +root (Boolean) = Indicates whether the object is the root Field (True) or not (False).
#Methods:
#       +__init__ =  adds " - Recursive" and initialize the list of children.
#       + total_field(): [Double, Double] = This method creates an array containing all the values of the field in all the meshes of the domain, organized by the flat indexation rule.
#       +Field methods.
class Field_recursive(Field):
    def __init__(self, n_children, n_root, n_type):
        n_type += " - Recursive"
        self.children = n_children
        self.root = n_root

    def __add__(self, obj2):
        try:
            for i in range(len(self.children)):
                self.children[i].__add__(ob2.children[i])
            super().__add__(self, obj2)
        except (IndexError, ValueError):
            print("Both Field objects must have the same Tree structure, referring to the same underlying meshes")
            print(sys.exc_info())
            raise
    
    def assignValuesToArray_recursive(self, name, indices, values, accIndex = None):
        if self.root:
            indices = self.pic.mesh.sortIndexByMeshes(indices)
            accIndex = [0]
        ind = indices.pop(0)
        array_t = self.__getattribute__(name) 
        array_t[ind] = values[accIndex[0]:accIndex[0]+len(ind)]
        self.__setattr__(name, array_t)
        accIndex[0] += len(ind)
        for child in self.children:
            child.assignValuesToArray_recursive(name, indices, values, accIndex = accIndex)

    def getTotalArray(self, name, seedList = None, index = None):
        field = self.__getattribute__(name)
        if self.root:
            dims = list(numpy.shape(field))
            dims[0] = self.pic.mesh.accPoints
            seedList = numpy.zeros(dims)
            index = [0]
        temp = index[0]
        index[0] += self.pic.mesh.nPoints
        seedList[temp:index[0]] += field
        for child in self.children:
            child.getTotalArray(name, seedList = seedList, index = index)
        return seedList

    def getTotalField(self):
        return self.getTotalArray("field")

    def fieldAtParticles(self, position):
        if self.root:
            tot_field = self.getTotalField()
            return self.pic.gather(position, tot_field)
        else:
            raise Exception('This instance of the functions should not have been executed')
        
    def saveVTK(self, mesh):
        return {self.type+"-field" : mesh.vtkOrdering(self.getTotalField())}

    #Fix this. The 'field' value from the vtr file is not coming with the information of all the meshes
    def loadVTK(self, mesh, output):
        temp = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(self.type+"-field")))
        self.assignValuesToArray_recursive("field", numpy.arange(self.pic.mesh.accPoints, dtype ='uint'), temp)


#Electric_Field (Inherits from Field):
#
#Definition = Electric field
#Attributes:
#	+potential ([double]) = Electric potential at each node of the mesh.
#	+Field attributes.
#Methods:
#	+Field methods.
class Electric_Field(Field):
    def __init__(self, n_pic, field_dim, n_string):
        self.potential = numpy.zeros((n_pic.mesh.nPoints))
        super().__init__(n_pic, field_dim, "Electric"+n_string)

    def getIndexCalculation(self, mesh):
        pass

    def computeField(self, species):
        pass

    def saveVTK(self, mesh):
        dic = super().saveVTK(mesh)
        dic[self.type+"-potential"] = mesh.vtkOrdering(self.potential)
        return dic

    def loadVTK(self, mesh, output):
        self.potential = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(self.type+"-potential")))
        super().loadVTK(mesh, output)


#Constant_Electric_Field(Electric_Field):
#
#Definition = Constant electric field impsoed by the user. Does not change through time.
#Attributes:
#	+type (string) = "Electric field - Constant".
#	+Electric_Field attributes.
#Methods:
#	+Electric_Field methods.
class Constant_Electric_Field(Electric_Field):
    def __init__(self, n_pic, field_dim):
        super().__init__(n_pic, field_dim, " - Constant")
        self.field[:,0] += 0.27992

    def computeField(self, species):
        pass


#Electrostatic_2D_rm_Electric_Field (Inherits from Electric_Field):
#
#Definition = Electric field for a 2D rectangular mesh, detached from the magnetic field. Uses methods from "solver.py" to calculate electric potential, and then electric field.
#Attributes:
#	+type (string) = "Electric - Electrostatic_2D_rm".
#	+Elctric_Field attributes.
#Methods:
#	+Electric_Field methods.
class Electrostatic_2D_rm(Electric_Field):
    def __init__(self, n_pic, field_dim):
        super().__init__(n_pic, field_dim, " - Electrostatic_2D_rm")

    def getIndexCalculation(self, mesh):
        temp = numpy.arange((mesh.nPoints))
        loc = numpy.unique(mesh.location)
        return numpy.delete(temp, loc)

    @Timing
    def computeField(self, species):
        #Prepare the right-hand-side of the Poisson equation 
        loc = numpy.unique(self.pic.mesh.location_sat)
        rho = numpy.zeros_like(species[0].mesh_values.density)
        for specie in species:
            rho += specie.mesh_values.density*specie.q
            rho[loc] += specie.mesh_values.accDensity*specie.q
        rho /= -c.EPS_0
        slv.poissonSolver_2D_rm_SORCA_p(self.pic.mesh, self.potential, rho, self.ind_calc)
        self.field = -slv.derive_2D_rm(self.pic.mesh, self.potential, self.ind_calc)
        for boundary in self.pic.mesh.boundaries:
            boundary.applyElectricBoundary(self)

#       +Computation of Dirichlet boundary condition at every node in location ([ind]). Every row in value ([double]) corresponds to one node in location.
#           boundary indicates whether the boundary is an 'inner' or 'outer' boundary. This is used for the calculation of the electric field.
    def dirichlet(self, values, boundary, nx, ny, dx, dy):
        #Dirichlet
        location, u_ind = numpy.unique(boundary.location, return_index = True)
        self.potential[location] = values[u_ind]
        #Electric field trough Pade 2nd order in the boundaries
        self.field[location, :] = -slv.derive_2D_rm_boundaries(self.potential, boundary, nx, ny, dx, dy)

#       +Neumann([ind] location, [double] valus) = Set Neumann conditions in the nodes at 'location'.
#           +values account for the values of the e_field normal to the border.
#           +Note: The Function doesn't handle the situation of the corners.
    def neumann(self, location, values):
        # Variables at hand
        nx = self.pic.mesh.nx
        ny = self.pic.mesh.ny
        dx = self.pic.mesh.dx
        dy = self.pic.mesh.dy
        #Field and potential
        for i in range(len(location)):
            if location[i] < nx:
                self.field[location[i],1] = values[i]
                self.potential[location[i]] = self.field[location[i],1]*dy+self.potential[location[i]+nx]
            elif location[i] > nx*(ny-1):
                self.field[location[i],1] = values[i]
                self.potential[location[i]] = -self.field[location[i],1]*dy+self.potential[location[i]-nx]
            elif location[i]%nx == 0:
                self.field[location[i],0] = values[i]
                self.potential[location[i]] = self.field[location[i],0]*dx+self.potential[location[i]+1]
            else:
                self.field[location[i],0] = values[i]
                self.potential[location[i]] = -self.field[location[i], 0]*dx+self.potential[location[i]-1]


#Electrostatic_2D_rm_sat (Inherits from Electrostatic_2D_rm):
#
#Definition = Same characteristics as Electrostatic_2D_rm but with an inner boundary representing the satellite.
#               For the class it is assumed that the satellite is stored as the second boundary in mesh. The surface is treated as a dielectric.
#Attributes:
#	+type (string) = "Electric - Electrostatic_2D_rm_sat".
#       +inv_capacity ([double,double]) = Inverse of the Capacity matrix for the nodes of the satellite.
#           The matrix is organized such that V = C^{-1}*q[location], with 'location' being the location of the nodes in the mesh in sorted order.
#	+Elctric_Field attributes.
#Methods:
#       +floating_potential([Species] species) = Computes the floating potential in a dielectric surface, updating the involved nodes of the 'potential' array.
#           This is done through the Capacity matrix method.
#       +computeField([Species] species) = First, the floating potential of a dielectric surface is calculated based on the accumulated charge.
#           Then, is the same behavior as the method in parent class.
#	+Electrostatic_2D_rm methods.
class Electrostatic_2D_rm_sat(Electrostatic_2D_rm):
    def __init__(self, n_pic, field_dim, capacity_file = c.CAPACITY_FILE):
        Electric_Field.__init__(self, n_pic, field_dim, " - Electrostatic_2D_rm_sat")
        tot_loc = len(numpy.unique(self.pic.mesh.location_sat))
        self.inv_capacity = numpy.zeros((tot_loc, tot_loc))
        if capacity_file is None:
            slv.capacity_Inv_Matrix_asym(self)
        else:
            self.inv_capacity = numpy.loadtxt('./data/'+capacity_file)

        try:
            self.capacity = numpy.linalg.inv(self.inv_capacity)
            att = 5e-4
            self.capacity *= c.EPS_0*att
            self.inv_capacity /= c.EPS_0*att
        except numpy.linalg.LinAlgError:
            print("Problem in Capacity Matrix")
            print(sys.exc_info())
            pdb.set_trace()

    def getIndexCalculation(self, mesh, border = False):
        temp = numpy.arange((mesh.nPoints))
        mask = numpy.ones((mesh.nPoints), dtype = bool)
        for boundary in mesh.boundaries:
            if boundary.type.split(sep="-")[0] == "Inner ":
                mask[boundary.ind_inner] = False
            if not border:
                mask[boundary.location] = False
        return temp[mask]

    @Timing
    def computeField(self, species):
        self.floating_potential(species)
        super().computeField(species)

    def floating_potential(self, species):
        loc = numpy.unique(self.pic.mesh.location_sat)
        charges = [specie.q*specie.mesh_values.accDensity*self.pic.mesh.volumes[loc] for specie in species]
        self.potential[loc] = numpy.matmul(self.inv_capacity, reduce(lambda x,y: x+y, charges).T)


#Electrostatic_2D_rm_sat_cond (Inherits from Electrostatic_2D_rm_sat):
#
#Definition = Same characteristics as Electrostatic_2D_rm_sat but the surface is conductive, as opposed to dielectric as in Electrostatic_2D_rm_sat.
#               For the class it is assumed that the satellite is stored as the second boundary in mesh.
#Attributes:
#	+type (string) = "Electric - Electrostatic_2D_rm_sat_cond".
#       +inv_capacity ([double,double]) = Inverse of the Capacity matrix for the nodes of the satellite.
#           The matrix is organized such that V = C^{-1}*q[location], with 'location' being the location of the nodes in the mesh in sorted order.
#       +capacity ([double,double]) = Capacity matrix for the nodes of the satellite. It is organized the same way as inv_caparcity.
#	+Electric_Field attributes.
#Methods:
#       +floating_potential([Species] species) = Computes the floating potential in a conductive surface, updating the involved nodes of the 'potential' array.
#           This is done through the Capacity matrix method.
#           WARNING: Here, first, the charges are accumuated as the particles impact or leave the surface. Then, the charges are redistributed to account for the
#               conductive surface. The change in densities in the surface is updated in 'Electron - Solar wind' class. In reality, all the electrons in the surface
#               can move, including, for example, photoelectrons that return to the surface. However, since this code does not track the movement of particles
#               in the surface, it is impossible to distingish among different types of electrons. Thus, changes are accumulated in the aforementioned class.
#	+Electrostatic_2D_rm_sat methods.
class Electrostatic_2D_rm_sat_cond(Electrostatic_2D_rm_sat):
    def __init__(self, n_pic, field_dim):
        super().__init__(n_pic, field_dim)
        self.type += "_cond"

    def getIndexCalculation(self, mesh, border = False):
        temp = numpy.arange((mesh.nPoints))
        mask = numpy.ones((mesh.nPoints), dtype = bool)
        for boundary in mesh.boundaries:
            if boundary.type.split(sep="-")[0] == "Inner ":
                mask[boundary.ind_inner] = False
            if not border:
                mask[boundary.location] = False
        return temp[mask]

    @Timing
    def computeField(self, species):
        #Potential at the material surfaces
        self.floating_potential(species)
        #Prepare the right-hand-side of the Poisson equation 
        loc = numpy.unique(self.pic.mesh.location_sat)
        rho = numpy.zeros_like(species[0].mesh_values.density)
        for specie in species:
            rho += specie.mesh_values.density*specie.q
            rho[loc] += specie.mesh_values.accDensity*specie.q
        rho /= -c.EPS_0
        slv.poissonSolver_2D_rm_SORCA(self.pic.mesh, self.potential, rho, self.ind_calc)
        self.field = -slv.derive_2D_rm(self.pic.mesh, self.potential, self.ind_calc)
        for boundary in self.pic.mesh.boundaries:
            boundary.applyElectricBoundary(self)

    def floating_potential(self, species):
        super().floating_potential(species)
        loc = numpy.unique(self.pic.mesh.location_sat)
        phi_c = numpy.sum(numpy.matmul(self.capacity, self.potential[loc].T))/numpy.sum(self.capacity)
        d_q = numpy.matmul(self.capacity, (phi_c-self.potential[loc]).T)
        #assert abs(numpy.sum(d_q)) < -c.QE or numpy.sum(d_q)/numpy.max(numpy.abs(d_q)) < 1e-4, "The redistribution of charge is creating or eliminating charge"

        #WARNING: See class documentation for more explanation.
        electron = list(filter(lambda specie: specie.name == "Electron - Solar wind", species))[0]
        #d_n = d_q/electron.q/self.pic.mesh.volumes[loc]
        #electron.mesh_values.accDensity += d_n
        self.potential[loc] = phi_c
        for boundary in self.pic.mesh.boundaries:
            if boundary.type == 'Inner - 2D_Rectangular':
                self.potential[boundary.ind_inner] = phi_c


class Electrostatic_2D_rm_sat_cond_recursive(Field_recursive, Electrostatic_2D_rm_sat_cond):
    def __init__(self, n_pic, field_dim, n_children, n_root, capacity_file = c.CAPACITY_FILE):
        Electric_Field.__init__(self, n_pic, field_dim, " - Electrostatic_2D_rm_sat_cond")
        super().__init__(n_children, n_root, self.type)

        if self.root == True:
            tot_loc = len(numpy.unique(self.pic.mesh.overall_location_sat))
            self.inv_capacity = numpy.zeros((tot_loc, tot_loc))
            if capacity_file is None:
                slv.capacity_Inv_Matrix_asym_recursive(self)
            else:
                self.inv_capacity = numpy.loadtxt('./data/'+capacity_file)

            try:
                self.capacity = numpy.linalg.inv(self.inv_capacity)
                att = 5e-4
                self.capacity *= c.EPS_0*att
                self.inv_capacity /= c.EPS_0*att
            except numpy.linalg.LinAlgError:
                print("Problem in Capacity Matrix")
                print(sys.exc_info())
                pdb.set_trace()

    def getIndexCalculation(self, mesh, border = False):
        return super(Field_recursive, self).getIndexCalculation(mesh, border = border)

    def computeField(self, species, rho = None, border = False, interpolate = False, adjustment = False):
        if self.root == True:
            if rho is None:
                loc = self.pic.mesh.overall_location_sat
                #Prepare the right-hand-side of the Poisson equation 
                rho = numpy.zeros_like(species[0].mesh_values.density)
                for specie in species:
                    rho += specie.mesh_values.density*specie.q
                    rho[loc] += specie.mesh_values.accDensity*specie.q
                rho /= -c.EPS_0
                self.floating_potential(species, rho)
            rho = self.pic.mesh.sortArrayByMeshes(rho)

        indices = self.ind_calc if not border else self.getIndexCalculation(self.pic.mesh, border = border)
        slv.poissonSolver_2D_rm_SORCA_p(self.pic.mesh, self.potential, rho.pop(0), indices, border = border, adjustment = adjustment)
        self.field = -slv.derive_2D_rm(self.pic.mesh, self.potential, self.ind_calc)
        if self.root == True and border == False:
            for boundary in self.pic.mesh.boundaries:
                boundary.applyElectricBoundary(self)
        else:
            for boundary in self.pic.mesh.boundaries:
                #if boundary.material != 'satellite':
                self.field[boundary.location, :] = -slv.derive_2D_rm_boundaries(self.potential, boundary,\
                        self.pic.mesh.nx, self.pic.mesh.ny, self.pic.mesh.dx, self.pic.mesh.dy)
        self.assignChildrenBoundaries(border = border, interpolate = interpolate)
        for child in self.children:
            child.computeField(species, rho = rho, border = border, interpolate = interpolate)

#   +induced_charge([double] rho) [double] charge = This functions computes the induced charge in the surface of the satellite due to the
#       electric field generated by the presence of the charge density distribution 'rho' in the domain.
    def induced_charge(self, rho, charge = None, acc_index = 0):
        #Charge induced by the electric field
        if self.root:
            rho = self.pic.mesh.sortArrayByMeshes(rho)
            charge = numpy.zeros_like(self.pic.mesh.overall_location_sat, dtype = 'float64')

        loc = self.pic.mesh.location_sat
        base_pot = copy.copy(self.potential)
        base_field = numpy.zeros_like(self.field)
        base_pot[self.pic.mesh.location] = 0
        slv.poissonSolver_2D_rm_SORCA_p(self.pic.mesh, base_pot, rho.pop(0), self.ind_calc)

        #Electric field in sat boundaires is calculated
        for boundary in self.pic.mesh.boundaries:
            if boundary.material == 'satellite':
                base_field[boundary.location,:] = -slv.derive_2D_rm_boundaries(base_pot, boundary,\
                        self.pic.mesh.nx, self.pic.mesh.ny, self.pic.mesh.dx, self.pic.mesh.dy, conductor = True)
                
                inner = -1 if boundary.type.split(sep= "-")[0] == "Inner " else 1
                base_field[boundary.location,1] *= numpy.where(boundary.directions == 0, 1*inner, 1)
                base_field[boundary.location,0] *= numpy.where(boundary.directions == 3, 1*inner, 1)
                base_field[boundary.location,0] *= numpy.where(boundary.directions == 1, -1*inner, 1)
                base_field[boundary.location,1] *= numpy.where(boundary.directions == 2, -1*inner, 1)
        #Induced charge is calculated
        charge[acc_index: acc_index+len(loc)] = af.induced_charge_p(base_field[loc,0], base_field[loc,1], self.pic.mesh.area_sat, c.EPS_0)
        acc_index += len(loc)

        #induced charge is calculed for children meshes' nodes
        for child in self.children:
            child.induced_charge(rho, charge = charge, acc_index = acc_index)
        #Returning induced charge for all the nodes
        return charge

    def floating_potential(self, species, rho):
        loc = self.pic.mesh.overall_location_sat
        #Charge induced by the electric field
        induced_charge = self.induced_charge(rho)
        #Charge acumulated in the surface
        charges = [specie.q*specie.mesh_values.accDensity*self.pic.mesh.volumes[loc] for specie in species]
        charge = reduce(lambda x,y: x+y, charges)
        #For later distribution of surface charges in electron specie
        electron = list(filter(lambda specie: specie.name == "Electron - Solar wind", species))[0]
        #Computation of potential and charge/density distribution
        phi_c, d_q, d_n = af.floating_potential_p(self.inv_capacity, self.capacity, charge-induced_charge, self.pic.mesh.volumes[loc], electron.q)
        #I/O
        max_d_q = numpy.max(d_q)
        max_d_q = max_d_q if max_d_q > 0 else 1.0
        print("Satellite potential: ", phi_c, flush = True)
        print("Sum of d_q over max d_q: ", numpy.abs(numpy.sum(d_q)/max_d_q),"Sum of d_q: ", numpy.sum(d_q), flush = True)
        #assert abs(numpy.sum(d_q)) < -c.QE or numpy.abs(numpy.sum(d_q)/numpy.max(d_q)) < 1e-1, "The redistribution of charge is creating or eliminating charge"
        ##NOTE: Only for diagnostics
        #new_potential = self.inv_capacity@charge
        #print("max, min of new potential: ", numpy.max(new_potential), numpy.min(new_potential))
        print("max, min of charge: ", numpy.max(charge), numpy.min(charge))
        print("max, min of induced charge: ", numpy.max(induced_charge), numpy.min(induced_charge))
        print("sum q and sum normal: ", numpy.sum(charge-induced_charge))
        #print("sum capacity: ", numpy.sum(self.capacity))
        #print("sum inv capacity: ", numpy.sum(self.inv_capacity))

        #Assign to 'Electron - Solar wind' the task of recording the changes in density in the conductive surface
        #electron.mesh_values.accDensity += d_n
        #Assigning the value to the surface nodes
        self.assignValuesToArray_recursive('potential', loc, phi_c*numpy.ones_like(loc))
        #Set same potential for interior nodes
        self.assignValuesToInteriorNodes(phi_c)

##NOTE: Floating potential with selection over one mesh
#    def floating_potential(self, species):
#        loc = self.pic.mesh.overall_location_sat
#        charges = [specie.q*specie.mesh_values.accDensity*self.pic.mesh.volumes[loc] for specie in species]
#        charge = reduce(lambda x,y: x+y, charges)
#        #For later distribution of surface charges in electron specie
#        electron = list(filter(lambda specie: specie.name == "Electron - Solar wind", species))[0]
#        #Computation of potential and charge/density distribution
#        #TODO: let's try
#        sel = len(self.pic.mesh.children[0].children[0].children[0].location_sat)
#        phi_c, d_q, d_n = af.floating_potential_p(self.inv_capacity[-sel:,-sel:], self.capacity[-sel:,-sel:], charge[-sel:], self.pic.mesh.volumes[loc][-sel:], electron.q)
#        #I/O
#        max_d_q = numpy.max(d_q)
#        max_d_q = max_d_q if max_d_q > 0 else 1.0
#        print("Satellite potential: ", phi_c, flush = True)
#        print("Sum of d_q over max d_q: ", numpy.abs(numpy.sum(d_q)/max_d_q),"Sum of d_q: ", numpy.sum(d_q), flush = True)
#        #assert abs(numpy.sum(d_q)) < -c.QE or numpy.abs(numpy.sum(d_q)/numpy.max(d_q)) < 1e-1, "The redistribution of charge is creating or eliminating charge"
#        #NOTE: Only for diagnostics
#        #new_potential = self.inv_capacity@charge
#        #print("max, min of new potential: ", numpy.max(new_potential), numpy.min(new_potential))
#        print("max, min of charge: ", numpy.max(charge), numpy.min(charge))
#        print("sum_q and sum_q_sel: ", numpy.sum(charge), numpy.sum(charge[-sel:]))
#        print("sum_capacity: ", numpy.sum(self.capacity))
#        print("sum_capacity_sel: ", numpy.sum(self.capacity[-sel:,-sel:]))
#        print("sum_inv_capacity: ", numpy.sum(self.inv_capacity))
#        print("sum_inv_capacity_sel: ", numpy.sum(self.inv_capacity[-sel:,-sel:]))
#
#        #Assign to 'Electron - Solar wind' the task of recording the changes in density in the conductive surface
#        #electron.mesh_values.accDensity += d_n
#        #Assigning the value to the surface nodes
#        self.assignValuesToArray_recursive('potential', loc, phi_c*numpy.ones_like(loc))
#        #Set same potential for interior nodes
#        self.assignValuesToInteriorNodes(phi_c)

# Puts the values of the boundaries to the children of the mesh, so that they it can be used for computeField in these meshes.
    def assignChildrenBoundaries(self, border = False, interpolate = False):
        for child in self.children:
            #NOTE: This part of the code was included to avoid doing the scatter process over the satellite boundaries, since those nodes
            #   have already the potential stored from the Capacity Matrix method.
            border_i = numpy.zeros((0), dtype = numpy.uint32)
            for boundary in child.pic.mesh.boundaries:
                if (boundary.type.split(sep="-")[0] == "Outer ") or (border and boundary.type.split(sep="-")[0] == "Inner "):
                    border_i = numpy.append(border_i, boundary.location, axis = 0)
            if interpolate:
                border_i = numpy.append(border_i, child.ind_calc)
            pos = super(Mesh_recursive, child.pic.mesh).getPosition(border_i)
            child.potential[border_i] = super(PIC_recursive, self.pic).gather(pos, self.potential[:,None])[:,0]

#   +assignValueToInteriorNodes(double value) = Assign 'value' to all the nodes in the meshes that are the interior of an inner boundary.
    def assignValuesToInteriorNodes(self, value):
        for boundary in self.pic.mesh.boundaries:
            if boundary.type == 'Inner - 2D_Rectangular':
                self.potential[boundary.ind_inner] = value
        for child in self.children:
            child.assignValuesToInteriorNodes(value)

    def saveVTK(self, mesh):
        dic = super().saveVTK(mesh)
        dic[self.type+"-potential"] = mesh.vtkOrdering(self.getTotalArray('potential'))
        return dic

    #Fix this. The 'field' value from the vtr file is not coming with the information of all the meshes
    def loadVTK(self, mesh, output):
        temp = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(self.type+"-potential")))
        self.assignValuesToArray_recursive("potential", numpy.arange(self.pic.mesh.accPoints, dtype ='uint'), temp)
        super().loadVTK(mesh, output)


class Constant_Electric_Field_recursive(Field_recursive, Constant_Electric_Field):
    def __init__(self, n_pic, field_dim, n_children, n_root):
        super(Field_recursive, self).__init__(n_pic, field_dim)
        self.field *= 0.0
        super().__init__(n_children, n_root, self.type)

    def computeField(self, species, rho = None, border = False, interpolate = False, adjustment = False):
        pass

        
#Time_Electric_Field(Electric_Field):
#
#Definition = Electric field dependent on time.
#Attributes:
#	+type (string) = "Electric field - Constant".
#	+Electric_Field attributes.
#Methods:
#	+Electric_Field methods.
#       +computeField(Species species, int p_step, int e_step = 0) = Recieves the steps in the simulation and computes the time from the start of the execution. Then, updates the field accordingly
#           with this and any function imposed by the user inside of this method.
class Time_Electric_Field(Electric_Field):
    def __init__(self, n_pic, field_dim):
        super().__init__(n_pic, field_dim, " - Constant")
        #Initial electric field
        self.field[:,0] += 0.27992

    def computeField(self, species, p_step, e_step = 0):
        time = p_step*c.P_DT+e_step*c.E_DT
        gradient = 98.4658
        self.field[:,0] = 0.27992+gradient*time


#Magnetic_Field (Inherits from Field):
#
#Definition = Magnetic field
#Attibutes:
#	+Field attributes.
#Methods:
#	+Field methods.
class Magnetic_Field(Field):
    def __init__(self, n_pic, field_dim, n_string):
        super().__init__(n_pic, field_dim, "Magnetic"+n_string)

    def computeField(self, species):
        pass

#Constant_Magnetic_Field(Inherits from Magnetic_Field):
#
#Definition = Constant Magnetic field impsoed by the user. Does not change through time. It works as a perpendicular field to the 2D dominion of the electric field and particular. Thus, field_dim = 1.
#Attributes:
#	+type (string) = "Magnetic - Constant".
#	+Magnetic_Field attributes.
#Methods:
#	+Magnetic_Field methods.
class Constant_Magnetic_Field(Magnetic_Field):
    def __init__(self, n_pic, field_dim):
        super().__init__(n_pic, field_dim, " - Constant")
        self.field += c.B_STRENGTH

    def computeField(self, species):
        pass

class Constant_Magnetic_Field_recursive(Field_recursive, Constant_Magnetic_Field):
    def __init__(self, n_pic, field_dim, n_children, n_root):
        super(Field_recursive, self).__init__(n_pic, field_dim)
        super().__init__(n_children, n_root, self.type)
