import numpy
import matplotlib.pyplot as plt
import os
import pdb
from vtk.util.numpy_support import vtk_to_numpy

import constants as c
from mesh import Mesh_2D_rm
from pic import PIC_2D_rm1o
import solver as slv

def initialCondition(mesh, pot):
    def potAtInd(ind):
        return (0 - 20)/(mesh.nx-1)*ind+20+numpy.random.rand()*2.7
    for i in range(mesh.nPoints):
        pot[i] = potAtInd(i%mesh.nx)


class Field(object):
    def __init__(self, n_name, n_pic, field_dim):
        self.name = n_name
        self.pic = n_pic
        self.field = numpy.zeros((self.pic.mesh.nPoints, field_dim))

    def computeField(self, species):
        pass

    def fieldAtParticles(self, position):
        return self.pic.gather(position, self.field)

    def saveVTK(self, mesh):
        return {self.name+"-field" : mesh.vtkOrdering(self.field)}

    def loadVTK(self, mesh, output):
        self.field = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(self.name+"-field")))

class Electric_Field(Field):
    def __init__(self, n_pic, field_dim, n_string):
        self.potential = numpy.zeros((n_pic.mesh.nPoints))
        #initialCondition(n_pic.mesh, self.potential)
        super().__init__("Electric"+n_string,n_pic, field_dim)

    def computeField(self, species):
        pass

    def saveVTK(self, mesh):
        dic = super().saveVTK(mesh)
        dic[self.name+"-potential"] = mesh.vtkOrdering(self.potential)
        return dic

    def loadVTK(self, mesh, output):
        self.potential = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(self.name+"-potential")))
        super().loadVTK(mesh, output)

class Electrostatic_2D_rm(Electric_Field):
    def __init__(self, n_pic, field_dim):
        super().__init__(n_pic, field_dim, " - Electrostatic_2D_rm")

    def computeField(self, rho):
        for boundary in self.pic.mesh.boundaries:
            boundary.applyElectricBoundary(self)
        slv.poissonSolver_2D_rm_SORCA(self.pic.mesh, self.potential, rho)
        self.field = -slv.derive_2D_rm(self.pic.mesh, self.potential)
        for boundary in self.pic.mesh.boundaries:
            boundary.applyElectricBoundary(self)

#       +Computation of Dirichlet boundary condition at every node in location ([ind]). Every row in value ([double]) corresponds to one node in location.
    def dirichlet(self, location, values):
        #Dirichlet
        #for i in location:
        #    if i%c.NX == 0:
        #        self.potential[i] = 20
        #    elif i%c.NX == c.NX-1:
        #        self.potential[i] = 0
        self.potential[location] = values
        #Electric field trough Pade 2nd order in the boundaries
        self.field[location, :] = -slv.derive_2D_rm_boundaries(location, self.pic.mesh, self.potential)

def saveVTK(mesh, rho, e_field):
    #Preparing file
    cwd = os.path.split(os.getcwd())[0]
    vtkstring = cwd+'/results/solver_tester'
    #Creating dictionary
    dic = e_field.saveVTK(mesh)
    dic["rho"] = rho
    #Executing through mesh
    mesh.saveVTK(vtkstring, dic)



mesh = Mesh_2D_rm(c.XMIN, c.XMAX, c.YMIN, c.YMAX, c.NX, c.NY, c.DEPTH)
print('{:e}'.format(c.P_SPWT/mesh.volumes[12]))
pdb.set_trace()
pic = PIC_2D_rm1o(mesh)
e_field = Electrostatic_2D_rm(pic, c.DIM)
rho = numpy.zeros((c.NX*c.NY))
#for i in range(mesh.nPoints):
#    index = i%mesh.nx
#    if index <= 25 and index >= 15 and i//mesh.nx >= 15 and i//mesh.nx <= 25:
#        rho[i] += -1e3*c.QE/c.EPS_0/mesh.volumes[i]
#rho[mesh.boundaries[0].location] = -1e3*c.QE/c.EPS_0/mesh.volumes[mesh.boundaries[0].location]
#rho[int(mesh.ny/2*mesh.nx+mesh.nx/2)] = -1e12*c.QE/c.EPS_0/mesh.volumes[int(mesh.ny/2*mesh.nx+mesh.nx/2)]
rho[int(mesh.ny/2*mesh.nx+mesh.nx/2)] = -15
pdb.set_trace()
e_field.computeField(rho)
saveVTK(mesh, rho, e_field)
