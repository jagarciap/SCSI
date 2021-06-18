#from itertools import zip
import copy
from functools import reduce
import numpy
import matplotlib.pyplot as plt
import os
import pdb
from scipy import stats, optimize
import sys
from matplotlib import rc
import time

sys.path.insert(0,'..')

import accelerated_functions as af
import constants as c
from mesh import Mesh_2D_rm_sat
from Boundaries.inner_2D_rectangular import Inner_2D_Rectangular
from Boundaries.outer_1D_rectangular import Outer_1D_Rectangular
from Boundaries.outer_2D_rectangular import Outer_2D_Rectangular
import vtr_to_numpy as vtn
from field import Constant_Magnetic_Field_recursive
from mesh_setup import mesh_file_reader
from Species.proton import Proton_SW
from Species.electron import Electron_SW, Photoelectron, Secondary_Emission__Electron
from Species.user_defined import User_Defined
import initial_conditions.satellite_condition as ic
from motion import Boris_Push
import output as out
from timing import Timing

plt.rc('text', usetex=True)
plt.rc('axes', linewidth=1.5)
plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']


#---------------------------------------------------------------------------------------------------------------------
# Creating mesh
#---------------------------------------------------------------------------------------------------------------------


#System:
#
#Definition = Is the class that contains every variable and class necessary for the simulation to be executed.
#Attributes:
#	+ts (int) = Timestep of the simulation.
#	+The rest of the variables will change with the simulation, but normally, there are:
#	+mesh (Mesh).
#	+pic (PIC).
#	+fields (Field) = Probably several of them.
#	+species (species) = Probably several of them.
#	+part_solver (Motion_Solver).
#Methods:
#	+Remark about init(): It will "declare" the attributes necessary for the simulation to run. The actual assignment of atributes
#		to instances of each class will occur during the 'initial condition' section of 'main.py'.
#	+arrangePickle() : Variable = Return a tuple of keys to iterate over when saving and loading of/from '.pkl' files, in the order required.
#	+arrangeVTK() : Variable = Return a tuple of keys to iterate over when saving and loading of/from VTK files, in the order required.
class System(object):
    def __init__(self):
        self.at = {}
        self.at['ts'] = 0
        #TODO: Change later
        self.at['mesh'], self.at['pic'], self.at['e_field'] = mesh_file_reader('2021_02_08.txt')
        self.at['mesh'].print()
        self.at['electrons'] = Electron_SW(0.0, c.E_SPWT, c.E_SIZE, c.DIM, c.DIM, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        self.at['photoelectrons'] = Photoelectron(c.PHE_T, c.PHE_FLUX, 0.0, c.PHE_SPWT, c.PHE_SIZE, c.DIM, c.DIM, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        self.at['see'] = Secondary_Emission__Electron(c.SEE_T, 0.0, c.SEE_SPWT, c.SEE_SIZE, c.DIM, c.DIM, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        self.at['protons'] = Proton_SW(0.0, c.P_SPWT, c.P_SIZE, c.DIM, c.DIM, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        #self.at['user'] = User_Defined(c.P_DT, -c.QE, c.MP, 0, c.P_SPWT, 1, c.DIM, c.DIM, self.at['mesh'].nPoints, 0, "1")
        self.at['m_field'] = Constant_Magnetic_Field_recursive(self.at['pic'], c.B_DIM, [], True)
        self.at['part_solver'] = Boris_Push(self.at['pic'], [self.at['electrons'].name, self.at['photoelectrons'].name, self.at['see'].name, self.at['protons'].name],\
                [self.at['electrons'].part_values.max_n, self.at['photoelectrons'].part_values.max_n, self.at['see'].part_values.max_n, self.at['protons'].part_values.max_n],\
                [self.at['electrons'].vel_dim, self.at['photoelectrons'].vel_dim, self.at['see'].vel_dim, self.at['protons'].vel_dim])

#Initialization of the system
system = System()

#---------------------------------------------------------------------------------------------------------------------
# Plotting functions
#---------------------------------------------------------------------------------------------------------------------

#def surface_charge_density_time(data, names, charges):
#    fig = plt.figure(figsize=(16,8))
#    loc = numpy.unique(mesh.boundaries[1].location)
#    net = numpy.zeros((len(data[names[0]][0,:])))
#    for name, charge in zip(names, charges):
#        d_loc = [numpy.flatnonzero(data[name][loc,j]) for j in range(numpy.shape(data[name])[1])]
#        arr = numpy.asarray([numpy.sum(data[name][loc[d_loc[j]],j])*charge/abs(charge) for j in range(numpy.shape(data[name])[1])])
#        arr[numpy.isnan(arr)] = 0
#        net += arr
#    
#    time = numpy.arange(len(data[names[0]][0,:]))*c.P_DT*100/1e-6
#    plt.plot(time, net)
#
#    avg = numpy.average(net[int(2*len(net)/3):])
#    print("Average density is: {:.4e} 1/m3".format(avg))
#    plt.axhline(y = avg)
#
#    plt.title(r'\textbf{Accumulated surface charge density}', fontsize = 24)
#    plt.ylabel(r'\textbf{Density (sign aware) [1/m$^{3}$]}', fontsize = 22)
#    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
#    plt.tick_params(axis='both', which='major', labelsize=20)
#    plt.gca().ticklabel_format(axis='y', style='sci')
#    plt.grid()
#    plt.show()

def total_surface_charge_time(mesh, data, total_charge, names, charges):
    loc = numpy.unique(mesh.boundaries[1].location)
    net = numpy.zeros((len(data[names[0]][0,:])))
    for name, charge in zip(names, charges):
        d_loc = [numpy.flatnonzero(data[name][loc,j]) for j in range(numpy.shape(data[name])[1])]
        arr = numpy.asarray([numpy.sum(data[name][loc[d_loc[j]],j]*charge*mesh.volumes[loc[d_loc[j]]]) for j in range(numpy.shape(data[name])[1])])
        arr[numpy.isnan(arr)] = 0
        net += arr
    
    time = numpy.arange(len(data[names[0]][0,:]))*c.P_DT*100/1e-6
    plt.plot(time, net, label = mesh.id)
    total_charge.append(net)

    avg = numpy.average(net[int(2*len(net)/3):])
    print("Average charge is: {:.4e} C".format(avg))
    plt.axhline(y = avg)


#---------------------------------------------------------------------------------------------------------------------
# Arrays to be uploaded
#---------------------------------------------------------------------------------------------------------------------

names = [\
        "Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"\
        ]

#---------------------------------------------------------------------------------------------------------------------
# Main function
#---------------------------------------------------------------------------------------------------------------------

def uploading_data(mesh, data_r = None):
    if data_r is None:
        data_r = []
    data = {}
    results = vtn.vtrToNumpy(mesh, vtn.loadFromResults(files_id = mesh.id), names)
    for name, array in zip(names, results):
        data[name] = array
    data_r.append(data)
    for child in mesh.children:
        uploading_data(child, data_r = data_r)
    return data_r

def location_indices(mesh, ind_list = None, acc = None):
    if ind_list is None:
        ind_list = []
        acc = [0]
    temp = acc[0]
    acc[0] += len(mesh.location_sat)
    ind_list.append([temp, acc[0]])
    for child in mesh.children:
        location_indices(child, ind_list = ind_list, acc = acc)
    return ind_list

def main():
    #Initializing and preparing data
    data = uploading_data(system.at['mesh'])
    ind = location_indices(system.at['mesh'])
    total_charges = []
    fig = plt.figure(figsize=(16,8))

    #Plot recursively
    def plot_charges(mesh, acc_ind = None):
        if mesh.root:
            acc_ind = [0]
        data_p = data[acc_ind[0]]
        acc_ind[0] += 1
        total_surface_charge_time(mesh, data_p, total_charges,\
                ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"],\
                [c.QE, c.QE, c.QE, -c.QE])
        for child in mesh.children:
            plot_charges(child, acc_ind = acc_ind)

    plot_charges(system.at['mesh'])

    #Plot post-process
    plt.title(r'\textbf{Total charge}', fontsize = 24)
    plt.ylabel(r'\textbf{Charge [C]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().ticklabel_format(axis='y', style='sci')
    plt.legend()
    plt.grid()
    plt.show()

    #Capacity and potential stuff
    #fig = plt.figure(figsize=(16,8))
    plt.spy(system.at['e_field'].capacity)
    plt.show()

    #Initializing and preparing data
    def plot_potentials(mesh, acc_ind = None):
        if mesh.root:
            acc_ind = [0]

        ind_p = ind[acc_ind[0]]
        inv_cap = system.at['e_field'].inv_capacity[ind_p[0]:ind_p[1], ind_p[0]:ind_p[1]]
        cap = system.at['e_field'].capacity[ind_p[0]:ind_p[1], ind_p[0]:ind_p[1]]
        cap_2 = numpy.linalg.inv(inv_cap)
        print(mesh.id, "inv_cap: ", '{:e}'.format(numpy.sum(inv_cap)), "cap: ", numpy.trace(cap), "cap_2: ", numpy.sum(cap_2))
        phi_c = total_charges[acc_ind[0]]/numpy.trace(system.at['e_field'].capacity)

        time = numpy.arange(len(phi_c))*c.P_DT*100/1e-6
        plt.plot(time, phi_c, label = mesh.id)

        ##Plotting matrices
        #plt.matshow(inv_cap)
        #plt.colorbar()
        #plt.show()
        #plt.matshow(cap)
        #plt.colorbar()
        #plt.show()

        #plt.plot(inv_cap[:,0])
        #plt.show()

        acc_ind[0] += 1
        for child in mesh.children:
            plot_potentials(child, acc_ind = acc_ind)

    plot_potentials(system.at['mesh'])
    print(numpy.sum(system.at['e_field'].capacity))

    #Plot post-process
    plt.title(r'\textbf{Spacecraft potential}', fontsize = 24)
    plt.ylabel(r'\textbf{Potential [V]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().ticklabel_format(axis='y', style='sci')
    plt.legend()
    plt.grid()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------
# Functions calls
#---------------------------------------------------------------------------------------------------------------------


#current_collected_time(data, ["Electron - Photoelectron-flux", "Electron - SEE-flux", "Electron - Solar wind-flux", "Proton - Solar wind-flux"])
#current_collected_time(data, ["Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux"])
#current_recollection_percentage_time(data, ["Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux"],\
#                                     ["Electron - Photoelectron-flux", "Electron - SEE-flux"])
main()
