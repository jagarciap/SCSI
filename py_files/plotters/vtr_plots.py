#from itertools import zip
from functools import reduce
import numpy
import matplotlib.pyplot as plt
import os
import pdb
from scipy import stats, optimize
import sys
from matplotlib import rc

sys.path.insert(0,'..')

import constants as c
from mesh import Mesh_2D_rm_sat
from Boundaries.inner_2D_rectangular import Inner_2D_Rectangular
from Boundaries.outer_1D_rectangular import Outer_1D_Rectangular
from Boundaries.outer_2D_rectangular import Outer_2D_Rectangular
import vtr_to_numpy as vtn

plt.rc('text', usetex=True)
plt.rc('axes', linewidth=1.5)
plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#---------------------------------------------------------------------------------------------------------------------
# Creating mesh
#---------------------------------------------------------------------------------------------------------------------
## 0-0-0-0
#outer = Outer_2D_Rectangular(7.0, 12.2, -2.6, 2.6, 'space')
#inner = Inner_2D_Rectangular(9.0, 10.2, -0.6, 0.6, 'satellite')
#mesh = Mesh_2D_rm_sat(7.0, 12.2, -2.6, 2.6, 9.0, 10.2, -0.6, 0.6, 0.02, 0.02, 1.2, [outer, inner])
#temp = numpy.arange(mesh.nPoints, dtype = numpy.uint32)
# 0-0-0
outer = Outer_2D_Rectangular(5.0, 19.2, -6.0, 6.0, 'space')
inner = Inner_2D_Rectangular(9.0, 16.2, -1.8, 1.8, 'satellite')
mesh = Mesh_2D_rm_sat(5.0, 19.2, -6.0, 6.0, 9.0, 16.2, -1.8, 1.8, 0.04, 0.04, 0.875, [outer, inner])
temp = numpy.arange(mesh.nPoints, dtype = numpy.uint32)
#temp = numpy.delete(temp, numpy.append(mesh.boundaries[1].location, mesh.boundaries[1].ind_inner))
temp = numpy.delete(temp, mesh.boundaries[1].ind_inner)

##Establishing a minimum of superparticles for a cell
## The minimum of required population of super particles will be 2 particles at the center of a cell with the maximum volume in the mesh
#spwt_min = 2*c.E_SPWT*0.5*0.5/numpy.max(mesh.volumes)
spwt_min = 0

#---------------------------------------------------------------------------------------------------------------------
# Plotting functions
#---------------------------------------------------------------------------------------------------------------------

def mean_temperature_time(name, density, accDensity):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    data_d = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), density)
    data_acc = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), accDensity)
    data_d = [data_i-data_acc_i for data_i, data_acc_i in zip(data_d, data_acc)]
    for i in range(len(name)):
        d_loc = [numpy.flatnonzero(data_d[i][temp,j]) for j in range(numpy.shape(data_d[i])[1])]
        arr = [numpy.average(data[i][temp[d_loc[j]],j]) for j in range(numpy.shape(data[i])[1])]
        arr = [0 if numpy.isnan(x) else x for x in arr]
        time = numpy.arange(len(data[i][0,:]))*c.P_DT*10/1e-6
        plt.plot(time, arr, label = name[i])

    plt.title(r'\textbf{Average temperature in domain}', fontsize = 24)
    plt.ylabel(r'\textbf{Temperature [eV]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.yscale('log')
    plt.grid()
    plt.legend(fontsize = 20)
    #plt.ylim(5e1, 1.5e2)
    plt.legend()
    plt.show()

def mean_vel_time(name, density, accDensity, ind = None):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    data_d = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), density)
    data_acc = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), accDensity)
    data_d = [data_i-data_acc_i for data_i, data_acc_i in zip(data_d, data_acc)]
    for i in range(len(name)):
        if ind is None:
            ind = temp
        d_loc = [numpy.flatnonzero(data_d[i][ind,j]) for j in range(numpy.shape(data_d[i])[-1])]
        arr = [numpy.average(numpy.linalg.norm(data[i][ind[d_loc[j]],:,j], axis = 1)) for j in range(numpy.shape(data[i])[-1])]
        arr = [0 if numpy.isnan(x) else x for x in arr]
        time = numpy.arange(len(data[i][0,0,:]))*c.P_DT*10/1e-6
        plt.plot(time, arr, label = name[i])

    plt.title(r'\textbf{Average velocity in domain}', fontsize = 24)
    plt.ylabel(r'\textbf{Velocity [m/s]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def mean_density_time(name, accDensity):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    data_acc = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), accDensity)
    data = [data_i-data_acc_i for data_i, data_acc_i in zip(data, data_acc)]
    for i in range(len(name)):
        d_loc = [numpy.flatnonzero(data[i][temp,j]) for j in range(numpy.shape(data[i])[1])]
        arr = [numpy.average(data[i][temp[d_loc[j]],j]) for j in range(numpy.shape(data[i])[1])]
        arr = [0 if numpy.isnan(x) else x for x in arr]
        time = numpy.arange(len(data[i][0,:]))*c.P_DT*10/1e-6
        plt.plot(time, arr, label = name[i])

    plt.title(r'\textbf{Average density in domain}', fontsize = 24)
    plt.ylabel(r'\textbf{Density [m$^{-3}$]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def extremes_density_time(name):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    arr_min = numpy.min(data[1][temp]-data[0][temp], axis = 0)
    arr_max = numpy.max(data[1][temp]-data[0][temp], axis = 0)
    plt.plot(arr_min, label = 'min')
    plt.plot(arr_max, label = 'max')
    plt.legend()
    plt.show()

def current_collected_time(data, names):
    fig = plt.figure(figsize=(16,8))
    loc = numpy.unique(mesh.boundaries[1].location)
    for name in names:
        d_loc = [numpy.flatnonzero(data[name][loc,j]) for j in range(numpy.shape(data[name])[1])]
        arr = [numpy.sum(data[name][loc[d_loc[j]],j]*mesh.area_sat[d_loc[j]])/1e-3 for j in range(numpy.shape(data[name])[1])]
        arr = [0 if numpy.isnan(x) else x for x in arr]
        #plt.plot(numpy.sum(data[i][loc]*numpy.repeat(mesh.volumes[loc]/(mesh.dx/2), numpy.shape(data[i])[1], axis = 1), axis = 0), label = name[i])
        time = numpy.arange(len(data[name][0,:]))*c.P_DT*100/1e-6
        plt.plot(time, arr, label = name.replace('_', '\_'))

        avg = numpy.average(arr[int(2*len(arr)/3):])
        print("Average {} current is: {:.4e}".format(name,avg))
        plt.axhline(y = avg)

    plt.title(r'\textbf{Currents to/from satellite}', fontsize = 24)
    plt.ylabel(r'\textbf{Current [mA]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def current_recollection_percentage_time(data, names_out, names_in):
    fig = plt.figure(figsize=(16,8))
    loc = numpy.unique(mesh.boundaries[1].location)
    for name_out, name_in in zip(names_out, names_in):
        #Out
        d_loc = [numpy.flatnonzero(data[name_out][loc,j]) for j in range(numpy.shape(data[name_out])[1])]
        arr = [numpy.sum(data[name_out][loc[d_loc[j]],j]*mesh.area_sat[d_loc[j]])/1e-3 for j in range(numpy.shape(data[name_out])[1])]
        arr_out = [0 if numpy.isnan(x) else x for x in arr]
        arr_out = numpy.asarray(arr_out)
        #In
        d_loc = [numpy.flatnonzero(data[name_in][loc,j]) for j in range(numpy.shape(data[name_in])[1])]
        arr = [numpy.sum(data[name_in][loc[d_loc[j]],j]*mesh.area_sat[d_loc[j]])/1e-3 for j in range(numpy.shape(data[name_in])[1])]
        arr_in = [0 if numpy.isnan(x) else x for x in arr]
        arr_in = numpy.asarray(arr_in)
        #plt.plot(numpy.sum(data[i][loc]*numpy.repeat(mesh.volumes[loc]/(mesh.dx/2), numpy.shape(data[i])[1], axis = 1), axis = 0), label = name[i])
        time = numpy.arange(len(data[name_out][0,:]))*c.P_DT*100/1e-6
        plt.plot(time, numpy.abs(arr_in/arr_out)*100, label = name_out.replace('_', '\_'))

        avg = numpy.average(numpy.abs(arr_in/arr_out)[int(2*len(arr_in)/3):])*100
        print("Average {} percentage recollection is: {:.4e}".format(name_out,avg))
        plt.axhline(y = avg)

    plt.title(r'\textbf{Recolletion percentage satellite}', fontsize = 24)
    plt.ylabel(r'\textbf{Percentage of recollection}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def out_in_currents_collected_time(name_out, name_in):
    fig = plt.figure(figsize=(16,8))
    data_out = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name_out)
    data_in = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name_in)
    loc = numpy.unique(mesh.boundaries[1].location)
    for i in range(len(name_out)):
        d_loc_out = [numpy.flatnonzero(data_out[i][loc,j]) for j in range(numpy.shape(data_out[i])[1])]
        d_loc_in = [numpy.flatnonzero(data_in[i][loc,j]) for j in range(numpy.shape(data_in[i])[1])]
        arr_out = [numpy.sum(data_out[i][loc[d_loc_out[j]],j]*mesh.area_sat[d_loc_out[j]])/1e-3 for j in range(numpy.shape(data_out[i])[1])]
        arr_out = [0 if numpy.isnan(x) else x for x in arr_out]
        arr_in = [numpy.sum(data_in[i][loc[d_loc_in[j]],j]*mesh.area_sat[d_loc_in[j]])/1e-3 for j in range(numpy.shape(data_in[i])[1])]
        arr_in = [0 if numpy.isnan(x) else -x for x in arr_in]
        #plt.plot(numpy.sum(data[i][loc]*numpy.repeat(mesh.volumes[loc]/(mesh.dx/2), numpy.shape(data[i])[1], axis = 1), axis = 0), label = name[i])
        time = numpy.arange(len(data_out[i][0,:]))*c.P_DT*100/1e-6
        plt.plot(time, arr_out, label = name_out[i].replace('_', '\_'))
        plt.plot(time, arr_in, label = name_in[i].replace('_', '\_'))

    plt.title(r'\textbf{Currents to/from satellite}', fontsize = 24)
    plt.ylabel(r'\textbf{Current [mA]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

#NOTE: It only works for species that its charge is c.QE
def approx_velocity_wall(current, accDensity):
    fig = plt.figure(figsize=(16,8))
    data_c = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), current)
    data_acc = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), accDensity)
    loc = numpy.unique(mesh.boundaries[1].location)
    for i in range(len(current)):
        data_acc[i][:,1:] -= data_acc[i][:,:-1]
        d_loc = [numpy.flatnonzero(data_acc[i][loc,j] > 0) for j in range(numpy.shape(data_acc[i])[1])]
        arr = [numpy.average(data_c[i][loc[d_loc[j]],j]/c.QE/data_acc[i][loc[d_loc[j]],j]) for j in range(numpy.shape(data_c[i])[1])]
        arr = [0 if numpy.isnan(x) else abs(x) for x in arr]
        time = numpy.arange(len(data_c[i][0,:]))*c.P_DT*10/1e-6
        plt.plot(time, arr, label = current[i].replace('_','\_'))

    plt.title(r'\textbf{Average impact velocity to/from satellite}', fontsize = 24)
    plt.ylabel(r'\textbf{Velocity [m/s]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.yscale('log')
    plt.legend(fontsize = 20)
    plt.show()

    
def debye_function(x, *args):
    q0 = -c.QE*1.6e12*(c.YMAX-c.YMIN)/(c.NY-1)*(c.XMAX-c.XMIN)/(c.NX-1)*c.DEPTH
    return args[0]*1/4/numpy.pi/c.EPS_0*q0/abs(x)*numpy.exp(-abs(x)/args[1])

def debye_shielding_fit(x, y, guess):
    return optimize.curve_fit(debye_function,x,y, p0=guess)

def debye_length_test(name):
    fig = plt.figure(figsize=(16,8))
    #Preparing data
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    left = c.NX*(int(c.NY/2)+1)
    cut = data[0][left:left+c.NX,-1]
    x = numpy.linspace(c.XMIN, c.XMAX, num = c.NX)
    offset = (c.XMAX-c.XMIN)/2
    dx = (c.XMAX-c.XMIN)/(c.NX-1)
    #Omitting the center
    ind = numpy.where(numpy.logical_or(x < offset-4*dx, x > offset+4*dx))

    #Debye Lenght study
    lambda_d = 1/numpy.sqrt(c.E_N*c.QE*c.QE/c.EPS_0/c.K/c.E_T+c.P_N*c.QE*c.QE/c.EPS_0/c.K/c.P_T)
    print("Theory", lambda_d)
    theory = debye_function(x-offset, 1.0, lambda_d)
    guess = (1.0, lambda_d)
    params, errors = debye_shielding_fit(x[ind]-offset, cut[ind], guess)

    plt.plot(x, theory, color = 'black', label = 'Theory')
    plt.scatter(x, cut, color = 'red', marker = '.', label = 'Simulation')
    plt.plot(x, debye_function(x-offset, *params), color = 'blue', label = 'Fit')
    plt.legend()
    plt.show()

    #Further analysis
    fig = plt.figure(figsize=(16,8))
    val = []
    err = []
    for i in range(1,10):
        ind = numpy.where(numpy.logical_or(x < offset-i*dx, x > offset+i*dx))
        guess = (1.0, lambda_d)
        params, errors = debye_shielding_fit(x[ind]-offset, cut[ind], guess)
        val.append(params[1])
        err.append(numpy.sqrt(numpy.diag(errors)[1]))
        print(params, numpy.sqrt(numpy.diag(errors)))
    plt.axhline(y=lambda_d, color = 'black')
    plt.errorbar(numpy.arange(1,10), val, yerr=err, marker = '.', linestyle = '', ecolor = 'black')
    plt.show()

def wall_potential(name):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)[0]
    top = numpy.unique(mesh.boundaries[1].top)
    left = numpy.unique(mesh.boundaries[1].left)
    right = numpy.unique(mesh.boundaries[1].right)
    bottom = numpy.unique(mesh.boundaries[1].bottom)
    plt.plot(numpy.arange(len(top)), data[top,-1], label = 'top')
    plt.plot(numpy.arange(len(top)), data[left,-1], label = 'left')
    plt.plot(numpy.arange(len(top)), data[right,-1], label = 'right')
    plt.plot(numpy.arange(len(top)), data[bottom,-1], label = 'bottom')
    plt.title(r'\textbf{Potential at the different borders of the satellite}', fontsize = 24)
    plt.ylabel(r'\textbf{Potential [V]}', fontsize = 22)
    plt.xlabel(r'\textbf{Borders}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def wall_density_diff(name):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    data = data[1]-data[0]
    top = numpy.unique(mesh.boundaries[1].top)
    left = numpy.unique(mesh.boundaries[1].left)
    right = numpy.unique(mesh.boundaries[1].right)
    bottom = numpy.unique(mesh.boundaries[1].bottom)
    plt.plot(numpy.arange(len(top)), data[top,-1], label = 'top')
    plt.plot(numpy.arange(len(top)), data[left,-1], label = 'left')
    plt.plot(numpy.arange(len(top)), data[right,-1], label = 'right')
    plt.plot(numpy.arange(len(top)), data[bottom,-1], label = 'bottom')
    plt.legend()
    plt.show()

def number_particles(data, names, spwts):
    fig = plt.figure(figsize=(16,8))
    for name, spwt in zip(names, spwts):
        d_loc = [numpy.flatnonzero(data[name][temp,j]) for j in range(numpy.shape(data[name])[1])]
        arr = [numpy.sum(data[name][temp[d_loc[j]],j]*mesh.volumes[temp[d_loc[j]]])/spwt for j in range(1, numpy.shape(data[name])[1])]
        arr = [0 if numpy.isnan(x) else x for x in arr]
        plt.plot(arr, label = name)
    plt.title(r'\textbf{Number of super particles}', fontsize = 24)
    plt.ylabel(r'\textbf{No. of SP}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def number_particles_per_cell(data, names, spwts):
    fig = plt.figure(figsize=(16,8))
    for name, spwt in zip(names, spwts):
        d_loc = [numpy.flatnonzero(data[name][temp,j]) for j in range(numpy.shape(data[name])[1])]
        arr = [numpy.min(data[name][temp[d_loc[j]],j]*mesh.volumes[temp[d_loc[j]]])/spwt for j in range(1, numpy.shape(data[name])[1])]
        arr = [0 if numpy.isnan(x) else x for x in arr]
        plt.plot(arr, label = name)
    plt.title(r'\textbf{Number of super particles per cell}', fontsize = 24)
    plt.ylabel(r'\textbf{No. of SP per cell}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

def increase_number_particles(name):
    fig = plt.figure(figsize=(16,8))
    data = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), name)
    for i in range(len(name)):
        arr = numpy.sum(data[i][temp]*mesh.volumes[temp][:,None], axis = 0)
        time = numpy.arange(len(arr[:-1]))*c.P_DT*10/1e-6
        plt.plot(time, arr[1:]-arr[:-1], label = name[i])
        #plt.plot(arr[1:]-arr[:-1], label = name[i])
    plt.axhline(y=3.310585e9, color = 'black', label='Photoelectron')
    plt.axhline(y=0, color = 'k')
    #plt.axhline(y=1.416000e11, color = 'red', label='electron-proton')
    plt.ylabel(r'\textbf{$\delta$ particles}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()

#The arguments will be received as tuples, as many as the species entered, and each tuple will be (charge, density, temperature).
def debye_length(*args):
    lambda_d = numpy.zeros_like(args[0][1])
    for i in range(len(args)):
        lambda_d += args[i][0]*args[i][0]*args[i][1]/c.EPS_0/c.K/args[i][2]
    lambda_d = 1/numpy.sqrt(lambda_d)
    return lambda_d

def debye_length_analysis(density, accDensity, temperature,\
        proton_density = ["Proton - Solar wind-density"], proton_accDensity = ["Proton - Solar wind-accumulated density"], proton_temperature = ["Proton - Solar wind-temperature"], n = 20):
    dictionary = {}

    data_p_density = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), proton_density)
    data_p_acc = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), proton_accDensity)
    data_p_density = [data_i-data_acc_i for data_i, data_acc_i in zip(data_p_density, data_p_acc)][0]
    data_p_temperature = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), proton_temperature)[0]

    data_e_density = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), density)
    data_e_acc = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), accDensity)
    data_e_density = [data_i-data_acc_i for data_i, data_acc_i in zip(data_e_density, data_e_acc)]
    data_e_temperature = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), temperature)

    #Average over the last n stored results
    ts = numpy.shape(data_p_density)[1]
    data_p_density = numpy.average(data_p_density[:,ts-n:], axis = 1)
    data_p_temperature = numpy.average(data_p_temperature[:,ts-n:], axis = 1)*c.EV_TO_K

    for i in range(len(density)):
        #Average over the last n stored results
        data_e_density_i = numpy.average(data_e_density[i][:,ts-n:], axis = 1)
        data_e_temperature_i = numpy.average(data_e_temperature[i][:,ts-n:], axis = 1)*c.EV_TO_K

        #Debye Length calcualtion
        lambda_d = debye_length((c.QE, data_e_density_i, data_e_temperature_i), (-c.QE, data_p_density, data_p_temperature))
        dictionary["Debye length - {}".format(density[i].rsplit("-", 1)[0])] = lambda_d

    #Create minimum Debye length case
    compact_l = list(dictionary.values())
    compact = reduce(lambda x, y: numpy.append(x, y[:,None], axis = 1), compact_l[1:], compact_l[0][:,None])
    compact = numpy.nanmin(compact, axis = 1)
    dictionary["Debye length - smallest"] = compact

    #Export to vtr files
    cwd = os.getcwd()
    vtkstring = os.path.join(cwd,'plotters','plots','debye_length')
    mesh.saveVTK(vtkstring, dictionary)

def surface_charge_density_time(data, names, charges):
    fig = plt.figure(figsize=(16,8))
    time = numpy.arange(len(data[names[0]][0,:]))*c.P_DT*100/1e-6
    loc = numpy.unique(mesh.boundaries[1].location)
    for name, charge in zip(names, charges):
        d_loc = [numpy.flatnonzero(data[name][loc,j]) for j in range(numpy.shape(data[name])[1])]
        arr = numpy.asarray([numpy.sum(data[name][loc[d_loc[j]],j])*charge/abs(charge) for j in range(numpy.shape(data[name])[1])])
        arr[numpy.isnan(arr)] = 0
        plt.plot(time, arr, label = name.replace('_','\_'))

    #avg = numpy.average(net[int(2*len(net)/3):])
    #print("Average density is: {:.4e} 1/m3".format(avg))
    #plt.axhline(y = avg)

    plt.title(r'\textbf{Accumulated surface charge density}', fontsize = 24)
    plt.ylabel(r'\textbf{Density (sign aware) [1/m$^{3}$]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().ticklabel_format(axis='y', style='sci')
    plt.grid()
    plt.show()

def net_surface_charge_density_time(data, names, charges):
    fig = plt.figure(figsize=(16,8))
    loc = numpy.unique(mesh.boundaries[1].location)
    net = numpy.zeros((len(data[names[0]][0,:])))
    for name, charge in zip(names, charges):
        d_loc = [numpy.flatnonzero(data[name][loc,j]) for j in range(numpy.shape(data[name])[1])]
        arr = numpy.asarray([numpy.sum(data[name][loc[d_loc[j]],j])*charge/abs(charge) for j in range(numpy.shape(data[name])[1])])
        arr[numpy.isnan(arr)] = 0
        net += arr
    
    time = numpy.arange(len(data[names[0]][0,:]))*c.P_DT*100/1e-6
    plt.plot(time, net)

    avg = numpy.average(net[int(2*len(net)/3):])
    print("Average density is: {:.4e} 1/m3".format(avg))
    plt.axhline(y = avg)

    plt.title(r'\textbf{Accumulated surface charge density}', fontsize = 24)
    plt.ylabel(r'\textbf{Density (sign aware) [1/m$^{3}$]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().ticklabel_format(axis='y', style='sci')
    plt.grid()
    plt.show()

def total_surface_charge_time(data, names, charges):
    fig = plt.figure(figsize=(16,8))
    loc = numpy.unique(mesh.boundaries[1].location)
    net = numpy.zeros((len(data[names[0]][0,:])))
    for name, charge in zip(names, charges):
        d_loc = [numpy.flatnonzero(data[name][loc,j]) for j in range(numpy.shape(data[name])[1])]
        arr = numpy.asarray([numpy.sum(data[name][loc[d_loc[j]],j]*charge*mesh.volumes[loc[d_loc[j]]]) for j in range(numpy.shape(data[name])[1])])
        arr[numpy.isnan(arr)] = 0
        net += arr
    
    time = numpy.arange(len(data[names[0]][0,:]))*c.P_DT*100/1e-6
    plt.plot(time, net)

    avg = numpy.average(net[int(2*len(net)/3):])
    print("Average charge is: {:.4e} C".format(avg))
    plt.axhline(y = avg)

    plt.title(r'\textbf{Total charge}', fontsize = 24)
    plt.ylabel(r'\textbf{Charge [C]}', fontsize = 22)
    plt.xlabel(r'\textbf{Time [$\mu$s]}', fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().ticklabel_format(axis='y', style='sci')
    plt.grid()
    plt.show()


#---------------------------------------------------------------------------------------------------------------------
# Extracting data
#---------------------------------------------------------------------------------------------------------------------

data ={}
names = [\
        "Electron - Photoelectron-flux", "Electron - SEE-flux", "Electron - Solar wind-flux", "Proton - Solar wind-flux",\
        "Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux",\
        "Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"\
        ]
results = vtn.vtrToNumpy(mesh, vtn.loadFromResults(), names)
for name, array in zip(names, results):
    data[name] = array

#---------------------------------------------------------------------------------------------------------------------
# Functions calls
#---------------------------------------------------------------------------------------------------------------------


#increase_number_particles(["Electron - Photoelectron-density"])
#increase_number_particles(["Electron - Photoelectron-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#                            [c.PHE_SPWT, c.E_SPWT, c.P_SPWT])
#extremes_density_time(["Electron - Solar wind-density", "Proton - Solar wind-density"])
#debye_length_analysis(["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density"],\
#                      ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density"],\
#                      ["Electron - Photoelectron-temperature", "Electron - SEE-temperature", "Electron - Solar wind-temperature"])
#mean_density_time(["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#                  ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"])
#number_particles(data, ["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#                 [c.PHE_SPWT, c.SEE_SPWT, c.E_SPWT, c.P_SPWT])
#number_particles_per_cell(data, ["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#                          [c.PHE_SPWT, c.SEE_SPWT, c.E_SPWT, c.P_SPWT])
#mean_temperature_time(["Electron - Photoelectron-temperature", "Electron - SEE-temperature", "Electron - Solar wind-temperature", "Proton - Solar wind-temperature"],\
#                      ["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#                      ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"])
#mean_vel_time(["Electron - Photoelectron-velocity", "Electron - SEE-velocity", "Electron - Solar wind-velocity", "Proton - Solar wind-velocity"],\
#              ["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#              ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"])
#mean_vel_time(["Electron - Photoelectron-velocity", "Electron - SEE-velocity", "Electron - Solar wind-velocity", "Proton - Solar wind-velocity"],\
#              ["Electron - Photoelectron-density", "Electron - SEE-density", "Electron - Solar wind-density", "Proton - Solar wind-density"],\
#              ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"],\
#              ind = numpy.unique(mesh.boundaries[1].location))
###debye_length_test(["Electric - Electrostatic_2D_rm-potential"])
#wall_potential(["Electric - Electrostatic_2D_rm_sat-potential"])
###wall_density_diff(["Electron - Solar wind-density", "Proton - Solar wind-density"])
current_collected_time(data, ["Electron - Photoelectron-flux", "Electron - SEE-flux", "Electron - Solar wind-flux", "Proton - Solar wind-flux"])
current_collected_time(data, ["Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux"])
current_recollection_percentage_time(data, ["Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux"],\
                                     ["Electron - Photoelectron-flux", "Electron - SEE-flux"])
##approx_velocity_wall(["Electron - Photoelectron-flux", "Electron - SEE-flux", "Electron - Solar wind-flux", "Proton - Solar wind-flux"],\
##                     ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"])
##approx_velocity_wall(["Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux"],\
##                     ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density"])
#out_in_currents_collected_time(["Electron - Photoelectron-outgoing_flux", "Electron - SEE-outgoing_flux"],\
#                               ["Electron - Photoelectron-flux", "Electron - SEE-flux"])
#surface_charge_density_time(["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"],\
#                            [c.QE, c.QE, c.QE, -c.QE])
net_surface_charge_density_time(data, ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"],\
                            [c.QE, c.QE, c.QE, -c.QE])
total_surface_charge_time(data, ["Electron - Photoelectron-accumulated density", "Electron - SEE-accumulated density", "Electron - Solar wind-accumulated density", "Proton - Solar wind-accumulated density"],\
                            [c.QE, c.QE, c.QE, -c.QE])
