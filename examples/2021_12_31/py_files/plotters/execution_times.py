#File aimed to plot exeuction times of the simulation and optimization-related plot#File aimed to plot exeuction times of the simulation and optimization-related plotss
import numpy
import matplotlib.pyplot as plt
import os
import pdb
import sys

sys.path.insert(0,'..')

import constants as c

#Setting up parameters for graphs
plt.rc('text', usetex=True)
plt.rc('axes', linewidth=1.5)
plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#Function that loads the dataset as a numpy array
def load_file(name = 'execution_times.dat'):
    cwd_base = os.getcwd().rsplit(sep = os.sep, maxsplit = 1)
    cwd = os.path.join(cwd_base[0], 'results','')
    filename = cwd+name
    data_loop = numpy.loadtxt(filename, skiprows = 2, usecols = range(51))
    return data_loop

#Global time per step
def time_per_step(*data, col = 50):
    fig = plt.figure(figsize=(12,8))
    for i in range (len(data)):
        plt.plot(data[i][:,0], data[i][:,col], marker = '.', label = '{:d}'.format(i))
    plt.ylabel(r'\textbf{Time\,per\,step [s]}', fontsize = 22)
    plt.xlabel(r'\textbf{Steps}', fontsize = 22)
    plt.legend(fontsize = 20)
    plt.axhline(y=0, color = 'k')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.show()
    
#Cumulative time along execution
def cumulative_time(*data, col = 50, steps_jump = 10):
    fig = plt.figure(figsize=(12,8))
    for i in range(len(data)):
        plt.plot(data[i][:,0], [sum(data[i][:j,col]*steps_jump)/3600 for j in range(1,len(data[i][:,0])+1)], marker = '.', label = '{:d}'.format(i))
    #plt.ylim(ymax = numpy.max(data[0][int(len(data[0][:,0])/2):,3])/1e-3, ymin = numpy.min(data[0][int(len(data[0][:,0])/2):,3])/1e-3)
    plt.ylabel(r'\textbf{Cumulative\,time [h]}', fontsize = 22)
    plt.xlabel(r'\textbf{Steps}', fontsize = 22)
    plt.legend(fontsize = 20)
    plt.axhline(y=0, color = 'k')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.show()

#Fractions of time spent in each part of the step
def fractions_of_time(*data, global_col = 50):
    fig = plt.figure(figsize=(12,8))
    st_part_e = 1
    st_mesh_e = 2
    st_adv_e = 3
    st_inject_e = 4
    e_total_f = 4
    e_steps = 10
    part_p = 41
    mesh_p = 42
    adv_p = 43
    inject_p = 44
    mesh_total_e = 44
    mesh_total_p = 46
    saveVTK = 47
    saveParticles = 48
    saveTracker = 49
    copy = 50
    for i in range(len(data)):
        plt.plot(data[i][:,0], 100*numpy.sum(data[i][:,st_part_e:st_part_e+e_steps*e_total_f:e_total_f], axis = 1)/data[i][:,global_col], marker = '.', label = 'part\_adv\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*numpy.sum(data[i][:,st_mesh_e:st_mesh_e+e_steps*e_total_f:e_total_f], axis = 1)/data[i][:,global_col], marker = '.', label = 'mesh\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*numpy.sum(data[i][:,st_adv_e:st_adv_e+e_steps*e_total_f:e_total_f], axis = 1)/data[i][:,global_col], marker = '.', label = 'adv\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*numpy.sum(data[i][:,st_inject_e:st_inject_e+e_steps*e_total_f:e_total_f], axis = 1)/data[i][:,global_col], marker = '.', label = 'inject\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*(numpy.sum(data[i][:,st_adv_e:st_adv_e+e_steps*e_total_f:e_total_f], axis = 1)+\
                numpy.sum(data[i][:,st_inject_e:st_inject_e+e_steps*e_total_f:e_total_f]))/data[i][:,global_col], marker = '.', label = 'total\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,part_p]/data[i][:,global_col], marker = '.', label = 'part\_adv\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,mesh_p]/data[i][:,global_col], marker = '.', label = 'mesh\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,adv_p]/data[i][:,global_col], marker = '.', label = 'adv\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,inject_p]/data[i][:,global_col], marker = '.', label = 'inject\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*numpy.sum(data[i][:,adv_p:inject_p+1], axis = 1)/data[i][:,global_col], marker = '.', label = 'total\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,mesh_total_e]/data[i][:,global_col], marker = '.', label = 'mesh\_total\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,mesh_total_p]/data[i][:,global_col], marker = '.', label = 'mesh\_total\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,saveVTK]/data[i][:,global_col], marker = '.', label = 'saveVTK-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,saveParticles]/data[i][:,global_col], marker = '.', label = 'saveParticles-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,saveTracker]/data[i][:,global_col], marker = '.', label = 'saveTracker-{:d}'.format(i))
        plt.plot(data[i][:,0], 100*data[i][:,copy]/data[i][:,global_col], marker = '.', label = 'copy-{:d}'.format(i))
    plt.ylim(ymin=0, ymax = 100)
    plt.ylabel(r'\textbf{Fractions of time per step [s]}', fontsize = 22)
    plt.xlabel(r'\textbf{steps}', fontsize = 22)
    plt.legend(fontsize = 14)
    plt.axhline(y=0, color = 'k')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.show()

def comparison_per_particle_step(*data):
    fig = plt.figure(figsize=(12,8))
    st_part_e = 1
    st_mesh_e = 2
    st_adv_e = 3
    st_inject_e = 4
    e_total_f = 4
    e_steps = 10
    part_p = 41
    mesh_p = 42
    adv_p = 43
    inject_p = 44
    for i in range(len(data)):
        plt.plot(data[i][:,0], numpy.average(data[i][:,st_part_e:st_part_e+e_steps*e_total_f:e_total_f], axis = 1), marker = '.', label = 'part\_adv\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], numpy.average(data[i][:,st_mesh_e:st_mesh_e+e_steps*e_total_f:e_total_f], axis = 1), marker = '.', label = 'mesh\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], numpy.average(data[i][:,st_adv_e:st_adv_e+e_steps*e_total_f:e_total_f], axis = 1), marker = '.', label = 'adv\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], numpy.average(data[i][:,st_inject_e:st_inject_e+e_steps*e_total_f:e_total_f], axis = 1), marker = '.', label = 'inject\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], numpy.average(data[i][:,st_adv_e:st_adv_e+e_steps*e_total_f:e_total_f], axis = 1)+\
                numpy.average(data[i][:,st_inject_e:st_inject_e+e_steps*e_total_f:e_total_f], axis = 1), marker = '.', label = 'total\_e-{:d}'.format(i))
        plt.plot(data[i][:,0], data[i][:,part_p], marker = '.', label = 'part\_adv\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], data[i][:,mesh_p], marker = '.', label = 'mesh\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], data[i][:,adv_p], marker = '.', label = 'adv\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], data[i][:,inject_p], marker = '.', label = 'inject\_p-{:d}'.format(i))
        plt.plot(data[i][:,0], numpy.sum(data[i][:,adv_p:inject_p+1], axis = 1), marker = '.', label = 'total\_p-{:d}'.format(i))
    plt.ylabel(r'\textbf{Time [s]}', fontsize = 22)
    plt.xlabel(r'\textbf{steps}', fontsize = 22)
    plt.legend(fontsize = 14)
    plt.axhline(y=0, color = 'k')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.show()

#Execution
data = load_file()
time_per_step(data)
cumulative_time(data)
fractions_of_time(data)
comparison_per_particle_step(data)
