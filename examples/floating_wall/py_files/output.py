# File in charge of printing
from datetime import datetime
import numpy
import os
import pdb
import pickle

import constants as c
import motion
from timing import Timing

#       +Method that prepares the system to be printed in a '.vtk' file.
#       +It receives in args[0] the timestep, and the rest of args are objects with functions saveVTK that provide dictionaries of the attributes to be stored in the file.
#       +The actual printing is handled by the mesh.
@Timing
def saveVTK(mesh, sys_dic, keys, filename_ext = ''):
    #Preparing file
    cwd = os.path.split(os.getcwd())[0]
    vtkstring = os.path.join(cwd,'results','ts{:05d}{}'.format(sys_dic[keys[0]], filename_ext))
    #Creating dictionary
    dic = {}
    for key in keys[1:]:
        dic.update(sys_dic[key].saveVTK(mesh))
    #Executing through mesh
    mesh.saveVTK(vtkstring, dic)

#       +Method that loads the information of the system from a '.vtk' and stores it in the arguments *args.
#       +Structure to be followed in *args:
#       ++ts (timestep); fields: Electrics, Magnetics; Species: Electrons, Protons, Ions, Neutrals.
#       ++Inside the types not further specified now, an alphabetical order with respect to the classes' names will be maintained.
def loadVTK(filename, mesh, sys_dic, keys):
    #Preparing path
    cwd = os.path.split(os.getcwd())[0]
    filename = os.path.join(cwd,'initial_conditions',filename)
    reader = mesh.vtkReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    for key in keys[1:]:
        sys_dic[key].loadVTK(mesh, output)


# The function prints a file for a particular timestep 'ts' where the species being tracked are printed. Columns are for each component of each species, so for 2D:
#   specie1.x \t specie1.y \t specie2.x etc. Each row is a different particle for a particular species.
@Timing
def particleTracker(ts, *args):
    # Checking tracking method
    for spc in args:
        if spc.part_values.current_n > spc.part_values.num_tracked and numpy.any(spc.part_values.trackers == spc.part_values.max_n):
            print("Error in species: ", spc.name)
            print(spc.part_values.current_n, spc.part_values.num_tracked)
            pdb.set_trace()
            raise ValueError("There should not be any invalid values")

    #Creating array to be printed and the header
    narray = numpy.zeros((args[0].part_values.num_tracked, args[0].pos_dim*len(args)))
    nHeader = ''
    for i in range(len(args)):
        ind = numpy.flatnonzero(args[i].part_values.trackers != args[i].part_values.max_n)
        narray[ind, args[i].pos_dim*i:args[i].pos_dim*(i+1)] = args[i].part_values.position[args[i].part_values.trackers[ind],:]
        nHeader += args[i].name + '\t'

    cwd = os.path.split(os.getcwd())[0]
    workfile = os.path.join(cwd, 'particle_tracker', 'ts={:05d}.dat'.format(ts))
    nHeader = 'No. of particles = {:d} \n'.format(args[0].part_values.num_tracked)+nHeader
    numpy.savetxt(workfile, narray , fmt = '%.5e', delimiter = '\t', header = nHeader)


#       +Method that stores the information of the system, given in *args, in a '.pkl' file. See 'Pickle' module for further information.
#       +Structure to be followed in *args:
#       ++ts (timestep); fields: Electrics, Magnetics; Species: Electrons, Protons, Ions, Neutrals; part_solver (Particle Solver).
#       ++Inside the types not further specified now, an alphabetical order with respect to the classes' names will be maintained.
def savePickle(sys_dic, keys):
    #Creating file's name
    time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    cwd = os.path.split(os.getcwd())[0]
    string = os.path.join(cwd,'previous_executions','sys_ts={:d}_'.format(sys_dic[keys[0]])+time+'.pkl')
    #Storing information
    with open (string, 'wb') as output:
        for key in keys:
            pickle.dump(sys_dic[key], output, -1)


#       +Structure to be followed in *args:
#       ++ts (timestep); fields: Electrics, Magnetics; Species: Electrons, Protons, Ions, Neutrals; part_solver (Particle Solver).
#       ++Inside the types not further specified now, an alphabetical order with respect to the classes' names will be maintained.
def loadPickle(filename, sys_dic, keys):
    #Preparing path
    cwd = os.path.split(os.getcwd())[0]
    filename = os.path.join(cwd,'initial_conditions',filename)
    with open (filename, 'rb') as pinput:
        for key in keys:
            sys_dic[key] = pickle.load(pinput)


#       +saveParticlesTXT(dict sys_dic, [String] keys) = Stores the values of positions and velocities of the different species at a certain time.
#           The file is created with the format "ts{step number}.dat" and stores the information in columns as:
#           columns for position, followed by columns for velocity, one of these blocks for each species, ordered in alphabetical order.
@Timing
def saveParticlesTXT(sys_dic, keys):
    #Preparing path
    cwd = os.path.split(os.getcwd())[0]
    filename = os.path.join(cwd,'results_particles','ts{:05d}.dat'.format(sys_dic[keys[0]]))
    species_id = []
    attributes = []
    arrays = []
    for key in keys[1:]:
        n_id, n_att, n_array = sys_dic[key].saveParticlesTXT()
        species_id.append(n_id)
        attributes.append(n_att)
        if key == keys[1]:
            arrays = n_array
        else:
            diff = arrays.shape[0]-n_array.shape[0]
            if diff < 0:
                fill = numpy.zeros((-diff, arrays.shape[1]))
                arrays = numpy.append(arrays, fill, axis = 0)
            else:
                fill = numpy.zeros((diff, n_array.shape[1]))
                n_array = numpy.append(n_array, fill, axis = 0)
            arrays = numpy.append(arrays, n_array, axis = 1)
    first_row = '\t'.join(species_id)
    second_row = '\t'.join(attributes)
    nHeader = first_row+'\n'+second_row
    numpy.savetxt(filename, arrays, fmt = '%+.6e', delimiter = '\t', header = nHeader)


#       +saveTimes(int ts, dict dictionary) = Stores execution times in a file. If it is the first step in the simulation, the file is created. Otherwise,
#           values are appended at the end of the file.
def saveTimes(ts, dictionary):
    #File name
    cwd = os.path.split(os.getcwd())[0]
    filename = os.path.join(cwd,'results','execution_times.dat')
    if ts == 0:
        workFile = open(filename, 'w', buffering = 1) 
        time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
        workFile.write('#'+time+'\n')
        workFile.write('#'+'\t'.join(dictionary.keys())+'\n')
        workFile.write('{:d}'.format(ts)+'\t'+'\t'.join(map( lambda x : '{:.5e}'.format(x), dictionary.values()))+'\n')
    else:
        workFile = open(filename, 'a', buffering = 1) 
        workFile.write('{:d}'.format(ts)+'\t'+'\t'.join(map( lambda x : '{:.5e}'.format(x), dictionary.values()))+'\n')
    workFile.close()
