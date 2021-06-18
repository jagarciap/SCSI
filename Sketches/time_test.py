import copy
import numpy
import matplotlib.pyplot as plt
import pdb
import time

def time_detached(old_species, new_species, part_solver):
    t1 = time.time()
    temp = copy.deepcopy(old_species)
    t2 = time.time()
    temp.part_values.velocity[:temp.part_values.current_n,:] = (old_species.part_values.velocity[:old_species.part_values.current_n,:]+new_species.part_values.velocity[:new_species.part_values.current_n,:])/2
    t3 = time.time()
    part_solver.scatter(old_species.part_values.position[:old_species.part_values.current_n,:],temp.part_values.velocity[:temp.part_values.current_n,0], old_species.mesh_values.velocity[:,0])
    t4 = time.time()
    n = old_species.part_values.current_n
    print('Overall time', t4-t1)
    print('Creation of temp species', t2-t1)
    print('Number of particles', n)
    print('Synchronization per particle', (t3-t2)/n)
    print('Scatter per particle', (t4-t3)/n)
    return t4-t1, t2-t1, (t3-t2)/n, (t4-t3)/n

def time_joint(old_species, new_species, part_solver):
    t1 = time.time()
    part_solver.scatter(old_species.part_values.position[:old_species.part_values.current_n,:],\
            (old_species.part_values.velocity[:old_species.part_values.current_n,0]+new_species.part_values.velocity[:new_species.part_values.current_n,0])/2, old_species.mesh_values.velocity[:,0])
    t2 = time.time()
    n = old_species.part_values.current_n
    print('Number of particles', n)
    print('Overall time', t2-t1)
    print('Scatter per particle', (t2-t1)/n)
    return t2-t1, (t2-t1)/n

def comparison(old_species, new_species, part_solver):
    case1 = []
    for n in range(1000):
        case1.append(time_detached(old_species, new_species, part_solver))
    array1 = numpy.asarray(case1)
    numpy.savetxt('time_test_detached.dat', array1)
    case2 = []
    for n in range(1000):
        case2.append(time_joint(old_species, new_species, part_solver))
    array2 = numpy.asarray(case2)
    numpy.savetxt('time_test_joint.dat', array2)
    fig = plt.figure(figsize=(16,12))
    plt.plot(array1[:,0], label = 'detached')
    plt.plot(array2[:,0], label = 'joint')
    plt.title('Overall time')
    plt.legend()
    fig.savefig('overall_time.png')

    fig = plt.figure(figsize=(16,12))
    plt.plot(array1[:,3], label = 'detached')
    plt.plot(array2[:,1], label = 'joint')
    plt.title('Scatter per particle')
    plt.legend()
    fig.savefig('scatter_per_particle.png')

    fig = plt.figure(figsize=(16,12))
    plt.plot(array1[:,3]+array1[:,2], label = 'detached')
    plt.plot(array2[:,1], label = 'joint')
    plt.title('Total time per particle')
    plt.legend()
    fig.savefig('total_time_per_particle.png')
