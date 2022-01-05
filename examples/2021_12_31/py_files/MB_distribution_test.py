import matplotlib.pyplot as plt
import numpy
import os
import pdb

import constants as c

#       +sampleIsotropicVelocity([double] vth, [int] num) [double,double] = It receives an array of most probable speeds vth = \sqrt{2kT/m} and creates
#           for each speed num 2D velocities with their magnitudes following a Maxwellian distribution.
#       NOTE: This function needs to be revised. random should not spread throughout different cells, and should use the same seed for every function call.
#    @nb.vectorize(signature = nb.double[:], target='cpu')
def sampleIsotropicVelocity(vth, num):
    #Prepare for the handling of different sets of temperature
    ##NOTE: Delete later
    #vth = numpy.asarray([vth[0]])
    #num = int(1e5)
    total = numpy.sum(num)
    index = numpy.repeat(numpy.arange(len(vth)), num)
    #pick maxwellian velocities
    rand_spread = numpy.random.rand(total,6)
    vm_x = numpy.sqrt(2)*vth[index]*(rand_spread[:,0]+rand_spread[:,1]+rand_spread[:,2]-1.5)
    vm_y = numpy.sqrt(2)*vth[index]*(rand_spread[:,3]+rand_spread[:,4]+rand_spread[:,5]-1.5)
    ##NOTE: Delete later
    #val = plt.hist(vm, 40)
    #length = val[1][1]-val[1][0]
    #integral = length*numpy.sum(val[0])
    #A = val[0].max()
    #x = numpy.linspace( val[1].min(), val[1].max(), num = 50)
    #y = A*numpy.exp(-x*x/vth/vth)
    #plt.plot(x,y)
    #pdb.set_trace()
    #plt.show()
    #2D components of velocity 
    return numpy.append(vm_x[:,None], vm_y[:, None], axis = 1)

## ---------------------------------------------------------------------------------------------------------------
# Graph functions
## ---------------------------------------------------------------------------------------------------------------

def graph_vel_distribution(velocities, expected_MP, title = "Velocity distribution"):
    datamag = plt.hist(numpy.linalg.norm(velocities, axis = 1), 81, alpha=0.5, density = True, label="histogram")
    #datamag = plt.hist(velocities, 81, alpha=0.5, label="histogram")
    plt.axvline(x=numpy.sqrt(2/3)*expected_MP)
    plt.title(title, fontsize = 24)
    plt.legend()
    plt.show()

def checking_angular_distribution(velocities, title = "Angular distribution"):
    plt.figure(figsize=(10,10))
    plt.scatter(velocities[:,0]/numpy.linalg.norm(velocities, axis = 1), velocities[:,1]/numpy.linalg.norm(velocities, axis =1), marker = '.')
    plt.title(title, fontsize = 24)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()

## ---------------------------------------------------------------------------------------------------------------
# Tests
## ---------------------------------------------------------------------------------------------------------------

def checking_sampleIsotropicVelocity():
    vth = numpy.asarray([c.E_V_TH_MP])
    num = numpy.asarray([int(1e6)])
    velocities = sampleIsotropicVelocity(vth, num)
    graph_vel_distribution(velocities, c.E_V_TH_MP, title = "sampleIsotropicVelocity")
    checking_angular_distribution(velocities, title = "sampleIsotropicVelocity")

def loading_particles(filenames):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.path.split(os.getcwd())[0]
    data = []
    for filename in filenames:
        filename = cwd+'/results_particles/'+filename
        raw_data = numpy.loadtxt(filename, delimiter = '\t')
        dic = {}
        with open(filename, 'r') as f:
            labels = f.readline().strip(' #\n').split('\t')
        offset = 0
        for i in range(0, len(labels), 2):
            name = labels[i]
            np, pos_dim, vec_dim = labels[i+1].split('-')
            np = int(np)
            temp = offset+int(pos_dim)+int(vec_dim)
            dic[name] = raw_data[:np, offset:temp]
            offset = temp
        data.append(dic)
    return data

def select_by_position(data, pos, radius = 0.3):
    dim_pos = len(pos)
    pos = numpy.asarray(pos)
    ind = numpy.flatnonzero(numpy.linalg.norm(data[:,:dim_pos]-pos, axis = 1) < radius)
    return data[ind,:]

def check_ParticlesTXT():
    #Loading the files
    filenames = ['ts00599.dat']
    data_dic = loading_particles(filenames)[0]

    #Region to check and species to check
    species = 'Electron - Solar wind'
    pos = (0.6,0.6)
    filtered = select_by_position(data_dic[species], pos)
    print("# of particles: ", filtered.shape[0])

    #Graphs
    graph_vel_distribution(filtered[:,2:], c.E_V_TH_MP, title = species+"-"+str(pos))
    checking_angular_distribution(filtered[:,2:], title = species+"-"+str(pos))
    pdb.set_trace()

checking_sampleIsotropicVelocity()
check_ParticlesTXT()
