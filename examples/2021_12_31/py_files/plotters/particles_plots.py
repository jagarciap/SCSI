#This file contains the functions used to plot the files in "results_particles"
import numpy
import matplotlib.pyplot as plt
import os
import pdb
from scipy import stats, optimize
from subprocess import check_output
import sys

import constants as c
from mesh import Mesh_2D_rm_sat

#---------------------------------------------------------------------------------------------------------------------
# Creating mesh
#---------------------------------------------------------------------------------------------------------------------

mesh = Mesh_2D_rm_sat(c.XMIN, c.XMAX, c.YMIN, c.YMAX, c.XMINSAT, c.XMAXSAT, c.YMINSAT, c.YMAXSAT, c.DX, c.DY, c.DEPTH)
temp = numpy.ones((mesh.nPoints), dtype = int)
temp[mesh.boundaries[1].location] -= 1
temp[mesh.boundaries[1].ind_inner] -= 1
temp = temp.astype(bool)

## ---------------------------------------------------------------------------------------------------------------
# Functions for handling Gaussian curves.
# SOURCE: 'DataAnalysis2.py' in '/home/jorge/Documents/Instituto_Balseiro/Semestre2017-1/Experimental_IV/Mediciones_en_coincidencia'.
#           There are addiional functions there that can be of use in the future.
## ---------------------------------------------------------------------------------------------------------------

def nGaussianFunc(x, *params):
    y = numpy.zeros_like(x,dtype=numpy.float)
    for i in range (0, len(params), 3):
        #print(params[i], params[i+1], params[i+2])
        y += params[i]*numpy.exp(-(x-params[i+1])**2/(2*params[i+2]**2))
    return y

def nGaussianAreas(*params):
    y = []
    for i in range(0,len(params),3):
        y.append(params[i]*numpy.sqrt(2*numpy.pi)*params[i+2])
    return y

def nGaussianFit (x, y, guess):
    return optimize.curve_fit(nGaussianFunc,x,y, p0=guess)

## ---------------------------------------------------------------------------------------------------------------

def loadFromResults():
    cwd_base = os.getcwd().rsplit(sep = os.sep, maxsplit = 1)
    cwd = os.path.join(cwd_base[0], 'results_particles','')
    stdout = check_output('ls' +' {}'.format(cwd), shell=True)
    files = stdout.decode().split(sep='\n')
    for i in range(len(files)):
        files[i] = os.path.join(cwd, files[i])
    return files[:-1]

def phase_space(filename, x, y, vely):
    array = numpy.loadtxt(filename)
    fig = plt.figure(figsize=(8,5))
    #plt.scatter(array[:,0]+array[:,1], array[:,3], marker = '.', label = 'electrons') 
    plt.scatter(array[:,x]+array[:,y], array[:,vely], marker = '.', label = 'protons') 
    plt.legend()
    plt.title('y_Vel-Pos phase space')
    plt.show()
    #fig.savefig('phase_space_vely_pos_2.png')

def vel_distribution_electrons(filename, velx, vely):
    array = numpy.loadtxt(filename)
    ind = numpy.flatnonzero(array[:,velx])
    array[ind,velx] -= c.E_V_SW
    fig = plt.figure(figsize=(8,5))
    #datax = plt.hist(array[:,2],81, alpha = 0.5, label = 'electrons, vel_x') 
    #datay = plt.hist(array[:,3],81, alpha = 0.5, label = 'electrons, vel_y') 
    datamag = plt.hist(numpy.linalg.norm(array[ind, velx:vely+1], axis = 1), 81, alpha = 0.5, label = 'electrons, vel_mag')
    plt.axvline(x=numpy.sqrt(2/3)*c.E_V_TH_MP)
    ##Gaussian fits
    #params, errors = nGaussianFit(datax[1][:-1], datax[0], [datax[0].max(), 3e5, numpy.sqrt(c.K*c.E_T/c.ME)])
    #x = numpy.linspace(datax[1].min(), datax[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'velx_fit')
    #print("electrons_vel_x")
    #print(params, errors)
    #print("Temperature_vel_x", params[2]**2*c.ME/c.K/c.EV_TO_K)
    #params, errors = nGaussianFit(datay[1][:-1], datay[0], [datay[0].max(), 0, numpy.sqrt(c.K*c.E_T/c.ME)])
    #x = numpy.linspace(datay[1].min(), datay[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vely_fit')
    #print("electrons_vel_y")
    #print(params, errors)
    #print("Temperature_vel_y", params[2]**2*c.ME/c.K/c.EV_TO_K)

    plt.legend()
    plt.title('vel_distribution_electrons')
    plt.show()
    #fig.savefig('vel_distribution_electrons.png')

def vel_distribution_protons(filename, velx, vely):
    array = numpy.loadtxt(filename)
    ind = numpy.flatnonzero(array[:,velx])
    fig = plt.figure(figsize=(8,5))
    #datax = plt.hist(array[:,6],81, alpha = 0.5, label = 'protons, vel_x') 
    #datay = plt.hist(array[:,7],81, alpha = 0.5, label = 'protons, vel_y') 
    datamag = plt.hist(numpy.linalg.norm(array[ind, velx:vely+1], axis = 1), 81, alpha = 0.5, label = 'protons, vel_mag')
    ##Gaussian fits
    #params, errors = nGaussianFit(datax[1][:-1], datax[0], [datax[0].max(), 3e5, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datax[1].min(), datax[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_x_fit')
    #print("protons_vel_x")
    #print(params, errors)
    #print("Temperature_vel_x", params[2]**2*c.MP/c.K/c.EV_TO_K)
    #params, errors = nGaussianFit(datay[1][:-1], datay[0], [datay[0].max(), 0, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datay[1].min(), datay[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_y_fit')
    #print("protons_vel_y")
    #print(params, errors)
    #print("Temperature_vel_y", params[2]**2*c.MP/c.K/c.EV_TO_K)

    plt.legend()
    plt.title('vel_distribution_protons')
    plt.show()
    #fig.savefig('vel_distribution_protons.png')

def vel_non_drifted_distribution_protons(filename, velx, vely):
    array = numpy.loadtxt(filename)
    ind = numpy.flatnonzero(array[:,velx])
    array[ind,velx] -= c.P_V_SW
    fig = plt.figure(figsize=(8,5))
    #datax = plt.hist(array[:,6],81, alpha = 0.5, label = 'protons, vel_x') 
    #datay = plt.hist(array[:,7],81, alpha = 0.5, label = 'protons, vel_y') 
    datamag = plt.hist(numpy.linalg.norm(array[ind, velx:vely+1], axis = 1), 81, alpha = 0.5, label = 'protons, vel_mag')
    plt.axvline(x=numpy.sqrt(2/3)*c.P_V_TH_MP)
    ##Gaussian fits
    #params, errors = nGaussianFit(datax[1][:-1], datax[0], [datax[0].max(), 3e5, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datax[1].min(), datax[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_x_fit')
    #print("protons_vel_x")
    #print(params, errors)
    #print("Temperature_vel_x", params[2]**2*c.MP/c.K/c.EV_TO_K)
    #params, errors = nGaussianFit(datay[1][:-1], datay[0], [datay[0].max(), 0, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datay[1].min(), datay[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_y_fit')
    #print("protons_vel_y")
    #print(params, errors)
    #print("Temperature_vel_y", params[2]**2*c.MP/c.K/c.EV_TO_K)

    plt.legend()
    plt.title('vel_distribution_protons')
    plt.show()
    #fig.savefig('vel_distribution_protons.png')

def vel_distribution_photoelectrons(filename, velx, vely):
    array = numpy.loadtxt(filename)
    ind = numpy.flatnonzero(array[:,velx])
    fig = plt.figure(figsize=(8,5))
    #datax = plt.hist(array[:,6],81, alpha = 0.5, label = 'protons, vel_x') 
    #datay = plt.hist(array[:,7],81, alpha = 0.5, label = 'protons, vel_y') 
    datamag = plt.hist(numpy.linalg.norm(array[ind, velx:vely+1], axis = 1), 81, alpha = 0.5, label = 'phe, vel_mag')
    plt.axvline(x=numpy.sqrt(2/3)*c.PHE_V_TH_MP)
    ##Gaussian fits
    #params, errors = nGaussianFit(datax[1][:-1], datax[0], [datax[0].max(), 3e5, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datax[1].min(), datax[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_x_fit')
    #print("protons_vel_x")
    #print(params, errors)
    #print("Temperature_vel_x", params[2]**2*c.MP/c.K/c.EV_TO_K)
    #params, errors = nGaussianFit(datay[1][:-1], datay[0], [datay[0].max(), 0, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datay[1].min(), datay[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_y_fit')
    #print("protons_vel_y")
    #print(params, errors)
    #print("Temperature_vel_y", params[2]**2*c.MP/c.K/c.EV_TO_K)

    plt.legend()
    plt.title('vel_distribution_photoelectrons')
    plt.show()
    #fig.savefig('vel_distribution_protons.png')

def vel_distribution_see(filename, velx, vely):
    array = numpy.loadtxt(filename)
    ind = numpy.flatnonzero(array[:,velx])
    fig = plt.figure(figsize=(8,5))
    #datax = plt.hist(array[:,6],81, alpha = 0.5, label = 'protons, vel_x') 
    #datay = plt.hist(array[:,7],81, alpha = 0.5, label = 'protons, vel_y') 
    datamag = plt.hist(numpy.linalg.norm(array[ind, velx:vely+1], axis = 1), 81, alpha = 0.5, label = 'see, vel_mag')
    plt.axvline(x=numpy.sqrt(2/3)*c.SEE_V_TH_MP)
    ##Gaussian fits
    #params, errors = nGaussianFit(datax[1][:-1], datax[0], [datax[0].max(), 3e5, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datax[1].min(), datax[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_x_fit')
    #print("protons_vel_x")
    #print(params, errors)
    #print("Temperature_vel_x", params[2]**2*c.MP/c.K/c.EV_TO_K)
    #params, errors = nGaussianFit(datay[1][:-1], datay[0], [datay[0].max(), 0, numpy.sqrt(c.K*c.P_T/c.MP)])
    #x = numpy.linspace(datay[1].min(), datay[1].max(), num = 100)
    #plt.plot(x, nGaussianFunc(x, *params,), label = 'vel_y_fit')
    #print("protons_vel_y")
    #print(params, errors)
    #print("Temperature_vel_y", params[2]**2*c.MP/c.K/c.EV_TO_K)

    plt.legend()
    plt.title('vel_distribution_secondary emission electrons')
    plt.show()
    #fig.savefig('vel_distribution_protons.png')

def particle_positions(filenames, x, y):
    fig = plt.figure(figsize=(12,12))
    for filename in filenames:
        array = numpy.loadtxt(filename)
        #file_n = open(filename, 'r')
        #line = file_n.readline()
        #np = int(line.split('\t')[3].split('-')[0])
        #file_n.close()
        #name = filename.rsplit('/', 1)[-1].split('.')[0][2:]
        #Plot
        plt.scatter(array[:,x], array[:,y], marker = 'o', s = 8.0, zorder = 5, label = filename)
        #plt.scatter(array[:np,x], array[:np,y], marker = '.', s = 1.0, label = filename)
        #print(name, np)
    #Plot boundaries
    plt.xlim(mesh.xmin, mesh.xmax)
    plt.ylim(mesh.ymin, mesh.ymax)
    DY = (mesh.ymax-mesh.ymin)
    DX = (mesh.xmax-mesh.xmin)
    plt.axvline(x = mesh.boundaries[1].xmin, ymin = (mesh.boundaries[1].ymin-mesh.ymin)/DY, ymax = (mesh.boundaries[1].ymax-mesh.ymin)/DY, color = 'black')
    plt.axvline(x = mesh.boundaries[1].xmax, ymin = (mesh.boundaries[1].ymin-mesh.ymin)/DY, ymax = (mesh.boundaries[1].ymax-mesh.ymin)/DY, color = 'black')
    plt.axhline(y = mesh.boundaries[1].ymin, xmin = mesh.boundaries[1].xmin/DX, xmax = mesh.boundaries[1].xmax/DY, color = 'black')
    plt.axhline(y = mesh.boundaries[1].ymax, xmin = mesh.boundaries[1].xmin/DX, xmax = mesh.boundaries[1].xmax/DY, color = 'black')
    plt.legend()
    plt.show()


#particle_positions(loadFromResults()[70:72])
#vel_distribution_electrons(loadFromResults()[-1], 2, 3)
#vel_distribution_photoelectrons(loadFromResults()[-1], 6, 7)
#vel_distribution_see(loadFromResults()[-1], 10, 11)
#vel_distribution_protons(loadFromResults()[-1], 14, 15)
#vel_non_drifted_distribution_protons(loadFromResults()[-1], 14, 15)
#phase_space(loadFromResults()[0])

particle_positions(["Outer - 2D_Rectangular_Proton - Solar wind.txt"], 0, 1)
particle_positions(["Outer - 2D_Rectangular_Electron - Solar wind.txt"], 0, 1)
particle_positions(["Inner - 2D_Rectangular_Electron - Photoelectron.txt"], 0, 1)
particle_positions(["Inner - 2D_Rectangular_Electron - SEE.txt"], 0, 1)
vel_distribution_electrons("Outer - 2D_Rectangular_Electron - Solar wind.txt", 2, 3)
vel_distribution_photoelectrons("Inner - 2D_Rectangular_Electron - Photoelectron.txt", 2, 3)
vel_distribution_see("Inner - 2D_Rectangular_Electron - SEE.txt", 2, 3)
vel_distribution_protons("Outer - 2D_Rectangular_Proton - Solar wind.txt", 2, 3)
vel_non_drifted_distribution_protons("Outer - 2D_Rectangular_Proton - Solar wind.txt", 2, 3)
