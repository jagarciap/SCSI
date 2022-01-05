import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy
import pdb

import constants as c

delta_max = 2.5
E_max = 300*(-c.QE)

def see_yield(E, theta):
    return 1.114*delta_max/numpy.cos(theta)*(E_max/E)**0.35*(1-numpy.exp(-2.28*numpy.cos(theta)*(E/E_max)**1.35))

def see_yield_cos(E, cos_theta):
    return 1.114*delta_max/cos_theta*(E_max/E)**0.35*(1-numpy.exp(-2.28*cos_theta*(E/E_max)**1.35))

def graph_yield():
    theta = numpy.linspace(0, numpy.pi*1.1/2, endpoint = True)
    E = numpy.linspace(20, 400, endpoint = True)*(-c.QE)
    theta_2D = numpy.repeat(theta[:,None], len(E), axis = 1).T
    E_2D = numpy.repeat(E[:,None], len(theta), axis = 1)

    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(E_2D/(-c.QE), theta_2D/numpy.pi*180, see_yield(E_2D, theta_2D))
    plt.xlabel("Energy (ev)")
    plt.ylabel("Angle of incidence respect to normal (°)")
    plt.title("See_yield")
    plt.show()

def see_yield_to_file():
    cos_theta = numpy.arange(0, 1.005, 0.005, dtype = numpy.float)
    E = numpy.arange(20, 405, 5, dtype = numpy.float)*(-c.QE)
    cos_theta_2D = numpy.repeat(cos_theta[:,None], len(E), axis = 1).T
    E_2D = numpy.repeat(E[:,None], len(cos_theta), axis = 1)
    matrix = see_yield_cos(E_2D, cos_theta_2D)

    #Little correction because theta=90deg behaves abnormally (due to divison by zero)
    line1 = see_yield(E, 89*numpy.pi/180)
    line2 = see_yield(E, 91*numpy.pi/180)
    correction = (line1+line2)/2
    matrix[:,0] = correction

    filename = "./data/SEE_yield.dat"
    header_file = "----------------------------------------------------------------\n"+\
             "SEE_yield function\n"+\
             "\tThe rows represent energies, from 20eV (top) to 400eV (bottom) with increases of 5eV\n"+\
             " \tThe columns represent cosines of angles of incidence, from 0.000 (left) to 1.000 (right) with increases of 0.005\n"+\
             "----------------------------------------------------------------\n"
    numpy.savetxt(filename, matrix, header = header_file)

def valuesToIndices(E, cos_theta, matrix):
    E_bot = 20
    E_top = 400
    new_E = E/(-c.QE)
    new_E = numpy.where(new_E < E_bot, E_bot, new_E)
    new_E = numpy.where(new_E > E_top, E_top, new_E)
    i = numpy.floor_divide(new_E-E_bot, 5).astype(numpy.uint8)
    j = numpy.floor_divide(cos_theta, 0.005).astype(numpy.uint8)
    return matrix[i,j]
