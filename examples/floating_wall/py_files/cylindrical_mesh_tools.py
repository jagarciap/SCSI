import numpy
import matplotlib.pyplot as plt
import pdb

# Creates "num" number of positions between rmin and rmax, distributed with a linear distribution normalized between rmin and rmax
def linearDistribution_sampling(num, rmin, rmax):
    pos = []
    while len(pos) < num:
        rand = numpy.random.rand(2)
        rand[0] = (rmax-rmin)*rand[0]+rmin
        prob = rand[0]*2*(rmax-rmin)/(rmax+rmin)
        if prob > 1:
            print("Error: prob should not be > 1.", prob, rand[0], rmin, rmax)
        if rand[1] < prob:
            pos.append(rand[0])
    return numpy.asarray(pos)

# For a 2 dimenions cylindrically symmetric mesh, generates sum(part_num) new positions with a normalized linear distribution for each node,
#   each node i having part_num[i] particles and being determined by pos[i,:]
def randomYPositions_2D_cm(part_num, ymin, ymax):
    pos = numpy.zeros((numpy.sum(part_num)))
    c = 0
    for i in range(len(part_num)):
        pos[c:c+part_num[i]] = linearDistribution_sampling(part_num[i], ymin[i], ymax[i])
        c += part_num[i]
    return pos

