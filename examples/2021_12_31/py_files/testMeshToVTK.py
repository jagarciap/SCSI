import numpy
import evtk.hl as evtk
from evtk.vtk import VtkGroup
import constants as c

#NOTE: In order for this to work it needs a folder called 'meshes' in the directory where this file is located.


def saveVTK(filename, i, j):
    ind = numpy.arange(0, len(i)*len(j), dtype = 'uint32')
    temp = numpy.zeros((1), dtype = 'int32')

    def fun(x,y):
        array = numpy.zeros((len(x),len(y)))
        for i in range(len(x)):
            array[i, :] = x[i]*y

        return array

    dictionary = {'whatever': fun(i,j)[:,None]}
    evtk.gridToVTK(filename, i, j, temp, pointData = dictionary)

i = numpy.arange(2, 10.5, 0.5)
j = numpy.arange(-4, 4.5, 0.5)
saveVTK("./meshes/test-1", i,j)

i = numpy.arange(2, 5.25, 0.25)
j = numpy.arange(-4, 4.25, 0.25)
saveVTK("./meshes/test-2", i,j)

g = VtkGroup("./group")
g.addFile(filepath = "./meshes/test-1.vtr", sim_time = 0.0)
g.addFile(filepath = "./meshes/test-2.vtr", sim_time = 0.0)
g.save()
