# This file provides methods to transform .vtr files into numpy arrays
import numpy
import os
import pdb
from subprocess import check_output
import sys
from vtk.util.numpy_support import vtk_to_numpy

import constants as c
from mesh import Mesh_2D_rm_sat

def vtrToNumpy(mesh, filenames, names):
    arrays = []
    #First filename
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.path.split(os.getcwd())[0]
    filename = cwd+'/results/'+filenames[0]
    reader = mesh.vtkReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    for name in names:
        temp = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(name)))
        temp = numpy.expand_dims(temp, axis = temp.ndim)
        arrays.append(temp)
    for filename in filenames[1:]:
        try:
            cwd = os.path.split(os.getcwd())[0]
            filename = cwd+'/results/'+filename
            reader = mesh.vtkReader()
            reader.SetFileName(filename)
            reader.Update()
            output = reader.GetOutput()
            for i in range(len(names)):
                temp = mesh.reverseVTKOrdering(vtk_to_numpy(output.GetPointData().GetArray(names[i])))
                temp = numpy.expand_dims(temp, axis = temp.ndim)
                arrays[i] = numpy.append(arrays[i], temp, axis = arrays[i].ndim-1)
        except:
            print("Filename:", filename)
            raise
    return arrays

def loadFromResults(files_id = '0-0-0'):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.path.split(os.getcwd())[0]
    cwd = cwd+'/results/'
    stdout = check_output('ls' +' {}'.format(cwd), shell=True)
    files = stdout.decode().split(sep='\n')
    #files = list(filter(lambda x: '0-0-0-0_ts' in x, files)) 
    files = list(filter(lambda x: x.partition('_')[0] == files_id and 'ts' in x.partition('_')[2], files)) 
    return files

## I commented this as I considered it unnecessary and more confusing than anything else. 'some_plots.py' in 'plotters' provides means to deal with the boundaries and the inner part of the satellite already.
#def filter_inner_boundary(mesh, data):
#    ind = mesh.getPosition(numpy.arange(mesh.nPoints))
#    mask = numpy.flatnonzero(numpy.logical_not(numpy.logical_and(numpy.logical_and(numpy.logical_and(ind[:,0] > mesh.boundaries[1].xmin,\
#                                                                                   ind[:,0] < mesh.boundaries[1].xmax),\
#                                                                                   ind[:,1] < mesh.boundaries[1].ymax),\
#                                                                                   ind[:,1] > mesh.boundaries[1].ymin)))
#    return [dat[mask,:] for dat in data]
