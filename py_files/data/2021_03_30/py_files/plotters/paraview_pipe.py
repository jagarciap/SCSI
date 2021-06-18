import matplotlib.pyplot as plt
import numpy
import os
import sys
sys.path.append(r'/home/jorge/ParaView-5.8.0-RC3-MPI-Linux-Python3.7-64bit/lib/python3.7/site-packages/')
from paraview.simple import *
from subprocess import check_output

cwd_base = os.getcwd().rsplit(sep = os.sep, maxsplit = 2)
cwd = os.path.join(cwd_base[0], 'results','')
stdout = check_output('ls' +' {}'.format(cwd), shell=True)
files = stdout.decode().split(sep='\n')
reader = OpenDataFile([cwd + a for a in files])
print(reader.PointData.keys())
print(reader.PointData.values())
p_vel = reader.PointData["Proton - Solar wind-velocity"]
print(type(p_vel))
print(type(p_vel.FieldData.GetArray(7)))
#help(reader)
