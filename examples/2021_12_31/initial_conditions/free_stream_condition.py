import numpy
import sys

sys.path.insert(0,'..')

import constants as c


def load_protons_SW(mesh, species):
    species.mesh_values.density[:] = c.P_N
    species.mesh_values.velocity[:,0] = c.P_V_SW
    species.mesh_values.velocity[:,1] = 0
    species.mesh_values.temperature[:] = c.P_T

def load_electrons_SW(mesh, species):
    species.mesh_values.density[:] = c.E_N
    species.mesh_values.velocity[:,0] = c.E_V_SW
    species.mesh_values.velocity[:,1] = 0
    species.mesh_values.temperature[:] = c.E_T
