 numpy.where(numpy.logical_not(numpy.isin(array, self.mesh.boundaries[1].location)))

import matplotlib.pyplot as plt
vel = ghost.part_values.velocity[:np,:]
mag = numpy.linalg.norm(vel, axis = 1)
plt.hist(mag)
plt.show()
 
 import matplotlib.pyplot as plt
 np = system.at['protons'].part_values.current_n
 vel = system.at['protons'].part_values.velocity[:np,:]
 mag = numpy.linalg.norm(vel, axis = 1)
 plt.hist(mag, 41)
 plt.show()
