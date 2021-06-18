from mesh import Mesh_2D_rm_sat as m
wtf = self.children[0].potential
oki = m.vtkOrdering(self.children[0].pic.mesh, wtf)
m.saveVTK(self.children[0].pic.mesh, "whatever", {'potential': oki})

import matplotlib.pyplot as plt
ook_v = self.mesh.volumes[2+61:self.mesh.nPoints:61]/self.mesh.volumes[2:self.mesh.nPoints-self.mesh.nx:61]
ook = species.mesh_values.density[2+61:self.mesh.nPoints:61]/species.mesh_values.density[2:self.mesh.nPoints-self.mesh.nx:61]
 plt.plot(ook, label = 'part')
 plt.plot(ook_v, label = 'vol')
 plt.legend()
 plt.show()

len(numpy.flatnonzero(pos[:,1] < self.mesh.dy/2))*val[0]/numpy.sum(self.mesh.volumes[:self.mesh.nx])
freq, bar, emm = plt.hist(pos[:,1], 91)
vols = self.mesh.volumes.reshape((self.mesh.ny,self.mesh.nx))
vol = numpy.sum(vols, axis = 1)
plt.plot(bar[:-1],vol*7e9/5e7, marker = '.', label = 'theory')
plt.plot(bar[:-1], freq, marker = '.', label = 'real')
plt.legend()
plt.show()
