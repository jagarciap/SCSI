
vol = part_solver.pic.mesh.volumes[:part_solver.pic.mesh.nPoints:part_solver.pic.mesh.nx]
temp = ghost.part_values.position[:np, :]
ind = numpy.logical_or(temp[:,0] < 0.6/2+1e-3, temp[:,0] > 36-0.6/2-1e-3)
temp2 = temp[ind,:]
bins = part_solver.pic.mesh.getPosition(numpy.arange(0,part_solver.pic.mesh.nPoints, part_solver.pic.mesh.nx))[:,1]
bins[1:] -= part_solver.pic.mesh.dy/2
bins = numpy.append(bins, [ymax])
ok1, ok2, ok3 = plt.hist(temp2[:,1], bins = bins)
plt.plot(ok2[:-1], ok1*c.P_SPWT/vol, marker = '.')
plt.axvline(x=0.0, color = 'r')
plt.axhline(y=0.0, color = 'r')

bins = part_solver.pic.mesh.getPosition(numpy.arange(0,part_solver.pic.mesh.nPoints, part_solver.pic.mesh.nx))[:,1]
ok4 = numpy.zeros_like(vol)
ok4[0] = ok1[0]
ok4[-1] = ok1[-1]
ok4[1:-1] = ok1[1:-1:2]+ok1[2:-1:2]
