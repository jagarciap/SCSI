##-----------------------------------------------------------------------------------
# Obviously, this is approximate. Is just to have an idea of the RAM usage and 
# thus decide whether I can run it in my pc.
##-----------------------------------------------------------------------------------

# Estimated RAM per node (Obtained from mprof tests)
# The RAM usage varies through time, so I picked the highest use.
rpn = 1.250/(40/0.25+1)**2 #GiB/node

filename = '2020_11_05_MR_1.txt'
def nPoints_mesh_2D_rectangular(xmin, xmax, ymin, ymax, dx, dy):
    return int(((xmax-xmin)/dx+1)*((ymax-ymin)/dy+1))
nodes = nPoints_mesh_2D_rectangular(0.0, 42.0, -24.0, 24.0, 0.6, 0.6) +\
        nPoints_mesh_2D_rectangular(0.0, 24.0, -24.0, 24.0, 0.2, 0.2) +\
        nPoints_mesh_2D_rectangular(5.0, 13.2, -4.0, 4.0, 0.04, 0.04) +\
        nPoints_mesh_2D_rectangular(7.0, 12.2, -2.6, 2.6, 0.02, 0.02)

print(filename, nodes*rpn)
