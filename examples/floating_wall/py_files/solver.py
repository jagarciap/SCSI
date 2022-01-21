#File containing the methods to calculate potentials and fields
import copy
from datetime import datetime
from numba import jit, njit, prange
import matplotlib.pyplot as plt
import numpy
import pdb

import accelerated_functions as af
import constants as c

def satellite_2Drm_adjustement(mesh, pot):
    pot[mesh.boundaries[1].bottom[2:-2]+mesh.nx] = pot[mesh.boundaries[1].bottom[1:-1]]
    pot[mesh.boundaries[1].top[2:-2]-mesh.nx] = pot[mesh.boundaries[1].top[1:-1]]
    pot[mesh.boundaries[1].left[1:-1]+1] = pot[mesh.boundaries[1].left[1:-1]]
    pot[mesh.boundaries[1].right[1:-1]-1] = pot[mesh.boundaries[1].right[1:-1]]

#       +Method that computes the electric potential for a 2D rectangular mesh using the method of
#       +Successive Over Relaxation with Chebyshev Acceleration (SORCA). The method assumes that the 
#       +boundaries has been treated previously and assumes a Dirichlet boundary, so it only calculates values
#       +for the inner nodes. If a Neumann condition is desired, then the outer layer nodes are calculated correspondingly 
#       +in the later method apply applyElectricBoundary(e_field), not here.
#       +Parameters:
#       +rhs ([double]) =  is the right-hand-side of the equation Poisson(\phi) = rhs.
#       +pot ([double]) = is the potential before being updated, or it can be seen as the first guess of the solution.
#       +err (double) = is the maximum variation expected between succesives steps to reach solution.
#       +ind_pot ([ind]) = Indices where the poisson solver is going to be calculated
#       +step_limit (int) = a boundary in number of steps to avoid infinite loops.
#       +border (boolean) = If True, a layer around the 2D_rm is formed to deal with the border nodes without breaking the code.
#@jit(nopython=True, cache=True)
def poissonSolver_2D_rm_SORCA_p(mesh, pot, rhs, ind_pot, err = 1e-6, step_limit = 20000, border = False, adjustment = False):
    #Coefficients of the equation
    a = 1/mesh.dy/mesh.dy*numpy.ones(mesh.nPoints)
    b = a
    c = 1/mesh.dx/mesh.dx*numpy.ones(mesh.nPoints)
    d = c
    e = -2*(1/mesh.dx/mesh.dx+1/mesh.dy/mesh.dy)*numpy.ones(mesh.nPoints)
    #Defining rho
    delta_sq = mesh.nx*mesh.nx if mesh.nx < mesh.ny else mesh.ny*mesh.ny
    rho = 1 - numpy.pi*numpy.pi/2/delta_sq
    #NOTE: Delete later
    #tot_norm = []
    #Solver
    w = 1.0
    err_comp = 1.0
    #w = 2/(1-numpy.pi/mesh.nx)
    if border:
        nx = int(mesh.nx+2)
        offset = (ind_pot//mesh.nx)*2+1+nx
        n_pot = numpy.zeros((int(mesh.nx+2), int(mesh.ny+2)))
        n_pot[1:-1,1:-1] = pot.reshape((mesh.nx, mesh.ny), order = 'F')
        n_pot = n_pot.reshape(int((mesh.nx+2)*(mesh.ny+2)), order = 'F')
    else:
        offset = numpy.zeros_like(ind_pot)
        nx = mesh.nx
        n_pot = pot
    for t in range(1, step_limit+1):
        if adjustment:
            af.satellite_2Drm_adjustment_p(n_pot, mesh.boundaries[1].bottom, mesh.boundaries[1].top, mesh.boundaries[1].left, mesh.boundaries[1].right, nx)
        norm = af.mesh_loop(n_pot, a, b, c, d, e, rhs, ind_pot, offset, t, nx, w)
        #tot_norm.append(norm)
        #print("norm",norm, "w", w)
        #mid = int(mesh.ny/2)
        #ext = int(mesh.nx/4)
        #for j in range (-ext+mid, mid+ext+1):
        #    print(pot[j*mesh.nx+mid-ext:j*mesh.nx+mid+ext+1])
        #print("-------------------------------")
        #pdb.set_trace()
        w = 1.0/(1-0.25*rho*rho*w)
        if t == 1:
            w = 1.0/(1-0.5*rho*rho)
        if t % 500 == 2 and norm > 0:
            err_comp = norm
        if (norm/numpy.max(numpy.abs(n_pot[ind_pot])) < err and t > 2) or (abs(err_comp-norm)/err_comp < 0.05 and t%500 == 499):
            print("norm: ",norm,", err comp: ",err_comp,", step: ",t)
            break
        #if t % 100 == 0:
        #    print("norm: ",norm,", step: ",t)
        #if t == step_limit:
        #    raise ValueError("step_limit reached, solution not obtained. Error = {:e}.".format(norm))

    if border:
        n_pot = n_pot.reshape((int(mesh.nx+2), int(mesh.ny+2)), order = 'F')[1:-1,1:-1]
        n_pot = n_pot.reshape(mesh.nx*mesh.ny, order = 'F')
    numpy.copyto(pot, n_pot)

#       +Method that computes the electric potential for a 2D cylindrical (r-z) mesh using the method of
#       +Successive Over Relaxation with Chebyshev Acceleration (SORCA). The method assumes that the 
#       +boundaries have been treated previously and assumes a Dirichlet boundary, so it only calculates values
#       +for the inner nodes. If a Neumann condition is desired, then the outer layer nodes are calculated correspondingly 
#       +in the later method apply applyElectricBoundary(e_field), not here.
#       +Parameters:
#       +rhs ([double]) =  is the right-hand-side of the equation Poisson(\phi) = rhs.
#       +pot ([double]) = is the potential before being updated, or it can be seen as the first guess of the solution.
#       +err (double) = is the maximum variation expected between succesives steps to reach solution.
#       +ind_pot ([ind]) = Indices where the poisson solver is going to be calculated
#       +step_limit (int) = a boundary in number of steps to avoid infinite loops.
#       +border (boolean) = If True, a layer around the 2D_rm is formed to deal with the border nodes without breaking the code.
#@jit(nopython=True, cache=True)
def poissonSolver_2D_cm_SORCA_p(mesh, pot, rhs, ind_pot, err = 1e-6, step_limit = 20000, border = False, adjustment = False):
    #Coefficients of the equation
    ind = numpy.arange(mesh.nPoints)
    pos = mesh.getPosition(ind)
    a = 1/mesh.dy/mesh.dy-1/2/pos[:,1]/mesh.dy
    b = 1/mesh.dy/mesh.dy+1/2/pos[:,1]/mesh.dy
    c = 1/mesh.dx/mesh.dx*numpy.ones(mesh.nPoints)
    d = c
    e = -2*(1/mesh.dx/mesh.dx+1/mesh.dy/mesh.dy)*numpy.ones(mesh.nPoints)
    #Case with r = 0:
    if mesh.ymin == 0.0:
        a[:mesh.nx] = 0
        b[:mesh.nx] = 4/mesh.dy/mesh.dy
        e[:mesh.nx] = -2*(2/mesh.dy/mesh.dy+1/mesh.dx/mesh.dx)

    #Defining rho
    delta_sq = mesh.nx*mesh.nx if mesh.nx < mesh.ny else mesh.ny*mesh.ny
    rho = 1 - numpy.pi*numpy.pi/2/delta_sq
    #NOTE: Delete later
    #tot_norm = []
    #Solver
    w = 1.0
    err_comp = 1.0
    #w = 2/(1-numpy.pi/mesh.nx)
    if border:
        if mesh.ymin == 0.0:
            nx = int(mesh.nx+2)
            offset = (ind_pot//mesh.nx)*2+1
            n_pot = numpy.zeros((int(mesh.nx+2), int(mesh.ny+1)))
            n_pot[1:-1,:-1] = pot.reshape((mesh.nx, mesh.ny), order = 'F')
            n_pot = n_pot.reshape(int((mesh.nx+2)*(mesh.ny+1)), order = 'F')
        else:
            nx = int(mesh.nx+2)
            offset = (ind_pot//mesh.nx)*2+1+nx+2
            n_pot = numpy.zeros((int(mesh.nx+2), int(mesh.ny+2)))
            n_pot[1:-1,1:-1] = pot.reshape((mesh.nx, mesh.ny), order = 'F')
            n_pot = n_pot.reshape(int((mesh.nx+2)*(mesh.ny+2)), order = 'F')
    else:
        offset = numpy.zeros_like(ind_pot)
        nx = mesh.nx
        n_pot = pot
    for t in range(1, step_limit+1):
        if adjustment:
            af.satellite_2Drm_adjustment_p(n_pot, mesh.boundaries[1].bottom, mesh.boundaries[1].top, mesh.boundaries[1].left, mesh.boundaries[1].right, nx)
        norm = af.mesh_loop(n_pot, a, b, c, d, e, rhs, ind_pot, offset, t, nx, w)
        #tot_norm.append(norm)
        #print("norm",norm, "w", w)
        #mid = int(mesh.ny/2)
        #ext = int(mesh.nx/4)
        #for j in range (-ext+mid, mid+ext+1):
        #    print(pot[j*mesh.nx+mid-ext:j*mesh.nx+mid+ext+1])
        #print("-------------------------------")
        #pdb.set_trace()
        w = 1.0/(1-0.25*rho*rho*w)
        if t == 1:
            w = 1.0/(1-0.5*rho*rho)
        if t % 500 == 2 and norm > 0:
            err_comp = norm
        if (norm/numpy.max(numpy.abs(n_pot[ind_pot])) < err and t > 2) or (abs(err_comp-norm)/err_comp < 0.05 and t%500 == 499):
            print("norm: ",norm,", err comp: ",err_comp,", step: ",t)
            break
        #if t % 100 == 0:
        #    print("norm: ",norm,", step: ",t)
        #if t == step_limit:
        #    raise ValueError("step_limit reached, solution not obtained. Error = {:e}.".format(norm))

    if border:
        if mesh.ymin == 0.0:
            n_pot = n_pot.reshape((int(mesh.nx+2), int(mesh.ny+1)), order = 'F')[1:-1,:-1]
            n_pot = n_pot.reshape(mesh.nx*mesh.ny, order = 'F')
        else:
            n_pot = n_pot.reshape((int(mesh.nx+2), int(mesh.ny+2)), order = 'F')[1:-1,1:-1]
            n_pot = n_pot.reshape(mesh.nx*mesh.ny, order = 'F')
    numpy.copyto(pot, n_pot)

#       +Method that computes the electric potential for a 2D rectangular mesh using the method of
#       +Successive Over Relaxation with Chebyshev Acceleration (SORCA). The method assumes that the 
#       +boundaries has been treated previously and assumes a Dirichlet boundary, so it only calculates values
#       +for the inner nodes. If a Neumann condition is desired, then the outer layer nodes are calculated correspondingly 
#       +in the later method apply applyElectricBoundary(e_field), not here.
#       +Parameters:
#       +rhs ([double]) =  is the right-hand-side of the equation Poisson(\phi) = rhs.
#       +pot ([double]) = is the potential before being updated, or it can be seen as the first guess of the solution.
#       +err (double) = is the maximum variation expected between succesives steps to reach solution.
#       +ind_pot ([ind]) = Indices where the poisson solver is going to be calculated
#       +step_limit (int) = a boundary in number of steps to avoid infinite loops.
#       +border (boolean) = If True, a layer around the 2D_rm is formed to deal with the border nodes without breaking the code.
def poissonSolver_2D_rm_SORCA(mesh, pot, rhs, ind_pot, err = 1e-3, step_limit = 20000, border = False, adjustment = False):
    #Coefficients of the equation
    a = 1/mesh.dy/mesh.dy*numpy.ones(mesh.nPoints)
    b = a
    c = 1/mesh.dx/mesh.dx*numpy.ones(mesh.nPoints)
    d = c
    e = -2*(1/mesh.dx/mesh.dx+1/mesh.dy/mesh.dy)*numpy.ones(mesh.nPoints)
    #Defining rho
    delta_sq = mesh.nx*mesh.nx if mesh.nx < mesh.ny else mesh.ny*mesh.ny
    rho = 1 - numpy.pi*numpy.pi/2/delta_sq
    #NOTE: Delete later
    #tot_norm = []
    #Solver
    w = 1.0
    err_comp = 0.0
    #w = 2/(1-numpy.pi/mesh.nx)
    if border:
        nx = int(mesh.nx+2)
        offset = (ind_pot//mesh.nx)*2+1+nx+2
        n_pot = numpy.zeros((int(mesh.nx+2), int(mesh.ny+2)))
        n_pot[1:-1,1:-1] = pot.reshape((mesh.nx, mesh.ny), order = 'F')
        n_pot = n_pot.reshape(int((mesh.nx+2)*(mesh.ny+2)), order = 'F')
    else:
        offset = numpy.zeros_like(ind_pot)
        nx = mesh.nx
        n_pot = pot
    for t in range(1, step_limit+1):
        norm = 0.0
        if adjustment:
            satellite_2Drm_adjustement(mesh, n_pot)
        for ind, i0 in zip(ind_pot, offset):
            if (ind%nx+ind//nx)%2 == t%2:
                continue
            res = a[ind]*n_pot[i0+ind-nx]+b[ind]*n_pot[i0+ind+nx]+c[ind]*n_pot[i0+ind-1]+d[ind]*n_pot[i0+ind+1]+e[ind]*n_pot[i0+ind]-rhs[ind]
            norm += res*res
            n_pot[i0+ind] = n_pot[i0+ind]-w*res/e[ind]
        #tot_norm.append(norm)
        #print("norm",norm, "w", w)
        #mid = int(mesh.ny/2)
        #ext = int(mesh.nx/4)
        #for j in range (-ext+mid, mid+ext+1):
        #    print(pot[j*mesh.nx+mid-ext:j*mesh.nx+mid+ext+1])
        #print("-------------------------------")
        #pdb.set_trace()
        w = 1.0/(1-0.25*rho*rho*w)
        if t == 1:
            w = 1.0/(1-0.5*rho*rho)
        if t % 500 == 1:
            err_comp = norm
        if (norm < err and t > 2) or (abs(err_comp-norm)/err_comp < 0.05 and t%500 == 499):
            print("norm: ",norm,", err comp: ",err_comp,", step: ",t, flush=True)
            break
        if t % 100 == 0:
            print("norm: ",norm,", step: ",t, flush=True)
        if t == step_limit:
            raise ValueError("step_limit reached, solution not obtained. Error = {:e}.".format(norm))

    if border:
        n_pot = n_pot.reshape((int(mesh.nx+2), int(mesh.ny+2)), order = 'F')[1:-1,1:-1]
        n_pot = n_pot.reshape(mesh.nx*mesh.ny, order = 'F')
    numpy.copyto(pot, n_pot)

def apply_dielectric_border(mesh, pot, rho, eps):
    pot[mesh.boundaries[1].bottom[1:-1]+mesh.nx] = pot[mesh.boundaries[1].bottom[1:-1]]
    pot[mesh.boundaries[1].top[1:-1]-mesh.nx] = pot[mesh.boundaries[1].top[1:-1]]
    pot[mesh.boundaries[1].left[1:-1]+1] = pot[mesh.boundaries[1].left[1:-1]]
    pot[mesh.boundaries[1].right[1:-1]-1] = pot[mesh.boundaries[1].right[1:-1]]
    loc = numpy.unique(mesh.boundaries[1].location)
    for ind in loc:
        pot[ind] = (rho[ind]+1/mesh.dy/mesh.dy*(pot[ind+mesh.nx]*eps[ind+mesh.nx]+pot[ind-mesh.nx]*eps[ind-mesh.nx])+\
                            1/mesh.dx/mesh.dx*(pot[ind+1]*eps[ind+1]+pot[ind-1]*eps[ind-1]))/\
                            (1/mesh.dy/mesh.dy*(eps[ind+mesh.nx]+eps[ind-mesh.nx])+1/mesh.dx/mesh.dx*(eps[ind+1]+eps[ind-1]))

def poissonSolver_2D_rm_SORCA_dielectric(mesh, pot, rhs, ind_pot, err = 5e-2, step_limit = 15000, border = False):
    #Defining permittivity array
    loc = numpy.unique(mesh.boundaries[1].location)
    eps = c.EPS_0*numpy.ones((mesh.nPoints))
    eps[loc] = c.EPS_SAT
    eps[mesh.boundaries[1].ind_inner] = c.EPS_SAT
    #Coefficients of the equation
    a = 1/mesh.dy/mesh.dy*numpy.ones(mesh.nPoints)
    b = a
    cd = 1/mesh.dx/mesh.dx*numpy.ones(mesh.nPoints)
    d = cd
    e = -2*(1/mesh.dx/mesh.dx+1/mesh.dy/mesh.dy)*numpy.ones(mesh.nPoints)
    #Defining radius
    delta_sq = mesh.nx*mesh.nx if mesh.nx < mesh.ny else mesh.ny*mesh.ny
    radius = 1 - numpy.pi*numpy.pi/2/delta_sq
    #NOTE: Delete later
    #tot_norm = []
    #Solver
    w = 1.0
    err_comp = 1.0
    #w = 2/(1-numpy.pi/mesh.nx)
    pdb.set_trace()
    if border:
        nx = int(mesh.nx+2)
        offset = (ind_pot//mesh.nx)*2+1+nx+2
        n_pot = numpy.zeros((int(mesh.nx+2), int(mesh.ny+2)))
        n_pot[1:-1,1:-1] = pot.reshape((mesh.nx, mesh.ny), order = 'F')
        n_pot = n_pot.reshape(int((mesh.nx+2)*(mesh.ny+2)), order = 'F')
    else:
        offset = numpy.zeros_like(ind_pot)
        nx = mesh.nx
        n_pot = pot
    for t in range(1, step_limit+1):
        norm = 0.0
        for ind, i0 in zip(ind_pot, offset):
            if (ind%nx+ind//nx)%2 == t%2:
                continue
            res = a[ind]*n_pot[i0+ind-nx]+b[ind]*n_pot[i0+ind+nx]+cd[ind]*n_pot[i0+ind-1]+d[ind]*n_pot[i0+ind+1]+e[ind]*n_pot[i0+ind]-rhs[ind]
            norm += res*res
            n_pot[i0+ind] = n_pot[i0+ind]-w*res/e[ind]
        apply_dielectric_border(mesh, n_pot, rhs, eps)
        #tot_norm.append(norm)
        #print("norm",norm, "w", w)
        #mid = int(mesh.ny/2)
        #ext = int(mesh.nx/4)
        #for j in range (-ext+mid, mid+ext+1):
        #    print(pot[j*mesh.nx+mid-ext:j*mesh.nx+mid+ext+1])
        #print("-------------------------------")
        #pdb.set_trace()
        w = 1.0/(1-0.25*radius*radius*w)
        if t == 1:
            w = 1.0/(1-0.5*radius*radius)
        if t % 500 == 2:
            err_comp = norm
        if (norm < err and t > 2) or (abs(err_comp-norm)/err_comp < 0.01 and t%500==499):
            print("norm: ",norm,", err comp: ",(err_comp-norm)/err_comp,", step: ",t, flush=True)
            break
        if t % 100 == 0:
            print("norm: ",norm,", step: ",t, flush=True)
        if t == step_limit:
            raise ValueError("step_limit reached, solution not obtained. Error = {:e}.".format(norm))
    if border:
        n_pot = n_pot.reshape((mesh.nx+2, mesh.ny+2))[1:-1,1:-1]
        n_pot = n_pot.respahe(mesh.nx*mesh.ny)
    numpy.copyto(pot, n_pot)
        #NOTE: Just to check the new addition of handling borders
        #pdb.set_trace()

#       +Derivation of the scalar field potential ([double]) with the method of central differences, at nodes denoted by pot_ind.
def derive_2D_rm(mesh, potential, pot_ind):
    #Creating temporary field
    field = numpy.zeros((mesh.nPoints, 2))
    pot_u = potential[pot_ind+mesh.nx]
    pot_d = potential[pot_ind-mesh.nx]
    pot_l = potential[pot_ind-1]
    pot_r = potential[pot_ind+1]
    fieldx, fieldy = af.derive_2D_rm_p(pot_u, pot_d, pot_l, pot_r, mesh.dx, mesh.dy)
    field[pot_ind, 0] = fieldx
    field[pot_ind, 1] = fieldy
    return field

#       +Derivation of the scalar field potential ([double]) with the method of central differences, at nodes denoted by pot_ind.
#           This method is for a (z-r) cylindrical mesh.
def derive_2D_cm(mesh, potential, pot_ind):
    #Creating temporary field
    field = numpy.zeros((mesh.nPoints, 2))
    pot_u = potential[pot_ind+mesh.nx]
    pot_d = potential[pot_ind-mesh.nx]
    pot_l = potential[pot_ind-1]
    pot_r = potential[pot_ind+1]
    if mesh.ymin == 0.0:
        pot_d = numpy.where(pot_ind < mesh.nx, pot_u, pot_d)
    fieldx, fieldy = af.derive_2D_rm_p(pot_u, pot_d, pot_l, pot_r, mesh.dx, mesh.dy)
    field[pot_ind, 0] = fieldx
    field[pot_ind, 1] = fieldy
    return field

#       +Pade derivation for the nodes in the boundaries. Normal 2nd order derivation when the boundary is perpendicular to the direction of the derivative.
#       +Arguments:
#       +boundary (Boundary) = Boundary object with the information of the nodes to be treated.
#       +mesh (2D and uniform) = mesh with the information to make the finite difference.
#       +potential ([double]) = scalar to be derivated.
#       +Return: [double, double] two-component derivation of potential, with every row being one node of location.
def derive_2D_rm_boundaries(potential, boundary, nx, ny, dx, dy, conductor = False):
    #Creating temporary field
    location = numpy.unique(boundary.location)
    nPoints = len(potential)
    #Creating markers and checking type of boundary
    if  "1D" in boundary.type:
        b = boundary.directions == 0
        l = boundary.directions == 3
        r = boundary.directions == 1
        t = boundary.directions == 2
    else:
        b = numpy.isin(location, boundary.bottom)
        l = numpy.isin(location, boundary.left)
        r = numpy.isin(location, boundary.right)
        t = numpy.isin(location, boundary.top)

    inner = True if boundary.type.split(sep= "-")[0] == "Inner " else False
    outer = True if boundary.type.split(sep= "-")[0] == "Outer " else False
    #Derivative
    if conductor:
        return af.derive_2D_rm_boundaries_conductor_p(potential, location, b, l, r, t, inner, outer, nx, ny, dx, dy, nPoints)
    else:
        return af.derive_2D_rm_boundaries_p(potential, location, b, l, r, t, inner, outer, nx, ny, dx, dy, nPoints)

def location_indexes_inv(val, store = True, location = None):
    if store == True:
        location_indexes_inv.dic = {locations: indexes[0] for indexes, locations in numpy.ndenumerate(location)}
        return numpy.asarray([location_indexes_inv.dic.get(val_i) for val_i in val], dtype = numpy.uint16)
    else:
        return numpy.asarray([location_indexes_inv.dic.get(val_i) for val_i in val], dtype = numpy.uint16)

def capacity_Inv_Matrix(field):
    #For file names
    time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    #Set up for the method
    mesh = field.pic.mesh
    potential = numpy.zeros((mesh.nPoints))
    rhs = numpy.zeros((mesh.nPoints))
    #Construction of satellite-symmetry indexes
    counterclockwise = numpy.append(numpy.append(numpy.append(field.pic.mesh.boundaries[1].bottom,\
                                                 field.pic.mesh.boundaries[1].right[1:]), \
                                                 field.pic.mesh.boundaries[1].top[-2::-1]),\
                                                 field.pic.mesh.boundaries[1].left[-2:0:-1])
    loc, ind = numpy.unique(counterclockwise, return_index = True)
    tot_sat = len(loc)
    aux = numpy.arange(tot_sat)
    n_iter = len(field.pic.mesh.boundaries[1].bottom[:-1])
    #Construction of the matrix
    for i in range(n_iter):
        potential *= 0
        rhs *= 0
        rhs[field.pic.mesh.boundaries[1].bottom[i]] -= 1.0/mesh.volumes[field.pic.mesh.boundaries[1].bottom[i]]/c.EPS_SAT
        print("Node ",i," of ",n_iter, flush=True)
        try:
            poissonSolver_2D_rm_SORCA(mesh, potential, rhs, numpy.sort(numpy.append(field.ind_calc, field.pic.mesh.boundaries[1].location)))
        except ValueError as error:
            numpy.savetxt('./data/incomplete_inv_capacity_matrix_'+time+'.txt', field.inv_capacity)
            print("Incomplete matrix stored")
            raise error

        field.inv_capacity[:,i] = potential[loc]
        field.inv_capacity[:,location_indexes_inv([(i+mesh.nxsat-1)%tot_sat], store = False)[0]] = \
            potential[counterclockwise[(aux-(mesh.nxsat-1))%tot_sat]][ind]
        field.inv_capacity[:,location_indexes_inv([(i+2*(mesh.nxsat-1))%tot_sat], store = False)[0]] = \
            potential[counterclockwise[(aux-2*(mesh.nxsat-1))%tot_sat]][ind]
        field.inv_capacity[:,location_indexes_inv([(i+3*(mesh.nxsat-1))%tot_sat], store = False)[0]] = \
            potential[counterclockwise[(aux-3*(mesh.nxsat-1))%tot_sat]][ind]

    #Putting the expected location reference for location_indexes_inv
    test = location_indexes_inv([field.pic.mesh.sat_i], store = True, location = numpy.unique(field.pic.mesh.boundaries[1].location))[0]
    assert test == 0, "location_indexes_inv is not correctly set up"

#capacity_Inv_Matrix_asym(Field field) = This method takes the field received as argument and populates the matrix inv_capacity.
#   The code does not assume any type of symmetry, and uses directly the field object itself for its calculations.
def capacity_Inv_Matrix_asym(field):
    #For file names
    time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    #Set up for the method
    mesh = field.pic.mesh
    potential = numpy.zeros((mesh.nPoints))
    rhs = numpy.zeros((mesh.nPoints))
    #Construction of the matrix
    loc = numpy.unique(field.pic.mesh.location_sat)
    n_iter = len(loc)
    for i in range(n_iter):
        potential *= 0
        rhs *= 0
        rhs[loc[i]] -= 1.0/mesh.volumes[loc[i]]/c.EPS_SAT
        print("Node ",i," of ",n_iter, flush=True)
        try:
            poissonSolver_2D_rm_SORCA_p(mesh, potential, rhs, numpy.sort(numpy.append(field.ind_calc, loc)), border = True)
        except ValueError as error:
            numpy.savetxt('./data/incomplete_inv_capacity_matrix_'+time+'.txt', field.inv_capacity)
            print("Incomplete matrix stored")
            raise error

        field.inv_capacity[:,i] = potential[loc]

    #Saving the matrix for other runs
    numpy.savetxt('./data/inv_capacity_matrix_'+time+'.txt', field.inv_capacity)

def capacity_Inv_Matrix_asym_recursive(field): 
    #For file names
    time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    #Set up for the method
    rhs = numpy.zeros((field.pic.mesh.accPoints))
    field.assignValuesToArray_recursive("potential", numpy.arange(field.pic.mesh.accPoints), numpy.zeros((field.pic.mesh.accPoints)))
    repeated_ind, positions = field.pic.mesh.groupIndexByPosition(field.pic.mesh.overall_location_sat, positions = True)
    n_iter = numpy.max(field.pic.mesh.overall_location_sat)
    #Construction of the matrix
    for ind in field.pic.mesh.overall_location_sat:
        for same, position in zip(repeated_ind, positions):
            if ind in same:
                same_ind = repeated_ind.pop(0)
                positions = numpy.delete(positions, 0, 0)
                position_p = position.reshape((1,2))
                #Create rhs for the whole mesh nodes 
                rhs *= 0
                field.pic.scatter(position_p, -1.0/c.EPS_SAT/field.pic.mesh.volumes[ind]*numpy.ones_like(position_p[:,0]), rhs, surface = True)
                print("Node ",ind, ", Entry: ", location_indexes_inv([ind], store = False)[0], "of ", len(field.inv_capacity[:,0]), flush=True)
                try:
                    field.computeField(None, rho = rhs, border = True, interpolate = True, adjustment = False)
                except ValueError as error:
                    numpy.savetxt('./data/incomplete_inv_capacity_matrix_'+time+'.txt', field.inv_capacity)
                    print("Incomplete matrix stored")
                    raise error

                matrix_ind = list(location_indexes_inv(same_ind, store = False))
                def create_inv_capacity_matrix(field_i, ind_list, counter_sat = None):
                    if counter_sat == None:
                        counter_sat = [0]
                    loc = numpy.unique(field_i.pic.mesh.location_sat)
                    #TODO: Check modification here
                    if ind_list[0] < counter_sat[0]+len(loc):
                        ind = ind_list.pop(0)
                        field.inv_capacity[counter_sat[0]:counter_sat[0]+len(loc),ind] = field_i.potential[loc]
                    counter_sat[0] += len(loc)
                    for child in field_i.children:
                        create_inv_capacity_matrix(child, ind_list, counter_sat = counter_sat)
                create_inv_capacity_matrix(field, matrix_ind)
                break
                #def create_inv_capacity_matrix(field, ind_list, counter = None, counter_sat = None):
                #    if counter == None:
                #        counter = [0]
                #        counter_sat = [0]
                #    ind = ind_list.pop(0)
                #    loc = numpy.unique(field.pic.mesh.location_sat)
                #    if len(ind) > 0:
                #        field.inv_capacity[counter_sat[0]:counter_sat[0]+len(loc),matrix_ind[counter[0]]] = field.potential[loc]
                #        counter[0] += 1
                #        counter_sat[0] += len(loc)

                #field.inv_capacity[:,matrix_ind] = numpy.repeat(field.getTotalArray("potential")[field.pic.mesh.overall_location_sat, None], len(same_ind), axis = 1)
    
    #Saving the matrix for other runs
    numpy.savetxt('./data/inv_capacity_matrix_'+time+'.txt', field.inv_capacity)
    #Reseting potential
    field.assignValuesToArray_recursive("potential", numpy.arange(field.pic.mesh.accPoints), numpy.zeros((field.pic.mesh.accPoints)))


def capacity_Inv_Matrix_asym_recursive_from_incomplete_matrix(field, i0 = 240): 
    #For file names
    time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')

    #Set up for the method
    rhs = numpy.zeros((field.pic.mesh.accPoints))
    field.assignValuesToArray_recursive("potential", numpy.arange(field.pic.mesh.accPoints), numpy.zeros((field.pic.mesh.accPoints)))
    repeated_ind, positions = field.pic.mesh.groupIndexByPosition(field.pic.mesh.overall_location_sat, positions = True)
    n_iter = numpy.max(field.pic.mesh.overall_location_sat)
    def calculate_off_sets(mesh, acc_list = None):
        if acc_list == None:
            acc_list= [0,len(numpy.unique(mesh.location_sat))]
        else:
            acc_list.append(acc_list[-1]+len(numpy.unique(mesh.location_sat)))
        for child in mesh.children:
            calculate_off_sets(child, acc_list = acc_list)
        return acc_list

    off_sets = calculate_off_sets(field.pic.mesh)

    #Construction of the matrix
    for ind in field.pic.mesh.overall_location_sat:
        for same, position in zip(repeated_ind, positions):
            if ind in same:
                same_ind = same
                repeated_ind.remove(same)
                matrix_ind = list(location_indexes_inv(same_ind, store = False))
                error = False

                if matrix_ind[0] < i0:
                    for i in range(len(matrix_ind)-1, -1, -1):
                        e1 = numpy.all(field.inv_capacity[off_sets[i]:off_sets[i+1], matrix_ind[i]] == 0)
                        mask = numpy.full((len(field.inv_capacity[:,0])), True)
                        mask[off_sets[i]:off_sets[i+1]] = False
                        e2 = numpy.any(field.inv_capacity[mask, matrix_ind[i]] != 0)
                        if e1 or e2:
                            error = True
                            print("Node ", ind, flush = True)
                            print("e1: ", e1, "e2: ", e2, flush = True)
                            break
                    if error:
                        for loc_i in matrix_ind:
                            field.inv_capacity[:, loc_i] = 0
                else:
                    error = True

                if error:
                    position = position.reshape((1,2))
                    #Create rhs for the whole mesh nodes 
                    rhs *= 0
                    field.pic.scatter(position, -1.0/c.EPS_SAT/field.pic.mesh.volumes[ind]*numpy.ones_like(position[:,0]), rhs)
                    print("Node ",ind, ". Entry: ", location_indexes_inv([ind], store = False)[0], "of ", len(field.inv_capacity[:,0]), flush=True)
                    try:
                        field.computeField(None, rho = rhs, border = True, interpolate = True, adjustment = False)
                    except ValueError as error:
                        numpy.savetxt('./data/incomplete_inv_capacity_matrix_'+time+'.txt', field.inv_capacity)
                        print("Incomplete matrix stored")
                        raise error

                    #NOTE: If there is an error here, check whether is because of eliminating same from repeated_ind
                    def create_inv_capacity_matrix(field_i, ind_list, counter_sat = None):
                        if counter_sat == None:
                            counter_sat = [0]
                        loc = numpy.unique(field_i.pic.mesh.location_sat)
                        #TODO: Check modification here
                        if ind_list[0] < counter_sat[0]+len(loc):
                            ind = ind_list.pop(0)
                            field.inv_capacity[counter_sat[0]:counter_sat[0]+len(loc),ind] = field_i.potential[loc]
                        counter_sat[0] += len(loc)
                        for child in field_i.children:
                            create_inv_capacity_matrix(child, ind_list, counter_sat = counter_sat)
                    create_inv_capacity_matrix(field, matrix_ind)
                    break

                #def create_inv_capacity_matrix(field, ind_list, counter = None, counter_sat = None):
                #    if counter == None:
                #        counter = [0]
                #        counter_sat = [0]
                #    ind = ind_list.pop(0)
                #    loc = numpy.unique(field.pic.mesh.location_sat)
                #    if len(ind) > 0:
                #        field.inv_capacity[counter_sat[0]:counter_sat[0]+len(loc),matrix_ind[counter[0]]] = field.potential[loc]
                #        counter[0] += 1
                #        counter_sat[0] += len(loc)

                #field.inv_capacity[:,matrix_ind] = numpy.repeat(field.getTotalArray("potential")[field.pic.mesh.overall_location_sat, None], len(same_ind), axis = 1)
    
    #Saving the matrix for other runs
    numpy.savetxt('./data/inv_capacity_matrix_'+time+'.txt', field.inv_capacity)
    #Reseting potential
    field.assignValuesToArray_recursive("potential", numpy.arange(field.pic.mesh.accPoints), numpy.zeros((field.pic.mesh.accPoints)))
