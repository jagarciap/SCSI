import numba as nb
import numpy
import pdb

import constants as c

## ---------------------------------------------------------------------------------------------------------------
# Boundaries/*
## ---------------------------------------------------------------------------------------------------------------

#This function returns True for particles which are inside the (domain+borders)
@nb.njit(parallel=True, cache=True)
def leq_2D_p(pos, xmin, xmax, ymin, ymax, prec = 10**(-5)):
    return numpy.logical_not(numpy.logical_or(numpy.logical_or(numpy.logical_or((pos[:,0]-xmin) < -prec, (pos[:,0]-xmax) > prec),\
                                                                                (pos[:,1]-ymin) < -prec), (pos[:,1]-ymax) > prec))

#This function returns True for particles which are inside the (domain-borders)
@nb.njit(parallel=True, cache=True)
def l_2D_p(pos, xmin, xmax, ymin, ymax, prec = 10**(-5)):
    return numpy.logical_and(numpy.logical_and(numpy.logical_and((pos[:,0]-xmin) > prec, (pos[:,0]-xmax) < -prec),\
                                                                (pos[:,1]-ymin) > prec),(pos[:,1]-ymax) < -prec)

#This function returns True for particles which are outside the (domain+borders)
@nb.njit(parallel=True, cache=True)
def g_2D_p(pos, xmin, xmax, ymin, ymax, prec = 10**(-5)):
    return numpy.logical_or(numpy.logical_or(numpy.logical_or((pos[:,0]-xmin) < -prec, (pos[:,0]-xmax) > prec),\
                                                             (pos[:,1]-ymin) < -prec), (pos[:,1]-ymax) > prec)

#This function returns True for particles which are outside the (domain-borders)
@nb.njit(parallel=True, cache=True)
def geq_2D_p(pos, xmin, xmax, ymin, ymax, prec = 10**(-5)):
    return numpy.logical_not(numpy.logical_and(numpy.logical_and(numpy.logical_and((pos[:,0]-xmin) > prec, (pos[:,0]-xmax) < -prec),\
                                                                                   (pos[:,1]-ymin) > prec),(pos[:,1]-ymax) < -prec))

#This function returns True for particles that are to the right of the border including border
@nb.guvectorize([(nb.float64[:], nb.float64, nb.b1[:])], "(n),()->(n)", nopython=True, cache = True, target = "parallel")
def geq_1D_p(diff, prec, res):
    for i in range(len(diff)):
        res[i] = not (diff[i] < -prec)

#This function returns True for particles that are to the right of border excluding border
@nb.guvectorize([(nb.float64[:], nb.float64, nb.b1[:])], "(n),()->(n)", nopython=True, cache = True, target = "parallel")
def g_1D_p(diff, prec, res):
    for i in range(len(diff)):
        res[i] = diff[i] > prec



## ---------------------------------------------------------------------------------------------------------------
# mesh.py
## ---------------------------------------------------------------------------------------------------------------

@nb.guvectorize([(nb.u4[:], nb.u4[:], nb.u4, nb.u4[:])], "(n),(n),()->(n)", nopython=True, cache = True, target = "parallel")
def indexToArray_p(indx, indy, nx, res):
    for i in range(len(indx)):
        res[i] = indy[i]*nx+indx[i]

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64[:])], "(n),(n),(),(),(),()->(n),(n)", nopython=True, cache = True, target = "parallel")
def getIndex_p(posx, posy, dx, dy, xmin, ymin, indx, indy):
    for i in range(len(posx)):
        indx[i] = (posx[i]-xmin)/dx
        indy[i] = (posy[i]-ymin)/dy

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64[:])], "(n),(n),(),(),(),()->(n),(n)", nopython=True, cache = True, target = "parallel")
def getPosition_p(index2Dx, index2Dy, dx, dy, xmin, ymin, posx, posy):
    for i in range(len(index2Dx)):
        posx[i] = xmin+dx*index2Dx[i]
        posy[i] = ymin+dy*index2Dy[i]

#@nb.njit(parallel=True, cache=True, fastmath = True)
#def arrayToIndex_p(array, nx):
#    j = array//nx
#    i = array-j*nx
#    returned = numpy.zeros((len(j), 2))
#    returned[:,0] = i
#    returned[:,1] = j
#    return returned
#
#@nb.guvectorize([(nb.int64[:], nb.int64, nb.int64[:], nb.int64[:])], "(n),()->(n),(n)", nopython=True, cache = True, target = "parallel")
#def arrayToIndex_p(arr, nx, i, j):
#    for n in range(arr.shape[0]):
#        i[n] = arr[n]%nx
#        j[n] = arr[n]//nx


## ---------------------------------------------------------------------------------------------------------------
# pic.py
## ---------------------------------------------------------------------------------------------------------------

#@nb.njit(parallel=True, cache=True)
#def gather_p(mc, index, dim, prec, dec_prec):
#    #Creating the array
#    values = numpy.zeros((numpy.shape(index)[0], dim))
#
#    di = numpy.zeros((mc.shape[0]))
#    dj = numpy.zeros((mc.shape[0]))
#    
#    numpy.around(mc[:,0] - index[:,0], dec_prec, di)
#    numpy.around(mc[:,1] - index[:,1], dec_prec, dj)
#
#    #Dealing with nodes
#    filter_i = di > prec
#    filter_j = dj > prec
#
#    #NOTE: Maybe this can be further optmized later
#    di = numpy.repeat(numpy.expand_dims(di, 1), dim, 1)
#    dj = numpy.repeat(numpy.expand_dims(dj, 1), dim, 1)
#
#    return values, di, dj, filter_i, filter_j

## ---------------------------------------------------------------------------------------------------------------
# field.py
## ---------------------------------------------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def floating_potential_p(inv_capacity, capacity, charge, volumes, q):
    new_potential = inv_capacity@charge
    #phi_c = numpy.sum(capacity@new_potential.T)/numpy.sum(capacity)
    phi_c = numpy.sum(charge)/numpy.trace(capacity)
    d_q = capacity@(phi_c-new_potential)
    d_n = d_q/q/volumes
    return phi_c, d_q, d_n

#NOTE: whaaaat, permitiviy should be 1/permittivity
@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64[:])], "(n),(n),(n),()->(n)", nopython=True, cache = True, target = "parallel")
def induced_charge_p(field_x, field_y, area, permittivity, res):
    for i in range(len(field_x)):
        res[i] = field_x[i] if field_y[i] == 0.0 else field_x[i]/2
        res[i] += field_y[i] if field_x[i] == 0.0 else field_y[i]/2
        res[i] *= area[i]*permittivity

## ---------------------------------------------------------------------------------------------------------------
# motion.py
## ---------------------------------------------------------------------------------------------------------------

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64[:])], "(n),(n),(),()->(n)", nopython=True, cache = True, target = "parallel")
def rewindVelocity_p(velocity, field, q_over_m, dt, res):
    for i in range(field.shape[0]):
        res[i] = velocity[i]-q_over_m*dt/2*field[i]


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64, nb.float64[:])], "(n),(n),()->(n)", nopython=True, cache = True, target = "parallel")
def updateParticles_p(position, velocity, dt, res):
    for i in range(position.shape[0]):
        res[i] = position[i]+velocity[i]*dt

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64[:])], "(n),(n),(),()->(n)", nopython=True, cache = True, target = "parallel")
def electricAdvance_p(velocity, e_field, q_over_m, dt, res):
    for i in range(velocity.shape[0]):
        res[i] =  velocity[i]+e_field[i]*q_over_m*dt

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64[:], nb.float64[:])], "(n),(n),(n),(),()->(n),(n)", nopython=True, cache = True, target = "parallel")
def magneticRotation_p(vx, vy, B, q_over_m, dt, newx, newy):
    for i in range(len(vx)):
        t = q_over_m*B[i]*dt/2
        sine = 2*t/(1+t*t)
        v_1 = vx[i] + vy[i]*t
        newy[i] = vy[i] - v_1*sine
        newx[i] = v_1 + newy[i]*t

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], "(n),(n)->(n)", nopython=True, cache = True, target = "parallel")
def motionTreatment_p(velocity, vel_dic, res):
    for i in range(velocity.shape[0]):
        res[i] = (velocity[i]+vel_dic[i])/2

## ---------------------------------------------------------------------------------------------------------------
# solver.py
## ---------------------------------------------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def satellite_2Drm_adjustment_p(pot, bottom, top, left, right, nx):
    pot[bottom[1:-1]+nx] = pot[bottom[1:-1]]
    pot[top[1:-1]-nx] = pot[top[1:-1]]
    pot[left[1:-1]+1] = pot[left[1:-1]]
    pot[right[1:-1]-1] = pot[right[1:-1]]

@nb.njit(parallel=True, cache=True)
def mesh_loop(n_pot, a, b, c, d, e, rhs, ind_pot, offset, t, nx, w):
    norm = 0.0
    for i in nb.prange(len(ind_pot)):
        ind = ind_pot[i]
        i0 = offset[i]
        if (ind%nx+ind//nx)%2 == t%2:
            continue
        res = a[ind]*n_pot[i0+ind-nx]+b[ind]*n_pot[i0+ind+nx]+c[ind]*n_pot[i0+ind-1]+d[ind]*n_pot[i0+ind+1]+e[ind]*n_pot[i0+ind]-rhs[ind]
        norm += res*res
        n_pot[i0+ind] = n_pot[i0+ind]-w*res/e[ind]
    return norm

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64[:], nb.float64[:])],\
        "(n),(n),(n),(n),(),()->(n),(n)", nopython=True, cache = True, target = "parallel")
def derive_2D_rm_p(pot_u, pot_d, pot_l, pot_r, dx, dy, fieldx, fieldy):
    for i in range(len(pot_u)):
        fieldx[i] = (pot_r[i]-pot_l[i])/(2*dx)
        fieldy[i] = (pot_u[i]-pot_d[i])/(2*dy)

@nb.njit(parallel=True, cache=True)
def derive_2D_rm_boundaries_p(potential, location, b, l, r, t, inner, outer, nx, ny, dx, dy, nPoints):
    field = numpy.zeros((len(location),2))
    field[:,1] += numpy.where(numpy.logical_or(numpy.logical_and(outer,b),numpy.logical_and(inner,t)),\
                              (-3*potential[location]+4*potential[(location+nx)%nPoints]-potential[(location+2*nx)%nPoints])/(2*dy), 0)
    field[:,1] += numpy.where(numpy.logical_or(numpy.logical_and(outer,t),numpy.logical_and(inner,b)),\
                              (3*potential[location]-4*potential[(location-nx)%nPoints]+potential[(location-2*nx)%nPoints])/(2*dy), 0)
    field[:,1] += numpy.where(numpy.logical_not(numpy.logical_or(t,b)),(potential[(location+nx)%nPoints]-potential[(location-nx)%nPoints])/(2*dy), 0)
    field[:,0] += numpy.where(numpy.logical_or(numpy.logical_and(outer,l),numpy.logical_and(inner,r)),\
                              (-3*potential[location]+4*potential[(location+1)%nPoints]-potential[(location+2)%nPoints])/(2*dx), 0)
    field[:,0] += numpy.where(numpy.logical_or(numpy.logical_and(outer,r),numpy.logical_and(inner,l)),\
                              (3*potential[location]-4*potential[(location-1)%nPoints]+potential[(location-2)%nPoints])/(2*dx), 0)
    field[:,0] += numpy.where(numpy.logical_not(numpy.logical_or(l,r)),(potential[(location+1)%nPoints]-potential[(location-1)%nPoints])/(2*dx), 0)
    return field

@nb.njit(parallel=True, cache=True)
def derive_2D_rm_boundaries_conductor_p(potential, location, b, l, r, t, inner, outer, nx, ny, dx, dy, nPoints):
    field = numpy.zeros((len(location),2))
    field[:,1] += numpy.where(numpy.logical_or(numpy.logical_and(outer,b),numpy.logical_and(inner,t)),\
                              (-3*potential[location]+4*potential[(location+nx)%nPoints]-potential[(location+2*nx)%nPoints])/(2*dy), 0)
    field[:,1] += numpy.where(numpy.logical_or(numpy.logical_and(outer,t),numpy.logical_and(inner,b)),\
                              (3*potential[location]-4*potential[(location-nx)%nPoints]+potential[(location-2*nx)%nPoints])/(2*dy), 0)
    field[:,0] += numpy.where(numpy.logical_or(numpy.logical_and(outer,l),numpy.logical_and(inner,r)),\
                              (-3*potential[location]+4*potential[(location+1)%nPoints]-potential[(location+2)%nPoints])/(2*dx), 0)
    field[:,0] += numpy.where(numpy.logical_or(numpy.logical_and(outer,r),numpy.logical_and(inner,l)),\
                              (3*potential[location]-4*potential[(location-1)%nPoints]+potential[(location-2)%nPoints])/(2*dx), 0)
    return field
