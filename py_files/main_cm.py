# Main file of the program
import copy
import numpy
import pdb
import sys
import time
numpy.seterr(invalid='ignore', divide='ignore', over = 'raise')

sys.path.insert(0,'..')
sys.stderr = open("err.txt", "w")

import constants as c
from field import Constant_Magnetic_Field_recursive
from mesh_setup import mesh_file_reader
from Species.proton import Proton_SW
from Species.xenon import Xenon_Ion
from Species.electron import Electron_SW, Photoelectron, Secondary_Emission_Electron, Electron_HET
from Species.user_defined import User_Defined
import initial_conditions.satellite_condition as ic
from motion import Leap_Frog_2D3Dcm
import output as out
from timing import Timing



## ---------------------------------------------------------------------------------------------------------------
# Initiating data structures for the system
## ---------------------------------------------------------------------------------------------------------------


#System:
#
#Definition = Is the class that contains every variable and class necessary for the simulation to be executed.
#Attributes:
#	+ts (int) = Timestep of the simulation.
#	+The rest of the variables will change with the simulation, but normally, there are:
#	+mesh (Mesh).
#	+pic (PIC).
#	+fields (Field) = Probably several of them.
#	+species (species) = Probably several of them.
#	+part_solver (Motion_Solver).
#Methods:
#	+Remark about init(): It will "declare" the attributes necessary for the simulation to run. The actual assignment of atributes
#		to instances of each class will occur during the 'initial condition' section of 'main.py'.
#	+arrangePickle() : Variable = Return a tuple of keys to iterate over when saving and loading of/from '.pkl' files, in the order required.
#	+arrangeVTK() : Variable = Return a tuple of keys to iterate over when saving and loading of/from VTK files, in the order required.
class System(object):
    def __init__(self):
        self.at = {}
        self.at['ts'] = 0
        #TODO: Change later
        self.at['mesh'], self.at['pic'], self.at['e_field'], self.at['part_reformer'] = mesh_file_reader('2021_11_29.txt')
        self.at['mesh'].print()
        self.at['electrons'] = Electron_SW(0.0, c.E_SPWT, c.E_SIZE, c.DIM, 3, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        self.at['photoelectrons'] = Photoelectron(c.PHE_T, c.PHE_FLUX, 0.0, c.PHE_SPWT, c.PHE_SIZE, c.DIM, 3, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        self.at['see'] = Secondary_Emission_Electron(c.SEE_T, 0.0, c.SEE_SPWT, c.SEE_SIZE, c.DIM, 3, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        self.at['protons'] = Proton_SW(0.0, c.P_SPWT, c.P_SIZE, c.DIM, 3, self.at['mesh'].accPoints, self.at['mesh'].overall_location_sat, c.NUM_TRACKED)
        #self.at['user'] = User_Defined(c.P_DT, -c.QE, c.MP, 0, c.P_SPWT, 1, c.DIM, c.DIM, self.at['mesh'].nPoints, 0, "1")
        self.at['m_field'] = Constant_Magnetic_Field_recursive(self.at['pic'], c.B_DIM, [], True)
        self.at['part_solver'] = Leap_Frog_2D3Dcm(self.at['pic'], [self.at['electrons'].name, self.at['photoelectrons'].name, self.at['see'].name, self.at['protons'].name],\
                [self.at['electrons'].part_values.max_n, self.at['photoelectrons'].part_values.max_n, self.at['see'].part_values.max_n, self.at['protons'].part_values.max_n],\
                [self.at['electrons'].vel_dim, self.at['photoelectrons'].vel_dim, self.at['see'].vel_dim, self.at['protons'].vel_dim])

    def arrangePickle(self):
        #return ('ts', 'e_field', 'electrons', 'protons', 'user', 'part_solver')
        return ('ts', 'e_field', 'm_field', 'electrons', 'photoelectrons', 'see', 'protons', 'part_solver')

    def arrangeVTK(self):
        #return ('ts', 'e_field', 'electrons', 'protons', 'user')
        return ('ts', 'e_field', 'm_field', 'electrons', 'photoelectrons', 'see', 'protons', 'part_reformer')

    def arrangeParticlesTXT(self):
        #return ('ts', 'electrons', 'protons', 'user')
        return ('ts', 'electrons', 'photoelectrons', 'see', 'protons')

#Initialization of the system
system = System()

## ---------------------------------------------------------------------------------------------------------------
# Initial condition
## ---------------------------------------------------------------------------------------------------------------
# Initial conditions are selected with a number next to the file when the program is executed, e.g. 'python3 main.py 2'.
# If the initial condition requires more arguments, they are written here in the file.
# Listed Initial conditions:
# 1: Basic empty system.
# 2: Execution from a VTK file.
# 3: Execution from a Pickle file.

if sys.argv[1] == '1':
    system.at['part_solver'].initialConfiguration(system.at['electrons'], system.at['e_field'])
    system.at['part_solver'].initialConfiguration(system.at['photoelectrons'], system.at['e_field'])
    system.at['part_solver'].initialConfiguration(system.at['see'], system.at['e_field'])
    system.at['part_solver'].initialConfiguration(system.at['protons'], system.at['e_field'])

elif sys.argv[1] == '2':
    #File to be used as source of initial condition
    filename = 'ts00009.vtr'
    system.at['ts'] = 0
    out.loadVTK(filename, system.at['mesh'], system.at, system.arrangeVTK())
    system.at['e_field'] = Electrostatic_2D_rm_sat_cond(system.at['pic'], c.DIM)
    system.at['part_solver'].initialConfiguration(system.at['electrons'], system.at['e_field'])
    system.at['part_solver'].initialConfiguration(system.at['photoelectrons'], system.at['e_field'])
    system.at['part_solver'].initialConfiguration(system.at['see'], system.at['e_field'])
    system.at['part_solver'].initialConfiguration(system.at['protons'], system.at['e_field'])

elif sys.argv[1] == '3':
    #File to be used as source of initial condition
    filename = 'sys_ts=16000_2020-06-06_23h02m.pkl'
    out.loadPickle(filename, system.at, system.arrangePickle())

elif sys.argv[1] == '4':
    ic.load_protons_SW(system.at['mesh'], system.at['protons'])
    ic.load_electrons_SW(system.at['mesh'], system.at['electrons'])
    system.at['mesh'].loadSpeciesVTK(system.at['protons'], system.at['pic'])
    system.at['mesh'].loadSpeciesVTK(system.at['electrons'], system.at['pic'])

else:
    raise("Somehing is wrong here")

#Initialization of the previous step
old_system = copy.deepcopy(system)

## ---------------------------------------------------------------------------------------------------------------
# Set up of the system before the Main loop
## ---------------------------------------------------------------------------------------------------------------

#Injection of particles at all outer boundaries
#Electrons
out_e_n = c.E_N*numpy.ones((len(system.at['pic'].mesh.boundaries[0].location)))
out_drift_e_vel = numpy.zeros((len(system.at['pic'].mesh.boundaries[0].location),3))
out_thermal_e_vel = c.E_V_TH_MP*numpy.ones((len(system.at['pic'].mesh.boundaries[0].location)))
out_drift_e_vel[:,0] += c.E_V_SW
#Protons
out_p_n = c.P_N*numpy.ones((len(system.at['pic'].mesh.boundaries[0].location)))
out_drift_p_vel = numpy.zeros((len(system.at['pic'].mesh.boundaries[0].location),3))
out_thermal_p_vel = c.P_V_TH_MP*numpy.ones((len(system.at['pic'].mesh.boundaries[0].location)))
out_drift_p_vel[:,0] += c.P_V_SW

#Injection of particles at all inner boundaries
#Photoelectrons
in_phe_n = (c.PHE_FLUX/(-c.QE)*c.E_DT)*numpy.ones((len(system.at['pic'].mesh.boundaries[1].left)))
in_drift_phe_vel = numpy.zeros((len(system.at['pic'].mesh.boundaries[1].left),3))
in_thermal_phe_vel = c.PHE_V_TH_MP*numpy.ones((len(system.at['pic'].mesh.boundaries[1].left)))
in_drift_phe_vel[:,0] += 0.0
#Secondary emission electrons
in_see_n = 0
in_drift_see_vel = numpy.zeros((len(system.at['pic'].mesh.boundaries[1].location),3))
in_thermal_see_vel = c.SEE_V_TH_MP*numpy.ones((len(system.at['pic'].mesh.boundaries[1].location)))
in_drift_see_vel[:,0] += 0.0
#Protons
in_p_n = c.P_N*numpy.ones((len(system.at['pic'].mesh.boundaries[1].location)))
in_drift_p_vel = numpy.zeros((len(system.at['pic'].mesh.boundaries[1].location),3))
in_thermal_p_vel = c.P_V_TH_MP*numpy.ones((len(system.at['pic'].mesh.boundaries[1].location)))
in_drift_p_vel[:,0] += c.P_V_SW

#Merge into variables for all boundaries
#Electrons
e_n = [out_e_n]
thermal_e_vel = [out_thermal_e_vel]
drift_e_vel = [out_drift_e_vel]
#Ptotons
p_n = [out_p_n]
thermal_p_vel = [out_thermal_p_vel]
drift_p_vel = [out_drift_p_vel]

#User defined
#system.at['user'] = User_Defined(c.P_DT, -c.QE, c.MP, 0, c.P_SPWT, 10000, c.DIM, c.DIM, system.at['mesh'].nPoints, 0, "1")
#vel = numpy.zeros((10000,2))
#pos = numpy.zeros((10000,2))
#pos[:,0] = 1.0
#system.at['mesh'].boundaries[0].addParticles(system.at['user'], pos, vel)
#system.at['part_solver'].updateMeshValues(system.at['user'], extent = 1)

#for i, boundary in enumerate(system.at['mesh'].boundaries):
#    boundary.injectParticlesDummyBox(boundary.location, system.at['part_solver'], system.at['e_field'], system.at['electrons'], e_n[i], thermal_e_vel[i], drift_e_vel[i])
#    boundary.injectParticlesDummyBox(boundary.location, system.at['part_solver'], system.at['e_field'], system.at['protons'], p_n[i], thermal_p_vel[i], drift_p_vel[i])

system.at['mesh'].boundaries[0].injectParticlesDummyBox_PRon(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
                                                        system.at['e_field'], system.at['electrons'], e_n[0], thermal_e_vel[0], drift_e_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
system.at['mesh'].boundaries[0].injectParticlesDummyBox_PRon(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
                                                        system.at['e_field'], system.at['protons'], p_n[0], thermal_p_vel[0], drift_p_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
#system.at['mesh'].boundaries[0].injectParticlesDummyBox(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
#                                                        system.at['e_field'], system.at['electrons'], e_n[0], thermal_e_vel[0], drift_e_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
#system.at['mesh'].boundaries[0].injectParticlesDummyBox(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
#                                                        system.at['e_field'], system.at['protons'], p_n[0], thermal_p_vel[0], drift_p_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
flux, n_delta_phe = system.at['mesh'].boundaries[1].createDistributionAtBorder(system.at['mesh'].boundaries[1].left, system.at['part_solver'], system.at['photoelectrons'], in_phe_n)
system.at['mesh'].boundaries[1].injectParticlesAtPositions_smooth(flux, system.at['part_solver'], \
                                                           system.at['e_field'], system.at['photoelectrons'], n_delta_phe, in_thermal_phe_vel, system.at['photoelectrons'].dt)

###Test
#np = system.at['protons'].part_values.current_n
###Test positioning
##fig = plt.figure(figsize=(8,8))
##plt.scatter(system.at['protons'].part_values.position[:np, 0], system.at['protons'].part_values.position[:np,1], marker = '.')
##plt.title("protons")
##plt.show()
##Test velocity
#fig = plt.figure(figsize=(8,8))
#datamag = plt.hist(numpy.sqrt((system.at['protons'].part_values.velocity[:np,0]-c.P_V_SW)*(system.at['protons'].part_values.velocity[:np,0]-c.P_V_SW)+ \
#                              system.at['protons'].part_values.velocity[:np,1]*system.at['protons'].part_values.velocity[:np,1]), 81, alpha=0.5, label="protons")
#x = numpy.linspace(0, 5e5, num = 1000)
#def maxwellian_2D(x,m,T, amp):
#    return amp*m*x/c.K/T*numpy.exp(-m*x*x/2/c.K/T)
#plt.plot(x,maxwellian_2D(x, c.MP, c.P_T, 16*numpy.sum(datamag[0]**2)))
#plt.axvline(x=c.P_V_TH_MP*numpy.sqrt(2/3))
#plt.legend()
#plt.show()
#datamag = plt.hist(numpy.sqrt((system.at['electrons'].part_values.velocity[:np,0]-c.E_V_SW)*(system.at['electrons'].part_values.velocity[:np,0]-c.E_V_SW)+ \
#                              system.at['electrons'].part_values.velocity[:np,1]*system.at['electrons'].part_values.velocity[:np,1]), 81, alpha=0.5, label="electrons")
#x = numpy.linspace(0, 1e7, num = 1000)
#plt.plot(x, maxwellian_2D(x, c.ME, c.E_T, numpy.sum(datamag[0]**2)))
#plt.axvline(x=c.E_V_TH_MP*numpy.sqrt(2/3))
#plt.legend()
#plt.show()
#pdb.set_trace()

#Update of mesh values
system.at['part_solver'].updateMeshValues(system.at['electrons'], extent = 1, scatter_flux = 0)
#TODO: Done to see whether the system reaches steady state faster
system.at['part_solver'].updateMeshValues(system.at['photoelectrons'], extent = 1, scatter_flux = 0)
system.at['part_solver'].updateMeshValues(system.at['protons'], extent = 1, scatter_flux = 0)

## ---------------------------------------------------------------------------------------------------------------
# Species Dynamics Functions
## ---------------------------------------------------------------------------------------------------------------

def SWE():
    if system.at['ts'] == 0:
        advance_dict_e = system.at['part_solver'].advance(system.at['electrons'], [system.at['e_field']], [system.at['m_field']], extent = 0, types_boundary = ['open', 'mixed'], albedo = c.E_ALBEDO)
    elif system.at['ts']%c.VTK_TS == 0:
        advance_dict_e = system.at['part_solver'].advance(system.at['electrons'], [system.at['e_field']], [system.at['m_field']], extent = 1, types_boundary = ['open', 'mixed'], albedo = c.E_ALBEDO)
    else:
        advance_dict_e = system.at['part_solver'].advance(system.at['electrons'], [system.at['e_field']], [system.at['m_field']], extent = 1, update_dic = 0, types_boundary = ['open', 'mixed'], albedo = c.E_ALBEDO)

    #Injection of solar wind electrons at the outer boundary
    system.at['mesh'].boundaries[0].injectParticlesDummyBox_PRon(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
                                                            system.at['e_field'],system.at['electrons'], e_n[0], thermal_e_vel[0], drift_e_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
    #system.at['mesh'].boundaries[0].injectParticlesDummyBox(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
    #                                                        system.at['e_field'],system.at['electrons'], e_n[0], thermal_e_vel[0], drift_e_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
    return advance_dict_e

def PHE():
    if system.at['ts'] == 0:
        system.at['part_solver'].advance(system.at['photoelectrons'], [system.at['e_field']], [system.at['m_field']], extent = 0)
    elif system.at['ts']%c.VTK_TS == 0:
        system.at['part_solver'].advance(system.at['photoelectrons'], [system.at['e_field']], [system.at['m_field']], extent = 1)
    else:
        system.at['part_solver'].advance(system.at['photoelectrons'], [system.at['e_field']], [system.at['m_field']], extent = 1, update_dic = 0)

    #Injection of Photoelectrons at the Satellite
    flux, n_delta_phe = system.at['mesh'].boundaries[1].createDistributionAtBorder(system.at['mesh'].boundaries[1].left, system.at['part_solver'], system.at['photoelectrons'], in_phe_n)
    system.at['mesh'].boundaries[1].injectParticlesAtPositions_smooth(flux, system.at['part_solver'], \
                                                                       system.at['e_field'], system.at['photoelectrons'], n_delta_phe, in_thermal_phe_vel, system.at['photoelectrons'].dt)

def SEE(advance_dict_e):
    if system.at['ts'] == 0:
        system.at['part_solver'].advance(system.at['see'], [system.at['e_field']], [system.at['m_field']], extent = 0)
    elif system.at['ts']%c.VTK_TS == 0:
        system.at['part_solver'].advance(system.at['see'], [system.at['e_field']], [system.at['m_field']], extent = 1)
    else:
        system.at['part_solver'].advance(system.at['see'], [system.at['e_field']], [system.at['m_field']], extent = 1, update_dic = 0)

    #Injection of Secondary Emission Electrons at the Satellite
    new_see = system.at['see'].see_yield_constant(advance_dict_e['flux'])
    system.at['mesh'].boundaries[1].injectParticlesAtPositions_smooth(advance_dict_e['flux'], system.at['part_solver'],\
                                                               system.at['e_field'], system.at['see'], new_see, in_thermal_see_vel, system.at['see'].dt)

def SWP():
    #Proton motion
    if system.at['ts'] == 0:
        system.at['part_solver'].advance(system.at['protons'], [system.at['e_field']], [system.at['m_field']], extent = 0)
    elif system.at['ts']%c.VTK_TS == 0:
        system.at['part_solver'].advance(system.at['protons'], [system.at['e_field']], [system.at['m_field']], extent = 1)
    else:
        system.at['part_solver'].advance(system.at['protons'], [system.at['e_field']], [system.at['m_field']], extent = 1, update_dic = 0)
    #for i, boundary in enumerate(system.at['mesh'].boundaries):
    #    boundary.injectParticlesDummyBox(boundary.location, system.at['part_solver'], system.at['e_field'], system.at['protons'], p_n[i], thermal_p_vel[i], drift_p_vel[i])
    system.at['mesh'].boundaries[0].injectParticlesDummyBox_PRon(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
                                                            system.at['e_field'], system.at['protons'], p_n[0], thermal_p_vel[0], drift_p_vel[0], n_part = system.at['part_reformer'].partoptimum*2)
    #system.at['mesh'].boundaries[0].injectParticlesDummyBox(system.at['mesh'].boundaries[0].location, system.at['part_solver'], \
    #                                                        system.at['e_field'], system.at['protons'], p_n[0], thermal_p_vel[0], drift_p_vel[0], n_part = system.at['part_reformer'].partoptimum*2)

## ---------------------------------------------------------------------------------------------------------------
# Main loop
## ---------------------------------------------------------------------------------------------------------------

# try/except block to capture any error and save before the step before the crash. It also allows the simulation to be stopped.
try:
    while system.at['ts'] < c.NUM_TS:
        print('ts = ', system.at['ts'])
        #Execution time of loop step
        t0 = time.perf_counter()
        #Variables passed among the species
        advance_dict_e = None

        #Solving the fields
        system.at['e_field'].computeField([system.at['protons'], system.at['electrons'], system.at['photoelectrons'], system.at['see']])

        #Particle reforming for the different species
        if system.at['ts']%c.PR_E_TS == 0:
        #if system.at['ts']%c.PR_E_TS == 0 and system.at['ts'] != 0:
            system.at['part_reformer'].computeParticleReform(system.at['electrons'])
            system.at['part_reformer'].computeParticleReform(system.at['photoelectrons'])
            system.at['part_reformer'].computeParticleReform(system.at['see'])

        if system.at['ts']%c.PR_P_TS == 0:
        #if system.at['ts']%c.PR_P_TS == 0 and system.at['ts'] != 0:
            system.at['part_reformer'].computeParticleReform(system.at['protons'])

        # Electron motion
        if system.at['ts']%c.E_TS == 0:
            advance_dict_e = SWE()
            PHE()
            SEE(advance_dict_e)

        # Protons motion
        if system.at['ts']%c.P_TS == 0:
            SWP()

        #Output vtk
        if system.at['ts']%c.VTK_TS == 0 and system.at['ts'] != 0:
            system.at['part_solver'].updateMeshValues(old_system.at['electrons'], extent = 2)
            system.at['part_solver'].updateMeshValues(old_system.at['photoelectrons'], extent = 2)
            system.at['part_solver'].updateMeshValues(old_system.at['see'], extent = 2)
            system.at['part_solver'].updateMeshValues(old_system.at['protons'], extent = 2)
            out.saveVTK(system.at['mesh'], old_system.at, system.arrangeVTK())

        #if system.at['ts']%100000 == 100000-1:
        #    out.saveParticlesTXT(old_system.at, system.arrangeParticlesTXT())
        #if system.at['ts']%100000 == 100000-1:
        #    out.particleTracker(old_system.at['ts'], old_system.at['protons'], old_system.at['electrons'], old_system.at['photoelectrons'], old_system.at['see'])
    
        #Updating previous state
        deepcopy = Timing(copy.deepcopy)
        old_system = deepcopy(system)
    
        #Execution time of loop step and storage
        t1 = time.perf_counter()
        getattr(Timing, 'time_dict')['Global'] = t1-t0
        if system.at['ts']%100 == 0:
            out.saveTimes(system.at['ts'], getattr(Timing, 'time_dict'))
        Timing.reset_dict()

        #Advance in timestep
        system.at['ts'] += 1


except KeyboardInterrupt:
    out.savePickle(old_system.at, old_system.arrangePickle())
    print('Process aborted')
    print('Previous state succesfully saved')
    sys.exit()
except:
    out.savePickle(old_system.at, old_system.arrangePickle())
    print(sys.exc_info())
    print('Unexpected error')
    print('Previous state succesfully saved')
    raise

out.savePickle(system.at, system.arrangePickle())
print('Simulation finished')
print('Last state succesfully saved')
sys.exit()
