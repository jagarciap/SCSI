import numpy

#Physical constants
EPS_0 =8.85418782e-12   # F/m, vacuum permittivity
K     =1.38064825e-23          # J/K, Boltzmann constant
ME    =9.10938215e-31       # kg, electron mass
QE    = -1.602176565e-19      # C, electron charge
AMU   =1.660538921e-27  # kg, atomic mass unit
MP = 1.007276467*AMU # kg, proton mass
EV_TO_K=11604.525        # 1eV in Kelvin
g = 9.80665 #Gravity on Earth surface
MU_0 = 4*numpy.pi*1e-7

#Simulation time's parameters
NUM_TS = numpy.int(70000)   #Number of total steps in the system
P_DT= 2.658e-8            # time step size
E_DT = 8.86e-11        # time step for electron dynamics
E_TS = 1    #Timesteps required for execution of electron dynamics
P_TS = 300   #Timesteps required for protons dynamics
VTK_TS = 100 #Timesteps required for printing of VTK files
PR_E_TS = 10 #Timesteps used for checking the electrons per cell in the domain
PR_P_TS = 100 #Timesteps used for checking the electrons per cell in the domain

#Geometrical parameters for a rectangular outer boundary
#XMIN = 0.0
#XMAX = 100.0#5.0
#YMIN = -50.0#-205
#YMAX = 50.0#2.5
#DEPTH = 1.0
#NX = numpy.uint16(21)#10
#NY = numpy.uint16(21)#10

#Parameters for 2D_rm_sat
EPS_SAT = 3.3378   # F/m, satellite material relative permittivity

#Geometrical parameters for 2D_rm_sat
XMIN = 0.0
XMAX = 0.01848#5.0
YMIN = 0.0#-205
YMAX = 0.00154#2.5
XMINSAT = 3.5
XMAXSAT = 6.5
YMINSAT = -1.5#-205
YMAXSAT = 1.5#2.5
DX = 3.08e-4
DY = 3.08e-4
DEPTH = 0.00154

#Particle physical parameters
E_N = 1e14
E_T = 116000
E_V_TH_MP = numpy.sqrt(2*K*E_T/ME)     #Most Probable speed
E_V_SW = 0.0
E_ALBEDO = 0.0

P_N = 1e14
P_T = 200
P_V_TH_MP = numpy.sqrt(2*K*P_T/MP)     #Most Probable speed
P_V_SW = -4063.1

#Particle simulation parameters
P_SIZE = numpy.uint32(60e3)     #Size of the ions array
E_SIZE = numpy.uint32(60e3)     #Size of the electrons array
P_SPWT = 200
E_SPWT = 200

#Number of particles traced
NUM_TRACKED = 100
DIM = numpy.uint8(2)

#Magnetic field
B_STRENGTH = 0
B_DIM = 1

#Machine-related parameters
INDEX_PREC = 6

#Capacity Matrix file
CAPACITY_FILE = None
