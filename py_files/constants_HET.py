import numpy

#Physical constants
EPS_0 =8.85418782e-12   # F/m, vacuum permittivity
K     =1.38064825e-23          # J/K, Boltzmann constant
ME    =9.10938215e-31       # kg, electron mass
QE    = -1.602176565e-19      # C, electron charge
AMU   =1.660538921e-27  # kg, atomic mass unit
MP = 1.007276467*AMU # kg, proton mass
M_XENON = 131.293*AMU     #kg, mass of neutral xenon
M_XENON_ION = M_XENON-ME      #kg, mass of singly-ionized xenon
EV_TO_K=11604.525        # 1eV in Kelvin
g = 9.80665 #Gravity on Earth surface
MU_0 = 4*numpy.pi*1e-7

#Simulation time's parameters
NUM_TS = numpy.int(200000)   #Number of total steps in the system
E_DT = 3e-9        # time step for the simulation
P_DT= 3e-8
HET_ION_DT = 45e-8
HET_N_DT = 1.5e-5
E_TS = 1    #Timesteps required for execution of electron dynamics
P_TS = 10   #Timesteps required for protons dynamics
HET_ION_TS = 150   #Timesteps required for HET ion dynamics
VTK_TS = 300 #Timesteps required for printing of VTK files

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
XMAX = 10.0#5.0
YMIN = -5.0#-205
YMAX = 5.0#2.5
XMINSAT = 3.5
XMAXSAT = 6.5
YMINSAT = -1.5#-205
YMAXSAT = 1.5#2.5
DX = 0.25
DY = 0.25
DEPTH = 1.0

#Parameters of the Hall Effect Thruster
R_MIN = 2.93e-2
R_MAX = 3.87e-2
EXIT_AREA = numpy.pi*(R_MAX**2-R_MIN**2)
EXIT_POT = 25        # In V, from Szabo (2013) paper

#Particle physical parameters
E_N = 7e9
E_T = 85.0*EV_TO_K
E_V_TH_MP = numpy.sqrt(2*K*E_T/ME)     #Most Probable speed
E_V_SW = 300e3
E_ALBEDO = 0.05

PHE_FLUX = 16e-3
PHE_T = 3*EV_TO_K
PHE_V_TH_MP = numpy.sqrt(2*K*PHE_T/ME)     #Most Probable speed
PHE_V_TH_AVG = numpy.sqrt(8*K*PHE_T/numpy.pi/ME)     #Thermal average
PHE_V_SW = 0
#PHE_N = PHE_FLUX/(PHE_V_TH*2/numpy.pi)/(-QE)
PHE_N = PHE_FLUX/(PHE_V_TH_AVG/numpy.sqrt(numpy.pi))/(-QE)     #This would be the density in space for FLUX and T.
PHE_N_WALL = PHE_FLUX/(-QE)*E_DT/DX   #This is the numeric density created in a mesh cell after one electron timestep

SEE_T = 2*EV_TO_K
SEE_V_TH_MP = numpy.sqrt(2*K*SEE_T/ME)

P_N = 7e9
P_T = 82.0*EV_TO_K
P_V_TH_MP = numpy.sqrt(2*K*P_T/MP)     #Most Probable speed
P_V_SW = 300e3

#Particle physical parameters - Thruster (BHT-1000)
HET_E_N = 1.6e17
HET_E_T = 11*EV_TO_K
HET_E_V_TH_MP = numpy.sqrt(2*K*HET_E_T/ME)
HET_E_V_DRIFT = 32.325e3

HET_ION_N = 1.6e17
HET_ION_T = 14*EV_TO_K
HET_ION_V_TH_MP = numpy.sqrt(2*K*HET_ION_T/M_XENON_ION)
HET_ION_V_DRIFT = 32.325e3

HET_N_N = 5.4e17
HET_N_T = 1*EV_TO_K
HET_N_V_TH_MP = numpy.sqrt(2*K*HET_N_T/M_XENON)
HET_N_V_DRIFT = 0.0

#Particle simulation parameters
P_SIZE = numpy.uint32(4e6)     #Size of the ions array
E_SIZE = numpy.uint32(4e6)     #Size of the electrons array
PHE_SIZE = numpy.uint32(2e6)     #Size of the electrons array
SEE_SIZE = numpy.uint32(4e6)     #Size of the electrons array
HET_ION_SIZE = numpy.uint32(1e6)
HET_E_SIZE = numpy.uint32(1e6)
P_SPWT = 2e6
E_SPWT = 2e6
PHE_SPWT = 1e5
SEE_SPWT = 3e5
HET_ION_SPWT = 1e10
HET_E_SPWT = 1e7

#Number of particles traced
NUM_TRACKED = 100
DIM = numpy.uint8(2)

#Magnetic field
B_STRENGTH = 0
B_DIM = 1

#Machine-related parameters
INDEX_PREC = 3

#Capacity Matrix file
CAPACITY_FILE = "inv_capacity_matrix_2021-03-30_16h14m.txt"
