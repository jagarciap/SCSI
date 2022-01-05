# Data structures for electrons
from Species.species import Species
import constants as c


#User_Defined (Inherits from Species):
#
#Definition = Species that serves as dummy species for test porpuses.
#Attributes:
#	+Species attributes.
class User_Defined(Species):
    def __init__(self, n_dt, n_q, n_m, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_num_tracked, n_string):
        super().__init__("User defined species- "+n_string, n_dt, n_q, n_m, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_num_tracked)
