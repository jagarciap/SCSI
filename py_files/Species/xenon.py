# Data structures for Xenon species
from Species.species import Species
import constants as c

#Xenon (Inherits from Species):
#
#Definition = Species that take care of xenon atoms.
#Attributes:
#	+Species attributes.
class Xenon_Neutral(Species):
    def __init__(self, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, n_string):
        super().__init__("Xe - Neutral"+n_string ,c.HET_N_DT, 0.0, c.M_XENON, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked)

#Xenon_Ion (Inherits from Species):
#
#Definition = Species that take care of single-ionized xenon atoms.
#Attributes:
#	+Species attributes.
class Xenon_Ion(Species):
    def __init__(self, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, n_string):
        super().__init__("Xe - Ion"+n_string ,c.HET_ION_DT, -c.QE, c.M_XENON_ION, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked)

