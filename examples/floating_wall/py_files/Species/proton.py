# Data structures for protons
from Species.species import Species
import constants as c

#Proton (Inherits from Species):
#
#Definition = Species that take care of protons.
#Attributes:
#	+Species attributes.
class Proton(Species):
    def __init__(self, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, n_string):
        super().__init__("Proton"+n_string ,c.P_DT, -c.QE, c.MP, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked)

#Proton_SW (Inherits from Species):
#
#Definition = Species that take care of protons coming from solar wind.
#Attributes:
#	+type (string) = "Proton - Solar wind"
#	+Species attributes.
class Proton_SW(Proton):
    def __init__(self, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_fluxind, n_num_tracked = 0):
        super().__init__(n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_fluxind, n_num_tracked, " - Solar wind")
