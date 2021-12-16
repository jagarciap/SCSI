# Data structures for electrons
import numpy
import pdb

from Species.species import Species
import constants as c
import see_yield as sy


#Electron (Inherits from Species):
#
#Definition = Species that take care of electrons.
#Attributes:
#	+Species attributes.
class Electron(Species):
    def __init__(self, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_fluxind, n_num_tracked, n_string):
        super().__init__("Electron"+n_string, c.E_DT ,c.QE, c.ME, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_fluxind, n_num_tracked)


#Electron_SW (Inherits from Electron):
#
#Definition = Species that take care of electrons coming from solar wind.
#Attributes:
#	+type (string) = "Electron - Solar wind"
#	+Species attributes.
class Electron_SW(Electron):
    def __init__(self, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked = 0):
        super().__init__(n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, " - Solar wind")


#Photoelectron (Inherits from Electron):
#
#Definition = Species that take care of electrons created from the Photoelectric effect.
#Attributes:
#	+type (string) = "Electron - Photoelectron"
#       +temperature (double) = Photoelectron temperature when they are created
#       +flux (double) = Nominal flux of photoelectrons being created at the designated surface.
#	+Species attributes.
class Photoelectron(Electron):
    def __init__(self, n_temperature, n_flux, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked = 0):
        self.temperature = n_temperature
        self.flux = n_flux
        super().__init__(n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, " - Photoelectron")


#Secondary_Emission_Electron (Inherits from Electron):
#
#Definition = Species that take care of electrons created from the impact of thermal electrons in surfaces.
#Attributes:
#	+type (string) = "Electron - SEE"
#       +temperature (double) = SEE temperature when they are created
#       +yields ([double,double]) = Matrix that stores a discrete version of the Whipple formula.
#	+Species attributes.
class Secondary_Emission_Electron(Electron):
    def __init__(self, n_temperature, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked = 0):
        self.temperature = n_temperature
        filename = './data/SEE_yield.dat'
        self.yields = numpy.loadtxt(filename, dtype = numpy.float)
        super().__init__(n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, " - SEE")

#       +see_yield_constant(flux_vals, SEEY) = This function receives 'flux_vals', a group of lists of parameters related with the flux of incoming electrons to the spacecraft, one entry in the lists per electron. 
#           Then, for each thermal electron that impacted on the surface, it calculates the amount of Secondary Emission Electrons created.
#           In this case, the Secondary Emission Electron Yield is kept constant at SEEY (keyword parameter).
    def see_yield_constant(self, flux_vals, SEEY = 2.9):
        new_yields = SEEY*flux_vals[0][:,3]/c.SEE_SPWT
        new_see, remainder = numpy.divmod(new_yields, 1)
        rand_spread = numpy.random.rand(len(new_yields))
        new_see += numpy.where(rand_spread < remainder, 1, 0)
        return new_see.astype(numpy.int)

#       +see_yield_constant(flux_vals, SEEY) = This function receives 'flux_vals', a group of lists of parameters related with the flux of incoming electrons to the spacecraft, one entry in the lists per electron. 
#           Then, for each thermal electron that impacted on the surface, it calculates the amount of Secondary Emission Electrons created.
#           In this case, the Secondary Emission Electron Yield calculated with the Whipple formula, through the 'yields' matrix.
#           Source: E. Whipple, Rep. Prog. Phys. 44, 1197 (1981)
    def see_yield_whipple(self, flux_vals):
        E = self.m/2*(flux_vals[1]*flux_vals[1]+flux_vals[2]*flux_vals[2])
        new_yields = flux_vals[0][:,3]/c.SEE_SPWT*sy.valuesToIndices(E, flux_vals[3], self.yields)
        new_see, remainder = numpy.divmod(new_yields, 1)
        rand_spread = numpy.random.rand(len(new_yields))
        new_see += numpy.where(rand_spread < remainder, 1, 0)
        return new_see.astype(numpy.int)
    

#Electron_HET (Inherits from Electron):
#
#Definition = Species that take care of electrons created in the plume region of the Hall Effect Thruster.
#Attributes:
#	+type (string) = "Electron - HET"
#       +temperature (double) = HET electron temperature when they are created
#       +flux (double) = Nominal flux of HET electrons being created at the designated surface.
#	+Species attributes.
class Electron_HET(Electron):
    def __init__(self, n_temperature, n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked = 0):
        self.temperature = n_temperature
        super().__init__(n_debye, n_spwt, n_max_n, n_pos_dim, n_vel_dim, n_nPoints, n_flux_ind, n_num_tracked, " - HET")
