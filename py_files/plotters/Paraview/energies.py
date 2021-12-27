from paraview.simple import *

#names = ['0_ts*', '0-0_ts*', '0-0-0_ts*']
#names = ['TemporalStatistics1', 'TemporalStatistics2', 'TemporalStatistics3']
names = ['TemporalStatistics1', 'TemporalStatistics2']
species = ["Proton - Solar wind", "Electron - Solar wind", "Electron - Photoelectron", "Electron - SEE"]
K = 1.38064852e-23
QE = 1.60217662e-19
charges = [-QE, QE, QE, QE]

for name in names:
    acs = FindSource(name)
    SetActiveSource(acs)
    for specie, charge in zip(species, charges):
        #Kinetic energy
        calculator = Calculator(Input=acs)
        calculator.ResultArrayName = specie+"-K"
        if acs.PointData.GetArray(specie+"-density") is None:
            calculator.Function = "({:s}-density_average)/2*mag({:s}-velocity_average)^2".format(specie, specie, specie)
        else:
            calculator.Function = "({:s}-density)/2*mag({:s}-velocity)^2".format(specie, specie, specie)
        renderView1 = GetActiveViewOrCreate('RenderView')
        calculatorDisplay = Show(calculator, renderView1)
        renderView1.Update()
        #Internal energy
        calculator = Calculator(Input=acs)
        calculator.ResultArrayName = specie+"-U"
        calculator.Function = "3/2*({:s}-density_average)*{:e}*({:s}-temperature_average)".format(specie, K, specie)
        renderView1 = GetActiveViewOrCreate('RenderView')
        calculatorDisplay = Show(calculator, renderView1)
        renderView1.Update()
        #Electric potential energy
        e_field = "Electric - Elecrostatic_2D_cm_sat_cond-potential_average"
        calculator = Calculator(Input=acs)
        calculator.ResultArrayName = specie+"-E"
        calculator.Function = "({:s}-density_average)*{:e}*({:s})".format(specie, charge, e_field)
        renderView1 = GetActiveViewOrCreate('RenderView')
        calculatorDisplay = Show(calculator, renderView1)
        renderView1.Update()
        #Total energy
        calculator = Calculator(Input=acs)
        calculator.ResultArrayName = specie+"-T"
        calculator.Function = "({:s}-K)+({:s}-U)+({:s}-E)".format(specie, specie, specie)
        renderView1 = GetActiveViewOrCreate('RenderView')
        calculatorDisplay = Show(calculator, renderView1)
        renderView1.Update()
