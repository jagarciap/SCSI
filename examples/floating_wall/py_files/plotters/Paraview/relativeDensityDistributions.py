from paraview.simple import *

#names = ['0_ts*', '0-0_ts*', '0-0-0_ts*']
names = ['TemporalStatistics1', 'TemporalStatistics2', 'TemporalStatistics3']
species = ["Proton - Solar wind", "Electron - Solar wind", "Electron - Photoelectron", "Electron - SEE"]
n0s = [7e9, 7e9, 3e10, 1.5e10]

for name in names:
    acs = FindSource(name)
    SetActiveSource(acs)
    for specie, n0 in zip(species, n0s):
        calculator = Calculator(Input=acs)
        calculator.ResultArrayName = specie+"-Relative"
        if acs.PointData.GetArray(specie+"-density") is None:
            calculator.Function = "({:s}-density_average)/{:e}".format(specie, n0)
        else:
            calculator.Function = "({:s}-density)/{:e}".format(specie, n0)
        renderView1 = GetActiveViewOrCreate('RenderView')
        calculatorDisplay = Show(calculator, renderView1)
        renderView1.Update()
