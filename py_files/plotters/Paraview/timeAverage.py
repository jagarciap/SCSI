from paraview.simple import *

firststep = 20
names = ['0_ts*', '0-0_ts*', '0-0-0_ts*']

for name in names:
    acs = FindSource(name)
    SetActiveSource(acs)
    laststep = int(acs.TimestepValues[-1])
    extractTimeSteps = ExtractTimeSteps(Input=acs)
    extractTimeSteps.TimeStepIndices = [i for i in range(firststep, laststep+1)]
    temporalStatistics = TemporalStatistics(Input=extractTimeSteps)
    renderView1 = GetActiveViewOrCreate('RenderView')
    temporalStatisticsDisplay = Show(temporalStatistics, renderView1)
    renderView1.Update()
