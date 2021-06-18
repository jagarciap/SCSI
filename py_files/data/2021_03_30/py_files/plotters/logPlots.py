import numpy
import matplotlib.pyplot as plt
import pdb
from scipy import stats, optimize
import sys
from matplotlib import rc

sys.path.insert(0,'..')

import constants as c

plt.rc('text', usetex=True)
plt.rc('axes', linewidth=1.5)
plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def SEE_to_SW_electron_production(last_line, filename = './log.txt'):
    see = []
    sw = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(last_line):
            if "Number of Electron - Solar wind eliminated - inner:" in lines[i]:
                new_sw = int(lines[i].split(":")[1].strip())
                if new_sw != 0:
                    sw.append(new_sw)
            elif "Total Electron - SEE" in lines[i]:
                see.append(int(lines[i-1].split(":")[1].strip()))
    #del sw[0]
    assert len(sw) == len(see), "There should be the same amount of items in the lists"
    division = [see[i]/sw[i] for i in range(len(sw))]
    print("Average SEE production per impacting electron", numpy.average(division))

    fig = plt.figure(figsize=(16,8))
    plt.plot(division, color = 'blue')
    plt.title("No. of SEE produced per impacting SW electron")
    plt.legend()
    plt.show()

SEE_to_SW_electron_production(44438)
