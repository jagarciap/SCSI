# File that graph the particles for the 'particle_tracker' tool
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import matplotlib.ticker as ticker
import numpy
import os
from scipy import spatial
import sys
import pdb

sys.path.insert(0,'..')

import constants as c

#colors = {'Proton - Solar wind': 'red', 'Electron - Solar wind': 'blue', 'Electron - Photoelectron': 'black'}
colors = {'Proton': 'red', 'Electron': 'white'}

class Species_Plotter(object):
    def __init__(self, name, num_tracked, pos_dim, ts):
        self.name = name
        self.pos = numpy.zeros((num_tracked, pos_dim, ts))
        self.marked = []
        self.graph = None


def load_files(ind):
    #Preparing instances of Species_plotter in order to store data
    cwd_base = os.getcwd().rsplit(sep = os.sep, maxsplit = 1)
    cwd = os.path.join(cwd_base[0], 'particle_tracker','')
    filename = cwd+'ts={:05d}.dat'.format(ind[0])
    f = open(filename)
    f.readline()
    names = f.readline().split(sep = '\t')[:-1]
    f.close()
    #Correction due to # at the beginning of the line
    names[0] = names[0][2:]
    #Creation of classes and assigning first timestep
    num_species = len(names)
    temparray = numpy.loadtxt(filename, delimiter = '\t')
    pos_dim = int(numpy.shape(temparray)[1]/num_species)
    species = []
    for i in range(num_species):
        n_species = Species_Plotter(names[i], numpy.shape(temparray)[0], pos_dim, len(ind))
        n_species.pos[:,:,0] = temparray[:, pos_dim*i:pos_dim*(i+1)]
        species.append(n_species)
    #Rest of timesteps
    for i in range(1, len(ind)):
        filename = cwd+'ts={:05d}.dat'.format(ind[i])
        temparray = numpy.loadtxt(filename)
        for j in range(num_species):
            species[j].pos[:,:,i] = temparray[:,pos_dim*j:pos_dim*(j+1)]

    return species

def create_figure():
    augment = 20
    inch = 0.0254
    DX = (c.XMAX-c.XMIN)
    DY = (c.YMAX-c.YMIN)
    dx = c.DX
    dy = c.DY
    fig = figure(figsize=(DX/inch*augment,DY/inch*augment))
    ax = fig.add_subplot(111)

    #plt.axvline(x = c.XMINSAT, ymin = (c.YMINSAT-c.YMIN)/DY, ymax = (c.YMAXSAT-c.YMIN)/DY, color = 'black')
    #plt.axvline(x = c.XMAXSAT, ymin = (c.YMINSAT-c.YMIN)/DY, ymax = (c.YMAXSAT-c.YMIN)/DY, color = 'black')
    #plt.axhline(y = c.YMINSAT, xmin = c.XMINSAT/DX, xmax = c.XMAXSAT/DY, color = 'black')
    #plt.axhline(y = c.YMAXSAT, xmin = c.XMINSAT/DX, xmax = c.XMAXSAT/DY, color = 'black')

    ax.xaxis.set_major_locator(ticker.NullLocator())
    gridx = numpy.arange(c.XMIN+dx, c.XMAX, dx)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(gridx))
    ax.set_xlim(c.XMIN, c.XMAX)

    ax.yaxis.set_major_locator(ticker.NullLocator())
    gridy = numpy.arange(c.YMIN+dy, c.YMAX, dy)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(gridy))
    ax.set_ylim(c.YMIN, c.YMAX)

    ax.grid(True, which = 'minor')

    return fig, ax

def choose_point(x,y, data):
    tree = spatial.KDTree(data)
    return tree.query([x,y])[1]


class IndexTracker:
    def __init__(self, ax, species):
        self.ax = ax
        ax.set_title('use RIGHT to advance, LEFT to move back')

        self.slices = numpy.shape(species[0].pos)[2]
        self.species = species
        self.num_species = len(species)
        self.ind  = 0

        print("Creating plots")
        for i in range(self.num_species):
            self.species[i].graph, = ax.plot(self.species[i].pos[:,0,self.ind], self.species[i].pos[:,1,self.ind], color = colors[self.species[i].name], marker = '.', linestyle = '', picker = 3.0)
        self.update()

    def keyevent(self, event):
        if event.key=='right':
            self.ind = numpy.clip(self.ind+1, 0, self.slices-1)
        elif event.key=='left':
            self.ind = numpy.clip(self.ind-1, 0, self.slices-1)

        self.update()

    def scrollevent(self, event):
        try:
            if event.button == 'up' or event.button == 'down':
                self.ind = numpy.clip(self.ind-int(event.step), 0, self.slices-1)
            else:
                print('Passing through scroll')
                pickevent(event)

            self.update()

        except RuntimeError:
            pass

    def pickevent(self, event):
        line = event.artist
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        for i in range(self.num_species):
            if line == self.species[i].graph:
                num = choose_point(x,y,self.species[i].pos[:,:,self.ind])
                graph, = ax.plot(self.species[i].pos[num,0,self.ind], self.species[i].pos[num, 1, self.ind], marker = 'o', markerfacecolor = colors[self.species[i].name],\
                        markeredgecolor = colors[self.species[i].name], linestyle = '-', color = colors[self.species[i].name], picker = 3.0)
                self.species[i].marked.append((num,self.ind,graph))
                print('Index of {}:{:d}'.format(self.species[i].name, num))
                break
            else:
                t = False
                j = 0
                while j < len(self.species[i].marked) and t == False:
                    if line == self.species[i].marked[j][2]:
                        t = True
                        del self.species[i].marked[j]
                        ax.lines.remove(line)
                    j += 1
            if t == True:
                break
        self.update()

    def update(self):
        for i in range(self.num_species):
            self.species[i].graph.set_data(self.species[i].pos[:,0,self.ind], self.species[i].pos[:,1,self.ind])
            for tpl in self.species[i].marked:
                path = self.species[i].pos[tpl[0], :, tpl[1]:self.ind+1]
                path = numpy.reshape(path, (2, self.ind-tpl[1]+1)).T
                tpl[2].set_data(path[:,0], path[:,1])
        ax.set_ylabel('slice %s'%self.ind)        
        plt.pause(0.001)



ts = numpy.arange(1,43463,300)

fig, ax = create_figure()
tracker = IndexTracker(ax, load_files(ts))
fig.canvas.mpl_connect('key_press_event', tracker.keyevent)
fig.canvas.mpl_connect('scroll_event', tracker.scrollevent)
fig.canvas.mpl_connect('pick_event', tracker.pickevent)

show()
