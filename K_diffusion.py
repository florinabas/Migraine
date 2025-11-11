# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:31:45 2023

@author: Floarea
"""

"""example of a cube (L**3) diffusing within a cube of extracellular space (Lecs**3) 
"""
from neuron import h, crxd as rxd
import numpy
from matplotlib import pyplot
from matplotlib_scalebar import scalebar
from scipy.special import erf

vf=0.07
tort=1.6

# plot functions
def boxoff(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def add_scalebar(ax, scale=1e-6):
    sb = scalebar.ScaleBar(scale)
    sb.location = 'lower left'
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.add_artist(sb)
    
h.load_file('stdrun.hoc')
sec = h.Section() #NEURON requires at least 1 section

# enable extracellular RxD
rxd.options.enable.extracellular = True

# simulation parameters
dx = 1.0    # voxel size
L = 9.0     # length of initial cube
Lecs = 21.0 # lengths of ECS

# define the extracellular region
extracellular = rxd.Extracellular(-Lecs/2., -Lecs/2., -Lecs/2.,
                                  Lecs/2., Lecs/2., Lecs/2., dx=dx,
                                  volume_fraction=vf, tortuosity=lambda x,y,z: tort)

# define the extracellular species
k_rxd = rxd.Species(extracellular, name='k', d=2.62, charge=1,
                    initial=lambda nd: 1.0 if abs(nd.x3d) <= L/2. and
                    abs(nd.y3d) <= L/2. and abs(nd.z3d) <= L/2. else 0.0)

# copy of the initial state to plot (figure 4a upper panel)
states_init = k_rxd[extracellular].states3d.copy()

# record the concentration at (0,0,0)
ecs_vec = h.Vector()
ecs_vec.record(k_rxd[extracellular].node_by_location(0, 0, 0)._ref_value)
# record the time
t_vec = h.Vector()
t_vec.record(h._ref_t)

h.finitialize()
h.dt = 0.1
h.continuerun(300) #run the simulation
# record states to plot (figure 4a lower panel)
states_mid = k_rxd[extracellular].states3d.copy()
#h.continuerun(200)


# plot the states in middle (z=0) of the cube (figure 4a)
fig = pyplot.figure()
ax1 = pyplot.subplot(2, 3, 1)
im1 = pyplot.imshow(states_init[:, :, int(numpy.ceil(extracellular._nz/2))]*1e3)
add_scalebar(ax1)
ax1.text(-0.1, 1.4, "A", transform=ax1.transAxes, size=12, weight='bold')
pyplot.colorbar()

ax2 = pyplot.subplot(2, 3, 4)
pyplot.imshow(states_mid[:, :, int(numpy.ceil(extracellular._nz/2))]*1e3)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
pyplot.colorbar()

pyplot.show()

