"""
This is just a file for figuring out how to place cells in a ring, with a specified space (in micrometers^2) per cell.
This is not used directly in the simulation code, but is the basis for the setCellLocations method in network_class.py.
This file can be used to visualize the spatial arrangement of cells.
"""
from __future__ import division
from matplotlib import pyplot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

area_cell=100 #micrometers^2 per cell
dist_cell=np.sqrt(area_cell) #closest linear distance between cells, if laid out on a square grid

Npyr=500 #number of pyr cells to lay out
Ninh=100
Ntot=Npyr+Ninh

pyr2inh_ratio=Npyr/Ninh
#for ease of coding, I am going to require that Npyr be a multiple of Ninh
assert Npyr%Ninh==0, "Error: Number of pyramidal cells must be a multiple of the number of inhibitory cells."

#fist column is x-coordinates, second column y-coordinates, third column z-coordinates
pyrcoords=np.zeros((Npyr,3))
inhcoords=np.zeros((Ninh,3))

#lay out cells in rings around origin. Each ring will be a distance dist_cell further from the origin than the previous one, and within
#each ring, cells will be spaced 'dist_cell' from one another
ring_ind=1 #current ring index. Radius of ring will be ring_ind*dist_cell
intraring_loc=0 #lattice location within present ring
cells_laid=0 #number of cells laid out
pyrcells_laid=0
inhcells_laid=0
cells_in_ring=int(np.floor(2.0*np.pi*ring_ind)) #cells to place in next ring; take circumference divided by cell spacing: 2*np.pi*(ring_ind*dist_cell)/dist_cell
phi=np.linspace(0,2*np.pi,cells_in_ring+1) #calculate angular coordinate for each cell; "+1" results in a "dummy cell" being placed at angle of 2*pi, so that we don't place an actual cell there. 
phi=phi+(phi[1]-phi[0])/2.0 #rotate everything so that first cell isn't always placed along the y-axis
while (cells_laid<Ntot):
    
    #set x- and y- coordinates
    if(cells_laid % (pyr2inh_ratio+1) == 0):
        inhcoords[inhcells_laid,1]=(dist_cell*ring_ind)*np.cos(phi[intraring_loc])
        inhcoords[inhcells_laid,2]=(dist_cell*ring_ind)*np.sin(phi[intraring_loc])
        inhcells_laid += 1
    else:
        pyrcoords[pyrcells_laid,1]=(dist_cell*ring_ind)*np.cos(phi[intraring_loc])
        pyrcoords[pyrcells_laid,2]=(dist_cell*ring_ind)*np.sin(phi[intraring_loc])        
        pyrcells_laid += 1
        
    intraring_loc += 1
    cells_laid += 1
    
    if(intraring_loc == cells_in_ring and cells_laid != Ntot): #condition for having completed a ring
        ring_ind += 1 #move out to the next ring
        intraring_loc=0 #reset lattice location within present ring
        
        cells_in_ring=int(np.floor(2.0*np.pi*ring_ind)) #cells to place in next ring; take circumference divided by cell spacing: 2*np.pi*(ring_ind*dist_cell)/dist_cell
        if(cells_laid+cells_in_ring > Ntot): #if the next ring can accommodate more neurons than is necessary, then have it only house the remaining neurons (this will make them symmetrically distributed around the ring)
            cells_in_ring = Ntot - cells_laid
            
        phi=np.linspace(0,2*np.pi,cells_in_ring+1)
        phi=phi+(phi[1]-phi[0])/2.0
        

##finish laying out remaining cells
#cells_in_ring=Npyr-cells_laid
#phi=np.linspace(0,2*np.pi,cells_in_ring+1) 
#phi=phi+(phi[1]-phi[0])/2.0 
#pyrcoords[cells_laid:cells_laid+cells_in_ring,1]=(dist_cell*ring_ind)*np.cos(phi[:-1])
#pyrcoords[cells_laid:cells_laid+cells_in_ring,2]=(dist_cell*ring_ind)*np.sin(phi[:-1])
#
##now lay out basket cells; make them in a concentric ring, 
#inhcoords=np.zeros((Ninh,3))
#inhcoords[:,0]=10.0 #displace the basket cells from the y-z plane
#ring_ind=1 #current ring index. Radius of ring will be ring_ind*dist_cell
#cells_laid=0 #number of cells laid out
#cells_in_ring=int(np.floor(2.0*np.pi*ring_ind)) #cells to place in next ring; take circumference divided by cell spacing: 2*np.pi*(ring_ind*dist_cell)/dist_cell
#while (cells_laid+cells_in_ring<Ninh):
#    phi=np.linspace(0,2*np.pi,cells_in_ring+1) #calculate angular coordinate for each cell; "+1" results in a "dummy cell" being placed at angle of 2*pi, so that we don't place an actual cell there. 
#    phi=phi+(phi[1]-phi[0])/2.0 #rotate everything so that first cell isn't always placed along the x-axis
#    #set x- and y- coordinates
#    inhcoords[cells_laid:cells_laid+cells_in_ring,1]=(dist_cell*ring_ind)*np.cos(phi[:-1])
#    inhcoords[cells_laid:cells_laid+cells_in_ring,2]=(dist_cell*ring_ind)*np.sin(phi[:-1])
#    
#    cells_laid += cells_in_ring
#    ring_ind += 1
#    cells_in_ring=int(np.floor(2.0*np.pi*ring_ind))
#
##finish laying out remaining basket cells
#cells_in_ring=Ninh-cells_laid
#phi=np.linspace(0,2*np.pi,cells_in_ring+1) 
#phi=phi+(phi[1]-phi[0])/2.0 
#inhcoords[cells_laid:cells_laid+cells_in_ring,1]=(dist_cell*ring_ind)*np.cos(phi[:-1])
#inhcoords[cells_laid:cells_laid+cells_in_ring,2]=(dist_cell*ring_ind)*np.sin(phi[:-1])
#    
##pyplot.figure()
##pyplot.plot(coords[:,0],coords[:,1],'.')
        
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pyrcoords[:,0], pyrcoords[:,1], pyrcoords[:,2], zdir='z', color='b', s=10, label='pyramidal')
ax.scatter(inhcoords[:,0], inhcoords[:,1], inhcoords[:,2], zdir='z', color='r', s=10, label='inhibitory')
ax.legend()

