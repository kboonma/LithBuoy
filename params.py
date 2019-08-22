#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
params.py 
-- basic paramters that can be changed from run to run 
-- call LithBuoy.py for the a time-stepping                    
#=======================================================================#
@author: Kittiphon Boonma, kboonma.b@gmail.com                 
Created 21/08/2019                                               
#=======================================================================#
"""

"""
-----------------------
0--Archon
1--Proton
2--Tecton
3--Oceanic  30 Ma
4--Oceanic 120 Ma
-----------------------
5--Slab  80 km thick
6--Slab 160 km thick
7--Oceanic  60 km thick 
8--Oceanic 110 km thick
-----------------------
"""
man_type=1
############################################
experiment_number = 'supertest' # for the .savefig title -- CHANGE to prevent overwriting
velocity = 80  # mm/year
nt = 51     # nº of steps to run
save_interval = 10     # save every n step
save_figure = True # True or False
plot_im_subplots = True
plot_check = False 
############################################

## PATH to save figures
#fig_path="~/ownCloud/PhD/delam_savefig/%s_%s_vel%d/" % (params.experiment_number,params.mantle_type,params.velocity)

#############################################
#input parameters
## Mesh setup:
h = 600e3               # height of model box [m]
w = 1500e3              # width of model box [m]
dx = 5e3                # discretization step [m]
dy = 5e3
angle=30
buoy_ylim=[-7e12, 7e12]

## Thermal parameters 
#Tlab = 1300     # [ºC]
#Tmoho = 500     # [ºC]
#Tbottom = 1500  # [ºC]
k_lith = 4.0  # Thermal conductivity [W/m.K]
Cp = 1000  # Specific heat capacity[J/K*kg]

kappa_lith = 1e-6
kappa_asth = 1e-5
############################################

if __name__ == '__main__':
	import lithbuoy