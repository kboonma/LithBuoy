#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
lithcall.py -- this file assign miscellaneous parameters to be called 
upon by other scripts.                      
#=======================================================================#
@author: Kittiphon Boonma, kboonma.b@gmail.com                 
Created 21/08/2019                                                 
#=======================================================================#
"""

import numpy as np

##############################
# INPUT PARAMETERS
#############################

experiment_number = params.experiment_number # for the .savefig title -- CHANGE to prevent overwriting
velocity = params.velocity         # 4 20 40 60 80mm/year_
nt = params.nt                 # nº of steps to run
save_interval = params.save_interval        # save every n step
save_figure = params.save_figure #True       # True or False
plot_im_subplots = params.plot_im_subplots
plot_check = params.plot_check 

# Model domain setup
h = params.h             # height of model box [m]
w = params.w             # width of model box [m]
dx = params.dx               # discretization step [m]
dy = params.dy
angle=params.angle
alpha = np.deg2rad(angle)  # Subduction angle

buoy_ylim=params.buoy_ylim

#===============================================================

#w=len(x_profile)*(h / dy)

box_moho = 0e3             # Moho depth (w.r.t model box) (crust/mantle) [m]
hinge_ax = 350e3        # the starting point of deviation

## Thermal parameters ##
#Tlab = 1300     # [ºC]
#Tmoho = 500     # [ºC]
#Tbottom = 1500  # [ºC]
k_lith = params.k_lith  # Thermal conductivity [W/m.K]
Cp = params.Cp  # Specific heat capacity[J/K*kg]
kappa_lith = params.kappa_lith
kappa_asth = params.kappa_asth

#===============================================================


mantle_type=mantle.mantle_type #[mman] #'Ocean30ma'  # Type of mantle (Tecton, Archon,Proton)
dlab=mantle.dlab 
dmoho=mantle.dmoho
Tlab=mantle.Tlab
Tmoho=mantle.Tmoho
Tbottom=mantle.Tbottom
ref_depth_lith=mantle.ref_depth_lith
ref_rho_lith=mantle.ref_rho_lith
ref_P_lith=mantle.ref_P_lith
ref_T_lith=mantle.ref_T_lith
drhodT_lith=mantle.drhodT_lith
drhodP_lith=mantle.drhodP_lith
dT_lith=mantle.dT_lith
ref_depth_asth=mantle.ref_depth_asth
ref_rho_asth=mantle.ref_rho_asth
dT_asth=mantle.dT_asth
ref_P_asth=mantle.ref_P_asth
ref_T_asth=mantle.ref_T_asth
drhodT_asth=mantle.drhodT_asth
drhodP_asth=mantle.drhodP_asth

#===============================================================

# Slice depth profile
slice_pos1 = 20e3  # the position of the slice along x-axis in m
slice_pos2 = hinge_ax-50e3
slice_sec1 = int(slice_pos1 / dx)
slice_sec2 = int(slice_pos2 / dx)
slice_diff = 'no'  # yes or no for plotting differene in T and rho wrt initial

x_profile=[]
y_profile=[]


for i in range(np.int(h/dy)+1):
    x_profile.append(np.int(slice_sec2+(i/np.tan(alpha)))) #dx int(x_profile[i-1] + dy*np.tan(np.deg2rad(90- alpha)))
    y_profile.append(np.int(i))
y_plot=[]
x_plot=[]
for i in range(len(x_profile)):
    x_plot.append(x_profile[i]*dx )
for i in range(len(y_profile)):
    y_plot.append(i*dy)



#===============================================================
## PLOT SETTING ##
# Visualisation parameters
vel_vec = 'no'  # yes or no for showing velocity arrows
lab_contour = 'yes'
quiv_skip = 2  # velocity vector arrows spacing



## PATH to files
fig_path= os.getcwd()+'/data/%s_%s_vel%d/subplots/' % (experiment_number,mantle_type,velocity)
csv_path= os.getcwd()+'/data/%s_%s_vel%d/' % (experiment_number,mantle_type,velocity)
#fig_path="~/gpfs/scratch/csic08/csic08208/delam_savefig/%s_%s_vel%d/" % (experiment_number,mantle_type,velocity)
#fig_path = os.getcwd()+"/delam_savefig/%s_%s_vel%d/" % (mantle_type, experiment_number,velocity)
dir1 = os.path.expanduser(fig_path)
dir2 = os.path.expanduser(csv_path)
if not os.path.exists(dir1):
    os.makedirs(dir1)
if not os.path.exists(dir2):
    os.makedirs(dir2)
##
#########################################################################
# Initialisation and setting up
#########################################################################
##
# Mesh setup:
nx = (w / dx) + 1
ny = (h / dy) + 1
x = np.linspace(0, np.int(params.w), np.int(nx))  # array for the finite difference mesh
y = np.linspace(0, np.int(params.h), np.int(ny))
[xx, yy] = np.meshgrid(x, y)

secinmyr = 1e6 * 365 * 24 * 3600   # amount of seconds in 1 Myr
t = 0                 # set initial time to zero

# Velocity definition
# Define solid rotation velocity field:
tau = 100 * secinmyr      # 100-Myr convection overturn period in seconds
v0 = 2 * np.pi / tau       # angular velocity (in rad/sec)
xh = xx + 0.5 * h          # x-distances to horz middle of mesh (on xx mesh)
yh = yy + 0.5 * h
vx = np.zeros(np.shape(xx))
vy = np.zeros(np.shape(xx))


slab_thickness = dlab-dmoho 
dT_lith=(Tlab-Tmoho)/(slab_thickness) 
#dT_asth=(Tbottom-Tlab)/(h-dlab) 
beg_depth = 0#np.int(dmoho/dy)

#########################################################
## Make temperature profile (1D)##
[my, mx] = np.shape(xx)
T_prof = np.ones((my, 1))

T_prof[0:np.int(slab_thickness / dy) + 1] = np.reshape((Tmoho + dT_lith * \
                                                            (yy[0:np.int(slab_thickness/ dy)+1 , 0])), \
                                                            (np.int((slab_thickness + dy) / dy), 1))

T_prof[np.int(slab_thickness / dy) + 1:np.int(h / dy) + 1] = np.reshape((T_prof[np.int(slab_thickness / dy)] + dT_asth * 
                                                            (yy[np.int(slab_thickness/dy)+1:np.int(h/dy)+1, 0] - slab_thickness)), \
                                                            (np.int((h+dy)/dy - (slab_thickness+dy)/dy),1))

## Make density profile (1D) ##
T = T_prof[:]
rho_prof = np.ones((my, 1))
P = np.ones((my, 1))

for j in range(my):
    if(yy[j, 0] >= box_moho and yy[j, 0] <= slab_thickness):
        P[j] = (ref_rho_lith * 9.8 * yy[j, 0]) / 1e5
        rho_prof[j] = ref_rho_lith+(T[j]-ref_T_lith)*drhodT_lith+(P[j]-ref_P_lith)*drhodP_lith   

    if(yy[j, 0] > slab_thickness and yy[j, 0] <= h):
        P[j] = (ref_rho_asth * 9.8 * yy[j, 0]) / 1e5
        rho_prof[j] =  ref_rho_asth+(T[j]-ref_T_asth)*drhodT_asth+(P[j]-ref_P_asth)*drhodP_asth #  +

## Make initial temp. and density distribustion (2D) 
rho_init = np.ones(np.shape(xx))
T_init = np.ones(np.shape(xx))

rho_init[0::, :] = rho_prof[0::]
T_init[0::, :] = T_prof[0::]

drho_lab = int(rho_init[int((slab_thickness+dy)/dy),1] - rho_init[(int((slab_thickness)/dy)),1]) #int(rho_prof[np.where(yy==dlab)[0][-1]+1]) - int(rho_prof[np.where(yy==dlab)[0][-1]-1])


yy_prep = yy[:,0]+dmoho
dfout=np.zeros([len(yy_prep),3])
dfout[:,0] = yy_prep[:]
dfout[:,1] = T_init[:,0]
dfout[:,2] = rho_init[:,0]



#########################################################
# Calculate and assign misc. paramters
dbottom = h
k_asth = 1.9#(k_lith * dT_lith) / dT_asth
# Velocity
pres_vel = velocity * 1e-3 / (365 * 24 * 3600)
# Vel. at dipping angle in the haning slab
# Need resultant to be the same as the pres_vel so it needs to be splitted
# into x- and z- component
pres_vx = pres_vel * np.cos(alpha)
pres_vy = pres_vel * np.sin(alpha)

#=============================================================================
beta_dummy = 180 - ((np.rad2deg(alpha) / 2.0) + 90.0)
beta = np.deg2rad(beta_dummy)
# the starting point of deviation
hinge_ay = box_moho
hinge_a = [hinge_ax, hinge_ay]
# diffetent in x between a and b
ab_diff = (slab_thickness) / np.tan(beta)
hinge_bx = hinge_ax - ab_diff
hinge_by = slab_thickness 
hinge_b = [hinge_by, hinge_bx]
# Calculate the gradients from the angles
gradient1 = np.tan(beta)
gradient2 = np.tan(alpha)
# Calculate the width of the hanging slab
wid1 = (slab_thickness) / np.tan(alpha)
wid2 = (slab_thickness) / np.tan(beta)
wid = wid1 + wid2

#============================================================================
#============================ END OF CODE ===================================
#============================================================================
