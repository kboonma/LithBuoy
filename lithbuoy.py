#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
LithBuoy.py 
-- this file contains subroutines and the time-stepping of the simulation 
-- an option to visualise outputs along with the calculation
#=======================================================================#
@author: Kittiphon Boonma, kboonma.b@gmail.com                 
Created 21/08/2019                                               
#=======================================================================#
"""

# Setup libraries
from __future__ import division
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pylab import rcParams
from copy import copy, deepcopy
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import csv
import pickle
import matplotlib.ticker as tkr
#from Stokes2D import Stokes2Dfunc 
from matplotlib.colors import LinearSegmentedColormap
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams['mathtext.fontset'] = u'cm'
#Colormaps
temp_data = np.loadtxt("colormap/roma.txt")
CBtemp_map = LinearSegmentedColormap.from_list('CBtemp', temp_data[::-1])

dens_data = np.loadtxt("colormap/lajolla.txt")
CBdens_map = LinearSegmentedColormap.from_list('CBdens', dens_data[::-1])

rcParams['figure.figsize'] = (15, 8)
rcParams['font.size'] = 8


##############################
# INPUT PARAMETERS
#############################
import params  #import input file
import mantle

"""
For sequential run, BASH Script
Testing del_rho 
"""
#mman = np.int(sys.argv[1])
#velocity = np.int(sys.argv[2])
#rr=np.int(sys.argv[3]) # prints var1)

#============================================================================
experiment_number = params.experiment_number # for the .savefig title -- CHANGE to prevent overwriting
velocity = params.velocity         # 4 20 40 60 80mm/year_
nt = params.nt                 # nº of steps to run
save_interval = params.save_interval        # save every n step
save_figure = params.save_figure #True       # True or False
plot_im_subplots = params.plot_im_subplots
plot_check = params.plot_check 

## Model domain setup
h = params.h             # height of model box [m]
w = params.w             # width of model box [m]
dx = params.dx               # discretization step [m]
dy = params.dy
angle=params.angle
alpha = np.deg2rad(angle)  # Subduction angle
buoy_ylim=params.buoy_ylim
box_moho = 0e3             # Moho depth (w.r.t model box) (crust/mantle) [m]
hinge_ax = 350e3        # the starting point of deviation

## Thermal parameters ##
k_lith = params.k_lith  # Thermal conductivity [W/m.K]
Cp = params.Cp  # Specific heat capacity[J/K*kg]
kappa_lith = params.kappa_lith
kappa_asth = params.kappa_asth
#============================================================================
## Mantle parameters from mantly.py
mantle_type=mantle.mantle_type #[mman]
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
#============================================================================
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
#============================================================================
## PLOT SETTING 
# Visualisation parameters
vel_vec = 'no'  # yes or no for showing velocity arrows
lab_contour = 'yes'
quiv_skip = 2  # velocity vector arrows spacing

#============================================================================
## PATH to files
fig_path= os.getcwd()+'/data/%s_%s_vel%d/subplots/' % (experiment_number,mantle_type,velocity)
dat_path= os.getcwd()+'/data/%s_%s_vel%d/' % (experiment_number,mantle_type,velocity)
csv_path= os.getcwd()+'/csv/' 
dir1 = os.path.expanduser(fig_path)
dir2 = os.path.expanduser(dat_path)
dir3 = os.path.expanduser(csv_path)
if not os.path.exists(dir1):
    os.makedirs(dir1)
if not os.path.exists(dir2):
    os.makedirs(dir2)
if not os.path.exists(dir3):
    os.makedirs(dir3)
#============================================================================
# Initialisation and setting up
#============================================================================
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
#dT_lith=(Tlab-Tmoho)/(slab_thickness) 
#dT_asth=(Tbottom-Tlab)/(h-dlab) 
beg_depth = 0#np.int(dmoho/dy)
dbottom = h
k_asth = 1.9#(k_lith * dT_lith) / dT_asth

# Velocity
pres_vel = velocity * 1e-3 / (365 * 24 * 3600)
# Vel. at dipping angle in the haning slab
# Need resultant to be the same as the pres_vel so it needs to be splitted
# into x- and z- component
pres_vx = pres_vel * np.cos(alpha)
pres_vy = pres_vel * np.sin(alpha)

#============================================================================
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
print (drho_lab)

yy_prep = yy[:,0]+dmoho
dfout=np.zeros([len(yy_prep),3])
dfout[:,0] = yy_prep[:]
dfout[:,1] = T_init[:,0]
dfout[:,2] = rho_init[:,0]

print("T-profile saved in directory %s" % dir2)
os.chdir(dir2)
np.savetxt('Trho_%s_%s_vel_%d.txt' % (mantle_type, experiment_number,velocity),dfout,delimiter=',')

#============================================================================

## ------------- BEGIN OF SUB-ROUTINES DEFINITIONS --------------------- ##

#============================================================================
# Func: Get initial velocity distribution
#============================================================================
def get_vel_init():
    # Purpose: This subroutine ontructs initial temperature and velocity
    #          structures.
    [my, mx] = np.shape(xx)

    for i in range(mx):
        for j in range(my):
            # Backgroud velocity of the whole box
            if (xx[j, i] >= 0 and xx[j, i] <= w
                    and yy[j, i] >= 0 and yy[j, i] <= h):
                vx[j, i] = 0 * xh[j, i]
                vy[j, i] = 0 * yh[j, i]
            # Left side - Lithospheric slab
            if (yy[j, i] >= 0 and yy[j, i] <= slab_thickness
                    and xx[j, i] <= hinge_ax - (yy[j, i] - box_moho) / gradient1):
                vx[j, i] = 0 * xh[j, i] + pres_vel
                vy[j, i] = 0 * yh[j, i]
            # Triangular block (joint)
            if (yy[j, i] >= 0 and yy[j, i] <= slab_thickness
                    and xx[j, i] >= hinge_ax - (yy[j, i] - box_moho) / gradient1
                    and xx[j, i] <= hinge_ax + (yy[j, i] - box_moho) / gradient2):
                vx[j, i] = 0 * xh[j, i] + pres_vx
                vy[j, i] = 0 * yh[j, i] + pres_vy
    return [vx, vy]
#============================================================================
# Func: move_vel
#============================================================================
def move_vel():
    # Purpose: This subroutine will move the hanging (velocity structure)
    #         slab down in the defined subducton angle. This vel structure
    #         will move down with the (resultant vel. vector) velocity
    #         defined as pres_vel
    [my, mx] = np.shape(xx)
    for i in range(mx):
        for j in range(my):
            # Hanging lithospheric slab
            if (yy[j, i] > slab_thickness and yy[j, i] < slab_thickness + (pres_vel * np.sin(alpha) * t)
                    and xx[j, i] <= hinge_bx + wid + (yy[j, i] - slab_thickness) / gradient2
                    and xx[j, i] >= hinge_bx + (yy[j, i] - slab_thickness) / gradient2):
                vx[j, i] = 0 * xh[j, i] + pres_vx
                vy[j, i] = 0 * yh[j, i] + pres_vy
    return [vx, vy]
#============================================================================
# Func: Advection -- rho and T
#============================================================================
def advect(temp,dens,P):
    # Purpose: This subroutine will move the hanging (velocity structure)
    #         slab down in the defined subducton angle. This vel structure
    #         will move down with the (resultant vel. vector) velocity
    #         defined as pres_vel

    dfdt_adv_dens = np.zeros(np.shape(xx))
    dfdt_adv_temp = np.zeros(np.shape(xx))
    dfdt_adv_P = np.zeros(np.shape(xx))
    [my, mx] = np.shape(xx)

    ## ################# Density advection ############### ##
   # Upwind in X-direction
    for i in range(1, mx - 1):
        for j in range(0, my):
            if vx[j, i] > 0:
                dfdtx_dens = vx[j, i] * (dens[j, i - 1] - dens[j, i]) / dx
                dfdtx_temp = vx[j, i] * (temp[j, i - 1] - temp[j, i]) / dx
                dfdtx_P = vx[j, i] * (P[j, i - 1] - P[j, i]) / dx
            else:
                dfdtx_dens = vx[j, i] * (dens[j, i] - dens[j, i + 1]) / dx
                dfdtx_temp = vx[j, i] * (temp[j, i] - temp[j, i + 1]) / dx
                dfdtx_P = vx[j, i] * (P[j, i] - P[j, i + 1]) / dx
            dfdt_adv_dens[j, i] = dfdtx_dens
            dfdt_adv_temp[j, i] = dfdtx_temp
            dfdt_adv_P[j,i] =   dfdtx_P
    # Upwind in Z-direction
    for i in range(0, mx):
        for j in range(1, my - 1):
            if vy[j, i] > 0:
                dfdty_dens = vy[j, i] * (dens[j - 1, i] - dens[j, i]) / dy
                dfdty_temp = vy[j, i] * (temp[j - 1, i] - temp[j, i]) / dy
                dfdty_P = vy[j, i] * (P[j - 1, i] - P[j, i]) / dy
            else:
                dfdty_dens = vy[j, i] * (dens[j, i] - dens[j + 1, i]) / dy
                dfdty_temp = vy[j, i] * (temp[j, i] - temp[j + 1, i]) / dy
                dfdty_P = vy[j, i] * (P[j, i] - P[j + 1, i]) / dy                
            dfdt_adv_dens[j, i] = dfdt_adv_dens[j, i] + dfdty_dens
            dfdt_adv_temp[j, i] = dfdt_adv_temp[j, i] + dfdty_temp
            dfdt_adv_P[j,i] = dfdt_adv_P[j,i] + dfdty_P

    # Add dt*df/dt-vector to old solution:
    rho_advected = dens + dt * dfdt_adv_dens
    temp_advected = temp + dt * dfdt_adv_temp
    P_advected = P + dt * dfdt_adv_P
    return [temp_advected, rho_advected, P_advected]
#============================================================================
# Func: Deformation box
#============================================================================
def deform_box():
    xs=1
    ys=1
    if(np.sum(vy) != 0.0):
        mock = np.zeros(np.shape(xx))
        mock[np.where(vx != 0.)] = 50.
        mock[np.where(vy < 0.)] = 50.
    
        # Find coordinates of the down-going slab
        max_x = np.max(xx[np.where(mock == 50.)])
        max_y = np.max(yy[np.where(mock == 50.)])
    
        # build coords of the deform box
        coord = [[0,0], [hinge_ax, hinge_ay], [max_x, max_y], [
            max_x - wid, max_y], [hinge_bx, hinge_by], [0, slab_thickness], [0, 0]]
        xs, ys = zip(*coord)
    else:  # in the case of advection is stopped at some polint
        xs, ys = xs, ys

    return [xs, ys]
#============================================================================
# Func: Get Pressure from Density
#============================================================================
def get_pressure(rho):
    ##
    # Purpose: This subroutine calculates the pressure distribution in the
    #          model box using P=density*gravity*depth. The pressure dist.
    #          is updated at every timestep, just as new destity dist. is
    #          recalculted at every timestep.
    ##
    gy = 9.81  # gravity in m/s²
    P_calc = np.ones(np.shape(xx))
    P_calc[0::, 0::] = ((yy[0::, 0::]) * gy * rho[0::, 0::]) / 1e5   # Pa-> bar

    return P_calc  # return initial density distribution in bar
#============================================================================
# Func: the effect of temp. and pressure on density 
#============================================================================
def density_P(P_new0, P_old0, rho_old0):#(new_temp, old_temp, old_rho):
    # Purpose: This subroutine calculates the density distribution, taking
    #         both temperature and pressure into consideration
    rho_TP = np.ones(np.shape(xx))
    [my, mx] = np.shape(xx)

    for i in range(mx):
        for j in range(my):
            #if (xx[j, i] >= 0 and xx[j, i] <= w
            #        and yy[j, i] < box_moho):
            #    rho_TP[j, i] = rho_old0[j, i]
            if (xx[j, i] >= 0 and xx[j, i] <= w
                    and yy[j, i] >= box_moho and yy[j, i] <= dbottom):
                rho_TP[j, i] = rho_old0[j, i]+ drhodP_lith * (P_new0[j, i] - P_old0[j, i])
            
            # Lithospheric mantle layer
            if (xx[j, i] >= 0 and xx[j, i] <= w
                    and yy[j, i] >= box_moho and yy[j, i] <= slab_thickness):
                rho_TP[j, i] = rho_old0[j, i]  + drhodP_lith * (P_new0[j, i] - P_old0[j, i])
            # Triangular block (joint)
            if (yy[j, i] >= box_moho and yy[j, i] <= slab_thickness
                    and xx[j, i] >= hinge_ax - (yy[j, i] - box_moho) / gradient1
                    and xx[j, i] <= hinge_ax + (yy[j, i] - box_moho) / gradient2):
                rho_TP[j, i] = rho_old0[j, i] + drhodP_lith * (P_new0[j, i] - P_old0[j, i])
            # Hanging lithospheric slab
            if (yy[j, i] >= slab_thickness and yy[j, i] <= dbottom
                    and xx[j, i] <= hinge_bx + wid + (yy[j, i] - slab_thickness) / gradient2
                    and xx[j, i] >= hinge_bx + (yy[j, i] - slab_thickness) / gradient2):
                rho_TP[j, i] = rho_old0[j, i] + drhodP_lith * (P_new0[j, i] - P_old0[j, i])
            
    return rho_TP
#============================================================================    
def density_T(T_new0, T_old0, rho_old0):#(new_temp, old_temp, old_rho):
    # Purpose: This subroutine calculates the density distribution, taking
    #         both temperature and pressure into consideration
    
    rho_TP = np.ones(np.shape(xx))
    [my, mx] = np.shape(xx)

    for i in range(mx):
        for j in range(my):
            #if (xx[j, i] >= 0 and xx[j, i] <= w
            #        and yy[j, i] < box_moho):
            #    rho_TP[j, i] = rho_old0[j, i]
            if (xx[j, i] >= 0 and xx[j, i] <= w
                    and yy[j, i] >= box_moho and yy[j, i] <= dbottom):
                rho_TP[j, i] = rho_old0[j, i] + drhodT_lith * \
                    (T_new0[j, i] - T_old0[j, i])   \
                # + drhodP_asth * (P_dummy[j, i] - P_old[j, i])
            
            # Lithospheric mantle layer
            if (xx[j, i] >= 0 and xx[j, i] <= w
                    and yy[j, i] >= box_moho and yy[j, i] <= slab_thickness):
                rho_TP[j, i] = rho_old0[j, i] + drhodT_lith * \
                    (T_new0[j, i] - T_old0[j, i])  \
                # + drhodP_lith * (P_dummy[j, i] - P_old[j, i])
            # Triangular block (joint)
            if (yy[j, i] >= box_moho and yy[j, i] <= slab_thickness
                    and xx[j, i] >= hinge_ax - (yy[j, i] - box_moho) / gradient1
                    and xx[j, i] <= hinge_ax + (yy[j, i] - box_moho) / gradient2):
                rho_TP[j, i] = rho_old0[j, i] + drhodT_lith * \
                    (T_new0[j, i] - T_old0[j, i])   \
                # + drhodP_lith * (P_dummy[j, i] - P_old[j, i])
            # Hanging lithospheric slab
            if (yy[j, i] >= slab_thickness and yy[j, i] <= dbottom
                    and xx[j, i] <= hinge_bx + wid + (yy[j, i] - slab_thickness) / gradient2
                    and xx[j, i] >= hinge_bx + (yy[j, i] - slab_thickness) / gradient2):
                rho_TP[j, i] = rho_old0[j, i] + drhodT_lith * \
                    (T_new0[j, i] - T_old0[j, i])   \
                # + drhodP_lith * (P_dummy[j, i] - P_old[j, i])
        
    return rho_TP
#============================================================================
# Func: 2D Heat Advection-Diffusion
#============================================================================
def diffuse(temp,dens):#(old_temp, old_rho):
    # Purpose: This subroutine solves diffusion & advection equations.
    #         The diffusion is applied internally (of the model box),
    #         with natural BCs imposed on the desired sides.
    #         For advection, the markers/nodes are moved with the
    #         upwind scheme in both x- and z-direction
    # Input:
    #          fin = the input matrix, in this case - the temperature
    # --->x     vx = the matrix containing velocity in x-direction
    # |         vy = the matrix containing velocity in z-direction
    # v y       dx = discretised grid spacing in x [m]
    #           dy = discretised grid spacing in y [m]
    #           kappa = thermal diffusivity [m2/s]
    #           dt = total time step
    ############### Temp. Diffusion ##################
    # Initialize a timestep df/dt vector:
    dfdt_diffus = np.zeros(np.shape(temp))
    kappa = np.zeros(np.shape(temp))
    # Setting up for the solver
    dx2 = dx**2
    dy2 = dy**2

    Txx = (temp[1:-1, 2::] - 2 * temp[1:-1, 1:-1] + temp[1:-1, 0:-2]) / dx2
    Tyy = (temp[2::, 1:-1] - 2 * temp[1:-1, 1:-1] + temp[0:-2, 1:-1]) / dy2

    kappa[np.int(box_moho / dy):np.int(slab_thickness / dy) + 1, :] = kappa_lith#k_lith / (dens[np.int(box_moho / dy):np.int(dlab / dy) + 1, :] * Cp)
    kappa[np.int(slab_thickness / dy) + 1::, :] = kappa_asth#k_asth / (dens[np.int(dlab / dy) + 1::, :] * Cp)

    # Apply diffusion:
    #   Internal diffusion:
    dfdt_diffus[1:-1, 1:-1] = kappa[1:-1, 1:-1] * (Txx[:, :] + Tyy[:, :])

    #   Natural b.c.'s at side boundaries:
    dfdt_diffus[1:-1, 0] = dfdt_diffus[1:-1, 0] + 2 * \
        kappa[1:-1, 0] * (temp[1:-1, 1] - temp[1:-1, 0]) / dx2
    dfdt_diffus[1:-1, -1] = dfdt_diffus[1:-1, -1] + 2 * \
        kappa[1:-1, 0] * (temp[1:-1, -2] - temp[1:-1, -1]) / dx2
    #  Natural b.c.'s at top and bottom boundaries:
    # dfdt_diffus[-1, 0::] = dfdt_diffus[-1, 0::] + 2 * \
    #     kappa[0, 0::] * (temp[-2, 0::] - temp[-1, 0::]) / dy2

    # Add df/dt-vector to old solution:
    temp_diffus = temp + (dt * dfdt_diffus)  # both diff & adv effects
    
    return temp_diffus
#============================================================================
## Func: Integrate
#============================================================================
# whole, triag, hang=whole-triang
def integrate(rho_init00, rho_new00, effect):
    g = 9.81  # m/s²
    # Whole down-going block
    if (effect == 1):
        slab_rho = np.sum(np.subtract(rho_new00[np.where(vy[:, :] > 0.)],rho_init00[np.where(vy[:, :] > 0.)]))
    if (effect == 2): # triang
        slab_rho= np.sum(np.subtract(rho_new00[np.where(vy[np.int(box_moho/dy):np.int(slab_thickness/dy)-2, :] > 0.)],rho_init00[np.where(vy[np.int(box_moho/dy):np.int(slab_thickness/dy)-2, :] > 0.)]))
    if (effect == 3): # hang
        slab_rho_whole = np.sum(np.subtract(rho_new00[np.where(vy[np.int(box_moho/dy)::, :] > 0.)], rho_init00[np.where(vy[np.int(box_moho / dy)::, :] > 0.)]))
        slab_rho_triang = np.sum(np.subtract(rho_new00[np.where(vy[np.int(box_moho/dy):np.int(slab_thickness/dy)-2, :] > 0.)],rho_init00[np.where(vy[np.int(box_moho/dy):np.int(slab_thickness/dy)-2, :] > 0.)]))
        slab_rho = np.subtract(slab_rho_whole,slab_rho_triang)
    weight_diff = (slab_rho)*(dx * dy)*g  
    return weight_diff
#============================================================================
## Func: Impose temp
#============================================================================
def imposeT(new_temp, init_temp):
    T_adjust0 = init_temp.copy()
    T_adjust0[np.int(box_moho / dy):np.int(slab_thickness / dy),:] = new_temp[np.int(box_moho / dy):np.int(slab_thickness / dy), :]
    T_adjust0[np.where(vy > 0.)] = new_temp[np.where(vy > 0.)]
    return T_adjust0
## -----------------END OF SUB-ROUTINES DEFINITIONS --------------------- ##

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
## Set up initial matrices ##
[vx, vy] = get_vel_init()  # initial vel dist.
yy_quiv = yy[:, :]
T_adv_init = T_init.copy()
T_diffus_init = T_init.copy()
rho_adv_init = rho_init.copy()
rho_diffus_init = rho_init.copy()
T_old =  T_init.copy()
T_adv_old =  T_init.copy()
T_diffus_old =  T_init.copy()
dfdt_adv_rho_old = np.zeros(np.shape(xx))
dfdt_adv_temp_old = np.zeros(np.shape(xx))
dfdt_diffus_temp_old = np.zeros(np.shape(xx))
rho_old = rho_init.copy()
rho_adv_old = rho_adv_init.copy()
rho_diffus_old = rho_diffus_init.copy()
P_init = get_pressure(rho_init)  # initial dist. (bar)
P_old = get_pressure(rho_init)  # preloop
P_adv_old = get_pressure(rho_init)
P_diffus_old = get_pressure(rho_init)

T_old2=T_init.copy()
rho_old2=rho_init.copy()
P_old2=P_init.copy()
## Determine Courant timestep criterion:##
vxmax = (abs(vx)).max()
vymax = (abs(vy)).max()
dt_diff = 1 * dx**2     # timestep in Myrs
dt_adv = min(dx / vxmax, dy / vymax)  # advection timestep
dt = 30000 * 0.5 * min(dt_diff, dt_adv)  # total timestep
#============================================================================
buoy_tot = [0]
buoy_sum = [0]
buoy_adv_a_differ = [0]
buoy_adv_b_differ = [0]
buoy_diffus_differ = [0]
buoy_adv = [0]
xbuoy = [0]
dfdt_adv_rho_old = np.zeros(np.shape(xx))
dfdt_adv_temp = np.zeros(np.shape(xx))
dfdt_diffus_temp_old = np.zeros(np.shape(xx))
df_old=pd.DataFrame(columns=['time','Ftot','Fsum','FadvA','FadvB','Fadv','Fdiffus',\
                             'shorten','angle','drho','bottom'])
slices_old=np.zeros(np.size(y_plot))

#============================================================================
## Timestepping ##
#============================================================================

for it in range(0, nt):

    # numerical solution
    shorten=pres_vel * t / 1e3
    print('No. timestep = %d ;  Time = %.2f My ;  Shortening = %.1f km' %
          (it, t / 365.25e6 / 86400, shorten))    
    # Move 'hangin vel. structure' down a pre-defined path with each timestep
    [vx, vy] = move_vel()
    [xs, ys] = deform_box()
    if ys[2] == h:
        hit_bottom=1
        sys.exit()
    else:
    	hit_bottom=0
     
    # TOTAL EFFECTS
    T_diffus_diffus = diffuse(T_old,rho_old)
    T_diffus_new=imposeT(T_diffus_diffus,T_init)
#    T_diffus_new= T_diffus_diffus.copy()
    rho_diffus_new_ = density_T(T_diffus_new,T_old,rho_old)
    rho_diffus_new = imposeT(rho_diffus_new_,rho_init)
#    rho_diffus_new=rho_diffus_new_.copy()
    P_new_=get_pressure(rho_diffus_new)
    
    [T_adv_advected,rho_adv_advected,P_adv_advected] = advect(T_diffus_new,rho_diffus_new,P_new_)
    T_adv_new =imposeT(T_adv_advected,T_init) # 
#    T_adv_new=T_adv_advected.copy()
    
    P_adv_old =imposeT(P_adv_advected,P_init)
#    P_adv_old=P_adv_advected.copy()
    P_adv_new = get_pressure(rho_adv_advected)
    rho_adv_new_ = density_P(P_adv_new,P_adv_old,rho_adv_advected)
    rho_adv_new =imposeT(rho_adv_new_,rho_init)
#    rho_adv_new=rho_adv_new_.copy()

    rho_new=rho_adv_new.copy()
    T_new =T_adv_new.copy()
    P_new=get_pressure(rho_new)


    # Difference in density distribution #
    rho_diff =  np.subtract(rho_new,rho_init) #rho_new - rho_init

    # Integrate 
    w_tot = -integrate(rho_old,rho_new,1)
    w_b_adv_differ = -integrate(rho_diffus_new,rho_adv_new_, 3)
#    w_b_adv_differ = -integrate(rho_old, rho_adv_new, 3) 
    w_a_adv_differ = -integrate(rho_old,rho_new, 2) 
    w_diffus_differ = -integrate(rho_old, rho_diffus_new_, 1) 
    w_b_adv = -integrate(rho_diffus_new,rho_adv_new_, 1)
    w_adv = -integrate(rho_diffus_new,rho_adv_new_, 1)

    mmm= rho_new-rho_old#(rho_new-rho_old)-(rho_adv_new_-rho_diffus_new)
    # Temperature profiles
    Tslice1 = copy(T_prof)
    Tslice2 = T_new[y_profile[0::],x_profile[0::]]
    rho_slice1 = copy(rho_prof)
    rho_slice2 = rho_new[y_profile[0::],x_profile[0::]]

    # update time
    t = t + dt                  # in sec
    tmyrs = float(t / secinmyr)   # time in Myrs
    
   
    # SAVE the T and rho slice profiles
    if (round(shorten,0) % 50 == 0):
        tmyr_array = np.zeros(np.size(y_plot))+tmyrs
        short_array = np.zeros(np.size(y_plot))+shorten
        slices_new = np.c_[tmyr_array,short_array,y_plot,Tslice2,rho_slice2]
        slices = np.c_[slices_old, slices_new]
        
#============================================================================
# Visualisations
#============================================================================

    if ys[2]>=slab_thickness+(1e-4*dy):
        # Total effect (summation) -- ignore this
        w_sum=-np.sum([-w_a_adv_differ,-w_b_adv_differ,-w_diffus_differ]) 
        buoy_sum.append(buoy_sum[-1]+w_sum) # Ignore this - only for saving to files
        buoy_tot.append(buoy_tot[-1]+w_tot) # TOTAL Effect
        buoy_adv_a_differ.append(buoy_adv_a_differ[-1]+w_a_adv_differ)
        buoy_adv_b_differ.append(buoy_adv_b_differ[-1]+w_b_adv_differ)
        buoy_diffus_differ.append(buoy_diffus_differ[-1]+w_diffus_differ)
        buoy_adv.append(buoy_adv[-1]+w_adv)
        xbuoy.append(tmyrs)

        
        # SAVE AGES AND BUOYANCY FORCES TO FILE ##########################
        y_plot1=np.array([y_plot])
        Tslice_21=np.array([Tslice2])
        rhoslice_21=np.array([rho_slice2])
        df_new=df_old.append(pd.Series([tmyrs,buoy_tot[-1]+w_tot, \
                                       buoy_sum[-1]+w_sum, \
                                       buoy_adv_a_differ[-1]+w_a_adv_differ, \
                                       buoy_adv_b_differ[-1]+w_b_adv_differ, \
                                       buoy_adv[-1]+w_a_adv_differ+w_b_adv_differ,\
                                       buoy_diffus_differ[-1]+w_diffus_differ, \
                                       shorten, angle,drho_lab,hit_bottom], \
                           index=['time','Ftot','Fsum','FadvA','FadvB','Fadv','Fdiffus', \
                           		  'shorten','angle','drho','bottom']), \
        						  ignore_index=True)
        df_old = df_new.copy()
#        if ((it) % 1) == 0:
#        print("Data saved in directory %s" % dir2)
        os.chdir(dir3)
        df_old.to_csv("%s_%s_vel_%d.csv" % (mantle_type, experiment_number,velocity))
#        else:
#            pass
#============================================================================
    # plot solution:
    if (it % 10 == 0):
        if plot_im_subplots == True:
            fig1 = plt.figure(1)
            fig1.set_size_inches(18.5, 10.5, forward=True)
            fig1.clf()
            # Set up GridSpec
            gs = gridspec.GridSpec(2, 4)
            ax1 = plt.subplot(gs[0:1, 2:])  # Temperature
#            ax2 = plt.subplot(gs[3:6, 6:])  # New density
            ax5 = plt.subplot(gs[1:, 2:])  # Density difference
            ax3 = plt.subplot(gs[1:, 0:1])  # Temp profile
            ax4 = plt.subplot(gs[1:, 1:2])  # Density profile
            ax6 = plt.subplot(gs[0:1, 0:2])  # Buoyancy
            xtick_label = [np.int(v) for v in np.arange(0, (w + dx) / 1000, 200)]
            ytick_label = [np.int(v) for v in np.arange(dmoho/1000, (h+50e3) / 1000,100)]
#============================================================================        
            # Subplot of temperature and velocity
            im1 = ax1.imshow(T_new[::-1],
                             extent=(0, w, 0, h),
                             #clim=(0, Tbottom),
                             interpolation='bilinear',
                             cmap=CBtemp_map)
            ax1.plot(xs, ys, 'k', alpha=0.5)
            ax1.set_ylim(0,600)
            ax1.invert_yaxis()
       
            ax1.set_xticklabels(xtick_label,fontsize=12)
            ax1.set_yticklabels(ytick_label,fontsize=12)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            cbar = plt.colorbar(im1, cax=cax,ticks=np.linspace(500,1500,6))
            cbar.ax.invert_yaxis()
            cbar.ax.set_title('$^{\circ}C$',fontsize=11)
            cbar.ax.tick_params(labelsize=12) 
            ax1.axis('equal')
            
            
            if (lab_contour == 'yes'):
                reverse_Tnew = T_new
                Tcon = ax1.contour(xx[beg_depth::, :], yy[beg_depth::, :],
                                   reverse_Tnew[beg_depth::, :],
                                   levels=(np.arange(500., Tlab + 600., 200.,dtype=int)), colors='k')
                ax1.clabel(Tcon,fmt = '%1.0f', fontsize=11)

            ax1.set_title(mantle_type+' T, t=' + str("%.2f" % tmyrs) + ' Myr, tstep =' +
                          str(it + 1) + ', v=' + str(velocity) + '$mm$ $yr^{-1}$'+
                          ', Shortening='+str("%.2f" % shorten)+'$km$',fontsize=15)
            ax1.set_xlabel('Distance in x-direction ($km$) $\longrightarrow$',fontsize=14)
            ax1.set_ylabel('Depth ($km$)',fontsize=14)
            ax1.set_xlim(0,w)
            ax1.set_ylim(h,0)
#============================================================================
            # Where is the profile coming from? -- position of the profiles
            ax1.plot(x_plot,y_plot,'k--', alpha=0.4)
            ax5.plot(x_plot,y_plot,'k--', alpha=0.4)
#============================================================================
            # Subplot  density difference
            im5 = ax5.imshow(rho_diff[::-1],
                             extent=(0, w, 0, h),
                             clim=(-20, 20),
                             interpolation='bilinear',
                             cmap=plt.cm.get_cmap('RdBu', 11))  # , norm=MidpointNormalize(midpoint=0.))
            ax5.invert_yaxis()
            ax5.set_xticklabels(xtick_label,fontsize=12)
            ax5.set_yticklabels(ytick_label,fontsize=12)
            divider = make_axes_locatable(ax5)
            cax5 = divider.append_axes("right", size="2%", pad=0.2)
            cbar = plt.colorbar(im5, cax=cax5, extend='both')
            
            # cbar.ax.invert_yaxis()
            ax5.plot(xs, ys, 'k', alpha=0.5)
            cbar.ax.set_title('$kgm^{-3}$',fontsize=12)
            cbar.ax.tick_params(labelsize=12) 
            ax5.set_title(mantle_type+ r" $\Delta \rho$"+', t=' + str("%.2f" % tmyrs) + ' Myr, tstep =' +
                          str(it + 1) + ', v=' + str(velocity) + '$mm$ $yr^{-1}$'+
                          r", $\Delta \rho_{LAB}$="+str("%.0f" % drho_lab),fontsize=15)
                          #r", $\Delta \rho_{LAB}$="+str("%.0f" % delrho[rr]),fontsize=15)
            ax5.set_xlabel('Distance in x-direction ($km$)',fontsize=14)
            ax5.set_ylabel('Depth ($km$)',fontsize=14)
            ax5.axis('equal')
            ax5.set_xlim(0,w)
            ax5.set_ylim(h,0)
#============================================================================
            if ys[2]>=slab_thickness+(1e-4*dy):
                ax6.plot(xbuoy[1::], buoy_tot[1::], 'k', linewidth='2', label='$F_b$')
                # ax6.plot(xbuoy[1::], buoy_adv_a_differ[1::], 'b', label='W_adv above LAB')
                # ax6.plot(xbuoy[1::], buoy_adv_b_differ[1::], 'g',linewidth='2', label='W_adv below LAB')
                ax6.plot(xbuoy[1::], buoy_adv[1::], 'g', label='$F_{advection}$')
                ax6.plot(xbuoy[1::], buoy_diffus_differ[1::], 'r', label='$F_{diffusion}$')
                #                ax6.plot(xbuoy[1::], buoy_sum[1::], 'grey', label='W_sum')
                
                ax6.grid(linestyle='dotted')
                ax6.legend(loc='best',fontsize=14)
                ax6.set_ylim(params.buoy_ylim)
                ax6.set_title('Buoyancy Plot',fontsize=15)
                ax6.set_xlabel('Time (Myr)',fontsize=14)
                ax6.set_ylabel('Buoyancy Force, $F_b$ ($Nm^{-1}$)',fontsize=14)
                ax6.tick_params(labelsize=12)
#============================================================================
            # Plot Temp. profiles
            ytick_prof = [np.int(v) for v in np.arange(dmoho/1000, h / 1000, 100)]
            y = np.linspace(0, h/1000, len(yy))
            im3 = ax3.plot(T_prof, y*1000, color='k', label=(
                'Init. $T$'))
            ax3.plot(Tslice2[beg_depth::],y_plot[beg_depth::],color='r', label=('Slab $T$'))
            #ax3.plot(Tslice2[beg_depth::], y[beg_depth::], 'r--',
            #         label=('x=' + str(slice_pos2 / 1000) + 'km'))

            ax3.plot((np.min(Tslice1), np.max(Tslice1)),
                     ((slab_thickness) , (slab_thickness) ), 'k--')

            ax3.invert_yaxis()
            ax3.legend(loc='best', fontsize=12)
            ax3.grid(linestyle='dotted')
            ax3.set_title('Temp-Depth profile', fontsize=14)
            ax3.set_xlabel('Temperature ($^{\circ}C$)', fontsize=14)
            ax3.set_ylabel('Depth ($km$)', fontsize=14)
            ax3.set_ylim(h,0)
            ax3.set_yticklabels(ytick_prof)
            ax3.xaxis.set_ticks(np.arange(min(T_prof), max(T_prof), 500))
            ax3.tick_params(labelsize=12)
#============================================================================
            # Plot density profiles
            im4 = ax4.plot(rho_prof, y*1000, color='k', label=(r'Init. $\rho$'))
            #ax4.plot(rho_slice2[beg_depth::], y[
            #    beg_depth::], 'r--', label=('x=' + str(slice_pos2 / 1000) + 'km'))
            ax4.plot(rho_slice2[beg_depth::],y_plot[beg_depth::],color='r', label=(r'Slab $\rho$'))
            ax4.plot((np.min(rho_slice1), np.max(rho_slice1)), 
                     ((slab_thickness) , (slab_thickness)), 'k--')
            ax4.invert_yaxis()
            ax4.legend(loc='best', fontsize=12)
            ax4.grid(linestyle='dotted')
            ax4.set_title('Density-Depth profile', fontsize=14)
            ax4.set_xlabel('Density ($kgm^{-3}$)', fontsize=14)
            ax4.set_ylabel('Depth ($km$)', fontsize=14)
            ax4.set_ylim(h,0)
            ax4.set_yticklabels(ytick_prof)
            ax4.xaxis.set_ticks(np.arange(np.round(min(rho_prof),-2), max(rho_prof), 200))
            ax4.tick_params(labelsize=12)
#============================================================================            
            plt.pause(0.0001)
            plt.draw()

            if matplotlib.get_backend().lower() in ['agg', 'macosx']:
                fig1.set_tight_layout(True)
            else:   
                fig1.tight_layout()
#============================================================================
            if save_figure == True:
                if ((it) % save_interval) == 0:
                    dir = os.path.expanduser(fig_path)
                    if not os.path.exists(dir1):
                         os.mkdir(dir1)
                    print("Images in directory %s" % dir1)
                    os.chdir(dir1)
                    plt.savefig(str(params.experiment_number) + str(mantle_type) +'_subplots' + str(it+1) + '.png', format='png', dpi=300)
#============================================================================
# END OF Visualisations
#============================================================================
                    
    ## SAVE DATA TO BINARY FILE                 
    os.chdir(dir2)                
    if (it % params.save_interval == 0):        
        ## Save T- rho- P-distribution to a binary file
        bin_file=open(str(experiment_number) + str(mantle_type) +str(it + 1) + '.bin',"wb")
        pickle.dump((T_new,rho_new,P_init,T_init,rho_init,P_init, \
                     buoy_tot,buoy_adv_a_differ,buoy_adv_b_differ, \
                     buoy_adv, buoy_diffus_differ, xbuoy, Tslice2, \
                     rho_slice2,tmyrs, xs, ys), bin_file)
        bin_file.close()

#============================================================================
    ## PASSING ON PARAMETERS FOR THE NEXT LOOP    
    P_old = P_new.copy()
    rho_old = rho_new.copy()
    T_old = T_new.copy()
    
    slices_old=slices.copy()
    os.chdir(dir2)
    aaa_name=(str(experiment_number)+'_'+str(mantle_type)+str(velocity)+'_slices.txt')
    np.savetxt(aaa_name, slices, fmt='%2f' )

sys.exit()
#============================================================================
#============================ END OF CODE ===================================
#============================================================================
