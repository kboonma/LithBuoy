#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_figs.py -- this script plot the 5 panels plot showing:
1 - the buoyancy force evolution plot
2 - Temperature profile and its evolution
3 - Density profile and its evolution
4 - Temperature box
5 - The density difference                  
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
import params
#from Stokes2D import Stokes2Dfunc 
from matplotlib.colors import LinearSegmentedColormap
import lithcall as lc
import os


#from LithBuoy import * 
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams['mathtext.fontset'] = u'cm'
#Colormaps
temp_data = np.loadtxt("colormap/roma.txt")
CBtemp_map = LinearSegmentedColormap.from_list('CBtemp', temp_data[::-1])

dens_data = np.loadtxt("colormap/lajolla.txt")
CBdens_map = LinearSegmentedColormap.from_list('CBdens', dens_data[::-1])

rcParams['figure.figsize'] = (15, 8)
rcParams['font.size'] = 8



#____________0_________1_________2___________3____________4_________5_________6_______7______8_____
mantle = ['Archon', 'Proton', 'Tecton', 'Ocean30ma','Ocean120ma','slab80','slab160','oc60','oc110']
mantle_type = mantle[1] #[mman] #'Ocean30ma'  # Type of mantle (Tecton, Archon,Proton)
experiment_number = 'TESTkappa_e6' # for the .savefig title -- CHANGE to prevent overwriting
velocity = 20  # mm/year
kappa_lith = 1e-6
kappa_asth = 1e-6

#fnum=[1,11,21,31,41,51,61,71,81,91,101] #[2001]

fnum=[2111]

plot_panels=0
plot_kappa=10




fig_path= os.getcwd()+'/data/%s_%s_vel%d/subplots/' % (experiment_number,mantle_type,velocity)
dat_path= os.getcwd()+'/data/%s_%s_vel%d/' % (experiment_number,mantle_type,velocity)
dir = os.path.expanduser(dat_path)
if not os.path.exists(dir):
    os.makedirs(dir)
os.chdir(dir)



for i in fnum:

    fin=open(str(experiment_number) + str(mantle_type) + str(i) + '.bin',"rb")
    data=pickle.load(fin)
    T_new=data[0]
    rho_new=data[1]
    P_new=data[2]
    T_init=data[3]
    rho_init=data[4]
    P_init=data[5]
    buoy_tot=data[6]
    buoy_adv_a_differ=data[7]
    buoy_adv_b_differ=data[8]
    buoy_adv=data[9]
    buoy_diffus_differ=data[10]
    xbuoy=data[11]
    Tslice2=data[12]
    rho_slice2=data[13]
    tmyrs=data[14]
    xs=data[15]
    ys=data[16]
    T_diffus_diffus=data[17]
    T_diffus_new=data[18]
    
    shorten= (velocity * 1e-3 / (365 * 24 * 3600)) * (tmyrs*lc.secinmyr) / 1e3
    
    
    # Mesh setup:
    nx = (params.w / params.dx) + 1
    ny = (params.h / params.dy) + 1
    x = np.linspace(0, np.int(params.w), np.int(nx))  # array for the finite difference mesh
    y = np.linspace(0, np.int(params.h), np.int(ny))
    [xx, yy] = np.meshgrid(x, y)
    xh = xx + 0.5 * params.h          # x-distances to horz middle of mesh (on xx mesh)
    yh = yy + 0.5 * params.h
    
    drho_lab = int(rho_init[int((lc.slab_thickness+params.dy)/params.dy),1] - rho_init[(int((lc.slab_thickness)/params.dy)),1])
    xtick_label = [np.int(v) for v in np.arange(0, (params.w + params.dx) / 1000, 200)]
    ytick_label = [np.int(v) for v in np.arange(lc.dmoho/1000, (params.h+50e3) / 1000,100)]     
    
    
    
    if(plot_kappa>0.):
        
        fig2=plt.figure(2)
       # fig2.set_size_inches(14, 7, forward=True)
        fig2.clf()
        
        T_diffus_diff=T_diffus_new-T_init
        T_diffus_diff2=T_diffus_new-T_diffus_diffus
        
        ax1=plt.subplot(221)
        im1 = ax1.imshow(T_diffus_new[::-1],
                         extent=(0, params.w, 0, params.h),
                         #clim=(500, 1500),
                         interpolation='bilinear',
                         cmap=CBtemp_map)
        ax1.plot(xs, ys, 'k', alpha=0.5)
        ax1.set_ylim(0,600)
        ax1.invert_yaxis()
        ax1.set_xticklabels(xtick_label,fontsize=12)
        ax1.set_yticklabels(ytick_label,fontsize=12)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="2%", pad=0.2)
        cbar = plt.colorbar(im1, cax=cax1,ticks=np.linspace(500,1500,6))
        cbar.ax.invert_yaxis()
        cbar.ax.set_title('$^{\circ}C$',fontsize=11)
        cbar.ax.tick_params(labelsize=12) 
        ax1.axis('equal')
        ax1.set_title(mantle_type+', t=' + str("%.2f" % tmyrs) + ' Myr, v=' + str(velocity) + '$mm$ $yr^{-1}$'+
                  ', Shortening='+str("%.2f" % shorten)+'$km$\nT_diffus_new, kappa_asth='+str("%.0e" % kappa_asth)+' $m^{2}s^{-1}$',fontsize=10)
        
        ax2=plt.subplot(222)
        im2 = ax2.imshow(T_diffus_diffus[::-1],
                         extent=(0, params.w, 0, params.h),
                         #clim=(0, Tbottom),
                         interpolation='bilinear',
                         cmap=CBtemp_map)
        ax2.plot(xs, ys, 'k', alpha=0.5)
        ax2.set_ylim(0,600)
        ax2.invert_yaxis()
        ax2.set_xticklabels(xtick_label,fontsize=12)
        ax2.set_yticklabels(ytick_label,fontsize=12)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="2%", pad=0.2)
        cbar = plt.colorbar(im2, cax=cax2,ticks=np.linspace(500,1500,6))
        cbar.ax.invert_yaxis()
        cbar.ax.set_title('$^{\circ}C$',fontsize=11)
        cbar.ax.tick_params(labelsize=12) 
        ax2.axis('equal')
        ax2.set_title('T_diffus_diffus', fontsize=10)
        
        ax3=plt.subplot(223)
        im3 = ax3.imshow(T_diffus_diff[::-1],
                         extent=(0, params.w, 0, params.h),
                         clim=(-500,500 ),
                         interpolation='bilinear',
                         cmap=plt.cm.get_cmap('RdBu', 11))
        ax3.plot(xs, ys, 'k', alpha=0.5)
        ax3.set_ylim(0,600)
        ax3.invert_yaxis()
        ax3.set_xticklabels(xtick_label,fontsize=12)
        ax3.set_yticklabels(ytick_label,fontsize=12)
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="2%", pad=0.2)
        cbar = plt.colorbar(im3, cax=cax3,extend='both',ticks=np.linspace(-500,500,5))
#        cbar.ax.invert_yaxis()
        cbar.ax.set_title('$^{\circ}C$',fontsize=11)
        cbar.ax.tick_params(labelsize=12) 
        ax3.axis('equal')
        ax3.set_title('T_diffus_new-T_init', fontsize=10)
        
        ax4=plt.subplot(224)
        im4 = ax4.imshow(T_diffus_diff2[::-1],
                         extent=(0, params.w, 0, params.h),
                         clim=(-2,2 ),
                         interpolation='bilinear',
                         cmap=plt.cm.get_cmap('RdBu', 11))
    #    ax4.plot(xs, ys, 'k', alpha=0.5)
        ax4.set_ylim(0,600)
        ax4.invert_yaxis()
        ax4.set_xticklabels(xtick_label,fontsize=12)
        ax4.set_yticklabels(ytick_label,fontsize=12)
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="2%", pad=0.2)
        cbar = plt.colorbar(im4, cax=cax4,extend='both',ticks=np.linspace(-2,2,5))
#        cbar.ax.invert_yaxis()
        cbar.ax.set_title('$^{\circ}C$',fontsize=11)
        cbar.ax.tick_params(labelsize=12) 
        ax4.axis('equal')
        ax4.set_title('T_diffus_new-T_diffus_diffus // Amount of diffusion', fontsize=10)
        
        
        #fig1 = plt.figure(1)
        ##fig1.set_size_inches(18.5, 10.5, forward=True)
        #fig1.clf()
        #ax1=plt.subplot(111)
        ## Set up GridSpec
        ##gs = gridspec.GridSpec(2, 4)
        ##ax1 = plt.subplot(gs[0:1, 2:])  # Temperature
        ###            ax2 = plt.subplot(gs[3:6, 6:])  # New density
        ##ax5 = plt.subplot(gs[1:, 2:])  # Density difference
        ##ax3 = plt.subplot(gs[1:, 0:1])  # Temp profile
        ##ax4 = plt.subplot(gs[1:, 1:2])  # Density profile
        ##ax6 = plt.subplot(gs[0:1, 0:2])  # Buoyancy
        #xtick_label = [np.int(v) for v in np.arange(0, (params.w + params.dx) / 1000, 200)]
        #ytick_label = [np.int(v) for v in np.arange(40e3/1000, (params.h+50e3) / 1000,100)] 
        ## Subplot of temperature and velocity
        #im1 = ax1.imshow(rho_new[::-1],
        #                 extent=(0, params.w, 0, params.h),
        #                 #clim=(0, Tbottom),
        #                 interpolation='bilinear',
        #                 cmap=CBtemp_map)
        ##    ax1.plot(xs, ys, 'k', alpha=0.5)
        #ax1.set_ylim(0,600)
        #ax1.invert_yaxis()
        #   
        #ax1.set_xticklabels(xtick_label,fontsize=12)
        #ax1.set_yticklabels(ytick_label,fontsize=12)
        #divider = make_axes_locatable(ax1)
        #cax = divider.append_axes("right", size="2%", pad=0.2)
        #cbar = plt.colorbar(im1, cax=cax,ticks=np.linspace(500,1500,6))
        #cbar.ax.invert_yaxis()
        #cbar.ax.set_title('$^{\circ}C$',fontsize=11)
        #cbar.ax.tick_params(labelsize=12) 
        #ax1.axis('equal')
    
    
    if(plot_panels>0.):    
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
    
        # Subplot of temperature and velocity
        im1 = ax1.imshow(T_new[::-1],
                         extent=(0, params.w, 0, params.h),
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
        
        
        if (lc.lab_contour == 'yes'):
            reverse_Tnew = T_new
            Tcon = ax1.contour(xx[0::, :], yy[0::, :],
                               reverse_Tnew[0::, :],
                               levels=(np.arange(500., lc.Tlab + 600., 200.,dtype=int)), colors='k')
            ax1.clabel(Tcon,fmt = '%1.0f', fontsize=11)
        
        ax1.set_title(mantle_type+' T, t=' + str("%.2f" % tmyrs) + ' Myr, tstep =' +
                      str(i) + ', v=' + str(velocity) + '$mm$ $yr^{-1}$'+
                      ', Shortening='+str("%.2f" % shorten)+'$km$',fontsize=15)
        ax1.set_xlabel('Distance in x-direction ($km$)',fontsize=14)
        ax1.set_ylabel('Depth ($km$)',fontsize=14)
        ax1.set_xlim(0,params.w)
        ax1.set_ylim(params.h,0)
        
        #================================================================
        # Where is the profile coming from? -- position of the profiles
        ax1.plot(lc.x_plot,lc.y_plot,'k--', alpha=0.4)
        ax5.plot(lc.x_plot,lc.y_plot,'k--', alpha=0.4)
        #================================================================
        
        # Difference in density distribution #
        rho_diff =  np.subtract(rho_new,rho_init) #rho_new - rho_init
        
        #================================================================
        # Subplot  density difference
        im5 = ax5.imshow(rho_diff[::-1],
                         extent=(0, params.w, 0, params.h),
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
                      str(i) + ', v=' + str(velocity) + '$mm$ $yr^{-1}$'+
                      r", $\Delta \rho_{LAB}$="+str("%.0f" % drho_lab),fontsize=15)
                      #r", $\Delta \rho_{LAB}$="+str("%.0f" % delrho[rr]),fontsize=15)
        ax5.set_xlabel('Distance in x-direction ($km$)',fontsize=14)
        ax5.set_ylabel('Depth ($km$)',fontsize=14)
        ax5.axis('equal')
        ax5.set_xlim(0,params.w)
        ax5.set_ylim(params.h,0)
        #================================================================
        #if ys[2]>=lc.slab_thickness+(1e-4*dy):
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
           
        #======================================================================
        # Plot Temp. profiles
        ytick_prof = [np.int(v) for v in np.arange(lc.dmoho/1000, params.h / 1000, 100)]
        y = np.linspace(0, params.h/1000, len(yy))
        im3 = ax3.plot(lc.T_prof, y*1000, color='k', label=(
            'Init. $T$'))
        ax3.plot(Tslice2[0::],lc.y_plot[0::],color='r', label=('Slab $T$'))
        #ax3.plot(Tslice2[beg_depth::], y[beg_depth::], 'r--',
        #         label=('x=' + str(slice_pos2 / 1000) + 'km'))
        
        ax3.plot((np.min(T_init[:,1]), np.max(T_init[:,1])),
                 ((lc.slab_thickness) , (lc.slab_thickness) ), 'k--')
        
        ax3.invert_yaxis()
        ax3.legend(loc='best', fontsize=12)
        ax3.grid(linestyle='dotted')
        ax3.set_title('Temp-Depth profile', fontsize=14)
        ax3.set_xlabel('Temperature ($^{\circ}C$)', fontsize=14)
        ax3.set_ylabel('Depth ($km$)', fontsize=14)
        ax3.set_ylim(params.h,0)
        ax3.set_yticklabels(ytick_prof)
        ax3.xaxis.set_ticks(np.arange(min(T_init[:,1]), max(T_init[:,1]), 500))
        ax3.tick_params(labelsize=12)
        
        # Plot density profiles
        im4 = ax4.plot(rho_init[:,1], y*1000, color='k', label=(r'Init. $\rho$'))
        #ax4.plot(rho_slice2[beg_depth::], y[
        #    beg_depth::], 'r--', label=('x=' + str(slice_pos2 / 1000) + 'km'))
        ax4.plot(rho_slice2[0::],lc.y_plot[0::],color='r', label=(r'Slab $\rho$'))
        ax4.plot((np.min(rho_init[:,1]), np.max(rho_init[:,1])), 
                 ((lc.slab_thickness) , (lc.slab_thickness)), 'k--')
        ax4.invert_yaxis()
        ax4.legend(loc='best', fontsize=12)
        ax4.grid(linestyle='dotted')
        ax4.set_title('Density-Depth profile', fontsize=14)
        ax4.set_xlabel('Density ($kgm^{-3}$)', fontsize=14)
        ax4.set_ylabel('Depth ($km$)', fontsize=14)
        ax4.set_ylim(params.h,0)
        ax4.set_yticklabels(ytick_prof)
        ax4.xaxis.set_ticks(np.arange(np.round(min(rho_init[:,1]),-2), max(rho_init[:,1]), 200))
        ax4.tick_params(labelsize=12)
        
        if matplotlib.get_backend().lower() in ['agg', 'macosx']:
            fig1.set_tight_layout(True)
        else:   
            fig1.tight_layout()
        
        dir1 = os.path.expanduser(fig_path)
        if not os.path.exists(dir1):
             os.mkdir(dir1)
        print("Images in directory %s" % dir1)
        os.chdir(dir1)
        plt.savefig(str(experiment_number) + str(mantle_type) +'_subplots' + str(i) + '.png', format='png', dpi=300)
    
    
    os.chdir('..')

