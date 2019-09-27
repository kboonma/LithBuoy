#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
mantle.py 
-- this file store parameters related to the difference mantle types 
-- to be called upon by other scripts.                      
#=======================================================================#
@author: Kittiphon Boonma, kboonma.b@gmail.com                 
Created 21/08/2019                                               
#=======================================================================#
"""

import params  #import input file

#____________0_________1_________2___________3____________4_________5_________6_______7______8_____
mantle = ['Archon', 'Proton', 'Tecton', 'Ocean30ma','Ocean120ma','slab80','slab160','oc60','oc110']
mantle_type = mantle[params.man_type] #[mman] #'Ocean30ma'  # Type of mantle (Tecton, Archon,Proton)

#########################################################
## Set up for different mantle type ##

if (mantle_type=='oc60'): #slab+thickness
    dlab = 70e3  # m  
    dmoho=10e3
    Tlab = 1300     # [ºC]
    Tmoho = 350     # [ºC]
    Tbottom = 1500  # [ºC]
    ref_depth_lith = 33e3  # [m]
    ref_rho_lith = 3252.6  # [kg/m^3]
    ref_P_lith = 10625 # 46541  # [bar]
    ref_T_lith = 926.77
    drhodT_lith = -0.1236
    drhodP_lith = 0.0046
    dT_lith = 22.12 / 1000
    ##Tecton asthenos. references##
    ref_depth_asth = 95e3 # [m]
    ref_rho_asth = RefRho[rr]#3296  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 30763 #51909 
    ref_T_asth = 1384.7
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044
    
if (mantle_type=='oc110'):
    dlab = 110e3  # m  
    dmoho=10e3
    Tlab = 1300     # [ºC]
    Tmoho = 350     # [ºC]
    Tbottom = 1500  # [ºC]
    ref_depth_lith = 33e3  # [m]
    ref_rho_lith = 3306.9  # [kg/m^3]
    ref_P_lith = 10802 # 46541  # [bar]
    ref_T_lith = 623.4472
    drhodT_lith = -0.1236
    drhodP_lith = 0.0046
    dT_lith = 9.47 / 1000
    ##Tecton asthenos. references##
    ref_depth_asth = 104.762e3 # [m]
    ref_rho_asth = RefRho[rr]#3313  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 34014 #51909 
    ref_T_asth = 1358.7
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044

if (mantle_type=='slab80'): #slab+thickness
    dlab = 120e3#200e3  # m  
    dmoho=40e3
    Tlab = 1300     # [ºC]
    Tmoho = 650     # [ºC]
    ref_depth_lith = 130e3  # [m]
    ref_rho_lith = 3300.  # [kg/m^3]
    ref_P_lith = 37040  # [bar]
    ref_T_lith = 1300
    drhodT_lith = -0.121836
    dT_lith = 0.0005

    ##
    ref_depth_asth = 214e3  # [m]
    ref_rho_asth = RefRho[rr]  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 51909 
    ref_T_asth = 1325.2
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044
    
if (mantle_type=='slab160'): #slab+thickness
    dlab = 200e3#200e3  # m  
    dmoho=40e3
    Tlab = 1300     # [ºC]
    Tmoho = 650     # [ºC]
    ref_depth_lith = 130e3  # [m]
    ref_rho_lith = 3300.  # [kg/m^3]
    ref_P_lith = 37040  # [bar]
    ref_T_lith = 1300
    drhodT_lith = -0.121836
    drhodP_lith = 0.00438
    dT_lith = 0.0005

    ##
    ref_depth_asth = 133.33e3  # [m]
    ref_rho_asth = RefRho[rr]  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 70235#51909 
    ref_T_asth = 1329.5#1325.2
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044

#### Archon (Arc_3 Griffin)
if (mantle_type=='Archon'):
    dlab = 200e3#200e3  # m  
    dmoho=40e3
    Tlab = 1300     # [ºC]
    Tmoho = 650     # [ºC]
    Tbottom = 1500  # [ºC]
#     temp_dummy1 = np.loadtxt('temp_Archon.dat')
#    dense_dummy1 = np.loadtxt('dens_Archon_lab_200km.dat')  # load Archon rho_dist
    ##Archon lithos. references##
    ref_depth_lith = 160.014e3  # [m]
    ref_rho_lith = 3348.1  # [kg/m^3]
    ref_P_lith = 65627  # [bar]
    ref_T_lith = 1300.4
    drhodT_lith = -0.13018
    drhodP_lith = 0.0042311
    slab_thickness = dlab-dmoho 
    dT_lith=(Tlab-Tmoho)/(slab_thickness) 
    
    ##Archon asthenos. references##
    ref_depth_asth = 164.763e3  # [m]
    ref_rho_asth = 3418.0  # [kg/m^3]
    ref_P_asth = 68588  # [bar]
    ref_T_asth = 1320.0
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044
    dT_asth = 0.0005

#### Tecton (Tc_1 Griffin)
if (mantle_type=='Tecton'):  
    dlab = 120e3  # m  
    dmoho=40e3
    Tlab = 1300     # [ºC]
    Tmoho = 650     # [ºC]
    Tbottom = 1500  # [ºC]
    # load Archon temp distribution
#     temp_dummy1 = np.loadtxt('temp_Tecton.dat')
#    dense_dummy1 = np.loadtxt('dens_Tecton_lab_120km.dat')  # load Archon rho_dist
    ##Tecton lithos. references##
    ref_depth_lith = 80.78e3  # [m]
    ref_rho_lith = 3319.0  # [kg/m^3]
    ref_P_lith = 39286  # [bar]
    ref_T_lith = 1274.5
    drhodT_lith = -0.12546
    drhodP_lith = 0.0044315
    slab_thickness = dlab-dmoho 
    dT_lith=(Tlab-Tmoho)/(slab_thickness)     
        ##Tecton asthenos. references##
    ref_depth_asth = 93.33e3  # [m]
    ref_rho_asth = 3342  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 43707 
    ref_T_asth = 1335.6
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044

if (mantle_type=='Proton'):
    dlab = 150e3  # m  
    dmoho=40e3   
    Tlab = 1300     # [ºC]
    Tmoho = 650     # [ºC]
    Tbottom = 1500  # [ºC]
#     temp_dummy1 = np.loadtxt('temp_Proton.dat')
#    dense_dummy1 = np.loadtxt('dens_Proton_lab_150km.dat')  # load Archon rho_dist
    ##Tecton lithos. references##
    ref_depth_lith = 100e3  # [m]
    ref_rho_lith = 3324.4  # [kg/m^3]
    ref_P_lith = 46541  # [bar]
    ref_T_lith = 1253.1
    drhodT_lith = -0.12848
    drhodP_lith = 0.0043
    slab_thickness = dlab-dmoho 
    dT_lith=(Tlab-Tmoho)/(slab_thickness)     
        ##Tecton asthenos. references##
    ref_depth_asth = 133.33e3  # [m]
    ref_rho_asth = 3370.7  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 51909 
    ref_T_asth = 1325.2
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044


if (mantle_type=='Ocean30ma'):
    dlab = 70e3  # m  
    dmoho=10e3
    Tlab = 1300     # [ºC]
    Tmoho = 350     # [ºC]
    Tbottom = 1500  # [ºC]
    ref_depth_lith = 33e3  # [m]
    ref_rho_lith = 3252.6  # [kg/m^3]
    ref_P_lith = 10625 # 46541  # [bar]
    ref_T_lith = 926.77
    drhodT_lith = -0.1236
    drhodP_lith = 0.0046
    dT_lith = 22.12 / 1000
    ##Tecton asthenos. references##
    ref_depth_asth = 95e3 # [m]
    ref_rho_asth = 3296  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 30763 #51909 
    ref_T_asth = 1384.7
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044

if (mantle_type=='Ocean120ma'):
    dlab = 110e3  # m  
    dmoho=10e3
    Tlab = 1300     # [ºC]
    Tmoho = 350     # [ºC]
    Tbottom = 1500  # [ºC]
    ref_depth_lith = 33e3  # [m]
    ref_rho_lith = 3306.9  # [kg/m^3]
    ref_P_lith = 10802 # 46541  # [bar]
    ref_T_lith = 623.4472
    drhodT_lith = -0.1236
    drhodP_lith = 0.0046
    dT_lith = 9.47 / 1000
    ##Tecton asthenos. references##
    ref_depth_asth = 104.762e3 # [m]
    ref_rho_asth = 3313  # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 34014 #51909 
    ref_T_asth = 1358.7
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044

if (mantle_type=='oc60'):
    RefRho = [3269.,3271.,3273.,3275.,3277.,3279., \
              3282.,3284.,3286.,3288.,3290., \
              3292.,3294.,3295.,3297.,3299., \
              3301.,3303.,3305.,3307.,3309.]
              
if (mantle_type=='oc110'):
    RefRho = [3306.,3308.,3309.,3311.,3313.,3315., \
              3318.,3320.,3322.,3324.,3326., \
              3328.,3330.,3331.,3333.,3335., \
              3337.,3339.,3341.,3343.,3345.]         

if (mantle_type=='slab80'):
    RefRho = [3354.,3356.,3358.,3360.,3362.,3364., \
              3366.,3368.,3370.,3372.,3374., \
              3376.,3378.,3379.,3381.,3383., \
              3385.,3387.,3389.,3391.,3393., \
              3395.,3397.,3399.,3401.,3403., \
              3405.,3406.,3408.,3410.,3412., \
              3414.,3416.,3418.,3420.,3422., \
              3424.,3426.,3428.,3430.,3432.] # 72 74 76 78 80
#              xxxx.,xxxx.,xxxx.,xxxx.,3441., \ # 12 14 16 18 20
#              xxxx.,xxxx.,xxxx.,xxxx.,3451.] # 12 14 16 18 20
if (mantle_type=='slab160'):
    RefRho = [3427.,3429.,3431.,3433.,3434.,3436., \
              3438.,3439.,3441.,3443.,3446., \
              3449.,3451.,3453.,3454.,3455., \
              3457.,3458.,3460.,3462.,3464., \
              3466.,3467.,3469.,3472.,3474., \
              3476.,3478.,3480.,3481.,3483., \
              3485.,3487.,3489.,3490.,3492., \
              3494.,3496.,3498.,3500.,3502.]