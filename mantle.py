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
mantle_type = mantle[params.mman]#mantle[params.man_type] #[mman] #'Ocean30ma'  # Type of mantle (Tecton, Archon,Proton)

#########################################################
## Set up for different mantle type ##


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
    RefRho = [3358.,3359.,3361.,3363.,3365.,3367., \
              3369.,3371.,3373.,3375.,3377, \
              3379.,3381.,3383.,3385.,3386., \
              3388.,3390.,3392.,3394.,3396., \
              3398.,3400.,3402.,3404.,3406., \
              3408.,3410.,3412.,3413.5,3415., \
              3417.,3419.,3421.,3423.,3425., \
              3427.,3429.,3431.,3433.,3435.] # 72 74 76 78 80
#              xxxx.,xxxx.,xxxx.,xxxx.,3441., \ # 12 14 16 18 20
#              xxxx.,xxxx.,xxxx.,xxxx.,3451.] # 12 14 16 18 20
if (mantle_type=='slab160'):
    RefRho = [3429.,3432.,3434.,3435.,3437.,3439., \
              3441.,3443.,3445.,3447.,3448.5, \
              3450.,3452.,3454.,3456.,3458., \
              3460.,3462.,3463.,3465.,3467., \
              3469.,3471.,3473.,3475.,3476.5, \
              3478.,3480.,3482.,3484.,3486., \
              3488.,3490.,3491.,3493.,3495., \
              3497.,3499.,3501.,3503.,3504.5]


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
    ref_rho_asth = RefRho[params.params.rr]#3296  # [kg/m^3]
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
    ref_rho_asth = RefRho[params.rr]#3313  # [kg/m^3]
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
    Tbottom = 1500  # [ºC]
    ref_depth_lith = 130e3  # [m]
    ref_rho_lith = 3300.  # [kg/m^3]
    ref_P_lith = 37040  # [bar]
    ref_T_lith = 1300
    drhodT_lith = -0.121836
    drhodP_lith = 0.00438
    dT_lith = 0.0005

    ##
    ref_depth_asth = 214e3  # [m]
    ref_rho_asth = RefRho[params.rr]  # [kg/m^3]
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
    Tbottom = 1500  # [ºC]
    ref_depth_lith = 130e3  # [m]
    ref_rho_lith = 3300.  # [kg/m^3]
    ref_P_lith = 37040  # [bar]
    ref_T_lith = 1300
    drhodT_lith = -0.121836
    drhodP_lith = 0.00438
    dT_lith = 0.0005

    ##
    ref_depth_asth = 133.33e3  # [m]
    ref_rho_asth = RefRho[params.rr]  # [kg/m^3]
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
    ref_rho_asth = 3313  #RefRho[params.rr] # [kg/m^3]
    dT_asth = 0.0005
    ref_P_asth = 34014 #51909 
    ref_T_asth = 1358.7
    drhodT_asth = -0.1165
    drhodP_asth = 0.0044

