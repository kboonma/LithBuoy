import pandas as pd
import numpy as np
import matplotlib, os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
from pylab import rcParams
from numpy import loadtxt
import seaborn as sns
from scipy import optimize
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = u'cm'
rcParams['axes.linewidth'] = 1.5 #set the value globally
#plt.rcParams['axes.grid'] = True
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as ticker
#from brokenaxes import brokenaxes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
import math
import errno
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import numpy.polynomial.polynomial as pol
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as PathEffects
from scipy.interpolate import make_interp_spline, BSpline
import params

#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
#from scipy.interpolate import Bspline
#from ggplot import *
#import easygui as eg

#sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})
#rcParams['figure.figsize'] = (12, 7)
#rcParams['font.size'] = 14
# rcParams['axes', linewidth=3]


vel=[40]
#____________0_________1_________2___________3____________4_________5_________6_______7______8_____
mantle = ['Archon', 'Proton', 'Tecton', 'Ocean30ma','Ocean120ma','slab80','slab160','oc60','oc110']
mantle_type = mantle[1] #[mman] #'Ocean30ma'  # Type of mantle (Tecton, Archon,Proton)
exp_name = 'TESTdiffus_kappa_e5' # for the .savefig title -- CHANGE to prevent overwriting

kappa_lith = 1e-6
kappa_asth = 1e-5
#vel=[1,4,10,20,40,80]

rho_contrast    =0
rho_contrast2   =0
rho_contrast3   =0
rho_contrast_rho=0
buoyancy_plot   =0
Trho_evol       =0
FbouyShorten    =0
FbouyTime       =0
FbouyVels       =0
Archon_FbouyTime=0
Proton_FbouyTime=0
Tecton_FbouyTime=0
Tecton_FbouyShorten=0
Effect_advA     =0
Effect_advB     =0
Effect_diffus  =0
FtotShorten     =10
minFbuoy        =0
InitTrho        =0
ComponentsTime  =0
FdiffusShortenVel =0
FadvAShortenVel =0
FadvBShortenVel =0
#============================================================================

#class MyApp(tk.Tk):
#   def close(self):
#       self.destroy()
#
#   def okbutton(self):
#       global FbouyShorten
#       global FbouyTime      
#       global FbouyVels       
#       global Archon_FbouyTime
#       global Proton_FbouyTime
#       global Tecton_FbouyTime
#       global Effect_advA     
#       global Effect_advB     
#       global Effect_diffus  
#       global FtotShorten     
#       global InitTrho        
#       global ComponentsTime 
#       global FdiffusShortenVel
#       global FadvAShortenVel
#       global FadvBShortenVel
#       # for i , j in zip(cbuts_text,range(len(cbuts_text))):
#       #   i = "var%d"%(j+1)
#       FbouyShorten    =var1.get()
#       FbouyTime       =var2.get()
#       FbouyVels       =var3.get()
#       Archon_FbouyTime=var4.get()
#       Proton_FbouyTime=var5.get()
#       Tecton_FbouyTime=var6.get()
#       Effect_advA     =var7.get()
#       Effect_advB     =var8.get()
#       Effect_diffus   =var9.get()
#       FtotShorten     =var10.get()
#       InitTrho        =var11.get()
#       ComponentsTime  =var12.get()
#       FdiffusShortenVel=var13.get()
#       FadvAShortenVel =var14.get()
#       FadvBShortenVel =var15.get()
#       self.destroy()
#
#   def mainloop(self):
#       tk.Tk.mainloop(self)
#     
#   def __init__(self):
#       tk.Tk.__init__(self)
#       self.title("Plots")
#       tk.Label(self, text="Experiment Name:").grid(row=0,stick=tk.W)
#       tk.Label(self, text="Velocities:").grid(row=1,stick=tk.W)
#       global exp_name
#       global vel
#       e1=tk.StringVar()
#       e1.set("geomod")
#       e1= tk.Entry(self,textvariable=e1)
#       e1.grid(row=0, column=1)
#       exp_name=e1.get()
#       e2=tk.StringVar()
#       e2.set("1,4,10,20,30,40")  #("1,4,10,20,30,40")
#       e2 = tk.Entry(self,textvariable=e2)
#       e2.grid(row=1, column=1)
#       vel=(e2.get()).split(",")
#       
#
#
#   def create_cbuts(self):
#     for i , j in zip(cbuts_text,range(len(cbuts_text))):
#       globals()['C{}'.format(j+1)]=tk.Checkbutton(self, text = i, variable=globals()['var{}'.format(j+1)],\
#                                                 onvalue = 1, offvalue = 0)
#       globals()['C{}'.format(j+1)].grid(stick=tk.W)
#       cbuts.append(globals()['C{}'.format(j+1)])
#
#def select_all():
# for j in cbuts:
#     j.select()
##===================================================================
#app = MyApp()
#cbuts_text = ['FbouyShorten','FbouyTime', 'FbouyVels', \
#             'Archon_FbouyTime','Proton_FbouyTime','Tecton_FbouyTime', \
#             'Effect_advA', 'Effect_advB','Effect_diffus', \
#             'FtotShorten','InitTrho','ComponentsTime',\
#             'FdiffusShortenVel','FadvAShortenVel','FadvBShortenVel']
#cbuts = []

#
#for j in range(len(cbuts_text)):
# globals()['var{}'.format(j+1)] = tk.IntVar()
#
#app.create_cbuts()
#
#close_button = tk.Button(app, text="Close", command=app.close)
#close_button.grid(row=20,column=0)
#ok_button = tk.Button(app, text="OK",command=app.okbutton)
#ok_button.grid(row=20,column=2)
#selectall = tk.Button(app, text='All', command=select_all)
#selectall.grid(row=20,column=1)
#
#app.mainloop()

# #============================================================================

#vel=[4,20,80]

#vel=[1,5,10,15,20,25,30,40]
#vel=[1,5,15,30]

#vel2=[1,4,10]
# vel=[5,15,30]
# exp_name='NEW_TEST'
#
fig_path = os.getcwd()+'/plots/'+exp_name
csv_path= os.getcwd()+'/csv/' #%s_%s_vel%d/' % (params.experiment_number,mantle_type,params.velocity)
dir = os.path.expanduser(fig_path)
if not os.path.exists(dir):
     os.mkdir(dir)    

os.chdir(csv_path)


for i in range(len(vel)):
    try:
        globals()['dfa{}'.format(i+1)]=pd.read_csv(csv_path+'Archon_'+exp_name+'_vel_'+str(vel[i])+'.csv')
    except (IOError, EOFError) as e:
         pass
    try:
        globals()['dfp{}'.format(i+1)]=pd.read_csv(csv_path+'Proton_'+exp_name+'_vel_'+str(vel[i])+'.csv')
    except (IOError, EOFError) as e:
        pass
    try:
       globals()['dft{}'.format(i+1)]=pd.read_csv(csv_path+'Tecton_'+exp_name+'_vel_'+str(vel[i])+'.csv')
    except (IOError, EOFError) as e:
        pass
    try:
        globals()['dfoc1{}'.format(i+1)]=pd.read_csv(csv_path+'Ocean30ma_'+exp_name+'_vel_'+str(vel[i])+'.csv')
    except (IOError, EOFError) as e:
        pass
    try:
        globals()['dfoc2{}'.format(i+1)]=pd.read_csv(csv_path+'Ocean120ma_'+exp_name+'_vel_'+str(vel[i])+'.csv')
    except (IOError, EOFError) as e:
        pass
    
    


for i in range(len(vel)):
    try:
        globals()['df_a{}'.format(i+1)]=(globals()['dfa{}'.format(i+1)]).values
    except (IOError, EOFError,KeyError) as e:
         pass
    try:
        globals()['df_p{}'.format(i+1)]=(globals()['dfp{}'.format(i+1)]).values
    except (IOError, EOFError,KeyError) as e:
         pass
     
    try:
        globals()['df_t{}'.format(i+1)]=(globals()['dft{}'.format(i+1)]).values
    except (IOError, EOFError,KeyError) as e:
         pass
     
    try:
        globals()['df_oc1{}'.format(i+1)]=(globals()['dfoc1{}'.format(i+1)]).values
    except (IOError, EOFError,KeyError) as e:
         pass
     
    try:
        globals()['df_oc2{}'.format(i+1)]=(globals()['dfoc2{}'.format(i+1)]).values
    except (IOError, EOFError,KeyError) as e:
         pass
     

    
 
#bottma = dfa1['shorten'][np.where(dfa1['bottom']==1)[0][0]]
#bottmp = dfp1['shorten'][np.where(dfp1['bottom']==1)[0][0]]
#bottmt = dft1['shorten'][np.where(dft1['bottom']==1)[0][0]]


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='Times',palette="bright", color_codes=True)

    # Make the background white, and specify the
    # specific font family
    sns.set_style("ticks", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    
def set_style2():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='Times',palette="bright", color_codes=True)

       
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]})
#    sns.axes_style("whitegrid")
    
#==============================================================================
if (rho_contrast>0.):
    lim = 400
#    vel2=np.array([4,20,40,60,80])
    vel2=np.array([4,20,80])
    delrho = np.arange(0,81,2)

    slab80_vel80=np.zeros((len(delrho),2))
    slab80_vel60=np.zeros((len(delrho),2))
    slab80_vel40=np.zeros((len(delrho),2))
    slab80_vel20=np.zeros((len(delrho),2))
    slab80_vel4=np.zeros((len(delrho),2))
    
    slab160_vel80=np.zeros((len(delrho),2))
    slab160_vel60=np.zeros((len(delrho),2))
    slab160_vel40=np.zeros((len(delrho),2))
    slab160_vel20=np.zeros((len(delrho),2))
    slab160_vel4=np.zeros((len(delrho),2))
    
    no_points=len(vel2)
## SLAB 80 ####
    for i in range(len(delrho)):
        globals()['slab80v80min{}'.format(i)]=np.zeros((len(delrho),2))
#        globals()['slab80v60min{}'.format(i)]=np.zeros((len(delrho),2))
#        globals()['slab80v40min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab80v20min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab80v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        globals()['dfslab80v80_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_80.csv')
        globals()['slab80v80min{}'.format(i)]=np.zeros((len(delrho),2))
       
#        globals()['dfslab80v60_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_60.csv')
#        globals()['slab80v60min{}'.format(i)]=np.zeros((len(delrho),2))
#        
#        globals()['dfslab80v40_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_40.csv')
#        globals()['slab80v40min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab80v20_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_20.csv')
        globals()['slab80v20min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab80v4_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_4.csv')
        globals()['slab80v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        limit80=np.where(globals()['dfslab80v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v80_min{}'.format(i)]=min(globals()['dfslab80v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['slab80v80min{}'.format(i)][i,0] = (globals()['slab80v80_min{}'.format(i)])
        slab80_vel80[i,0] = (globals()['slab80v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfslab80v80_{}'.format(i)]['Ftot']==min(globals()['dfslab80v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        slab80_vel80[i,1] =globals()['dfslab80v80_{}'.format(i)]['time'][jj1]
        
#        limit60=np.where(globals()['dfslab80v60_{}'.format(i)]['shorten']<lim)[0][-1]
#        globals()['slab80v60_min{}'.format(i)]=min(globals()['dfslab80v60_{}'.format(i)]['Ftot'][0:limit60])/1e12
#        globals()['slab80v60min{}'.format(i)][i,0] = (globals()['slab80v60_min{}'.format(i)])
#        slab80_vel60[i,0] = (globals()['slab80v60_min{}'.format(i)])
#        jj2 = np.where(globals()['dfslab80v60_{}'.format(i)]['Ftot']==min(globals()['dfslab80v60_{}'.format(i)]['Ftot'][0:limit60]))[0][-1]
#        slab80_vel60[i,1] =np.round(globals()['dfslab80v60_{}'.format(i)]['time'][jj2],1)
#
#        limit40=np.where(globals()['dfslab80v40_{}'.format(i)]['shorten']<lim)[0][-1]
#        globals()['slab80v40_min{}'.format(i)]=min(globals()['dfslab80v40_{}'.format(i)]['Ftot'][0:limit40])/1e12
#        globals()['slab80v40min{}'.format(i)][i,0] = (globals()['slab80v40_min{}'.format(i)])
#        slab80_vel40[i,0] = (globals()['slab80v40_min{}'.format(i)])
#        jj3 = np.where(globals()['dfslab80v40_{}'.format(i)]['Ftot']==min(globals()['dfslab80v40_{}'.format(i)]['Ftot'][0:limit40]))[0][-1]
#        slab80_vel40[i,1] =np.round(globals()['dfslab80v40_{}'.format(i)]['time'][jj3],1)
        
        limit20=np.where(globals()['dfslab80v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v20_min{}'.format(i)]=min(globals()['dfslab80v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['slab80v20min{}'.format(i)][i,0] = (globals()['slab80v20_min{}'.format(i)])
        slab80_vel20[i,0] = (globals()['slab80v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfslab80v20_{}'.format(i)]['Ftot']==min(globals()['dfslab80v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        slab80_vel20[i,1] =globals()['dfslab80v20_{}'.format(i)]['time'][jj4]
        
        limit4=np.where(globals()['dfslab80v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v4_min{}'.format(i)]=min(globals()['dfslab80v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['slab80v4min{}'.format(i)][i,0] = (globals()['slab80v4_min{}'.format(i)])
        slab80_vel4[i,0] = (globals()['slab80v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfslab80v4_{}'.format(i)]['Ftot']==min(globals()['dfslab80v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        slab80_vel4[i,1] =globals()['dfslab80v4_{}'.format(i)]['time'][jj5]
 
## SLAB 160 ####           
    for i in range(len(delrho)):
        globals()['slab160v80min{}'.format(i)]=np.zeros((len(delrho),2))
#        globals()['slab160v60min{}'.format(i)]=np.zeros((len(delrho),2))
#        globals()['slab160v40min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab160v20min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab160v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        globals()['dfslab160v80_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_80.csv')
        globals()['slab160v80min{}'.format(i)]=np.zeros((len(delrho),2))
       
#        globals()['dfslab160v60_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_60.csv')
#        globals()['slab160v60min{}'.format(i)]=np.zeros((len(delrho),2))
#        
#        globals()['dfslab160v40_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_40.csv')
#        globals()['slab160v40min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab160v20_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_20.csv')
        globals()['slab160v20min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab160v4_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_4.csv')
        globals()['slab160v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        limit80=np.where(globals()['dfslab160v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v80_min{}'.format(i)]=min(globals()['dfslab160v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['slab160v80min{}'.format(i)][i,0] = (globals()['slab160v80_min{}'.format(i)])
        slab160_vel80[i,0] = (globals()['slab160v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfslab160v80_{}'.format(i)]['Ftot']==min(globals()['dfslab160v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        slab160_vel80[i,1] =globals()['dfslab160v80_{}'.format(i)]['time'][jj1]
        
#        limit60=np.where(globals()['dfslab160v60_{}'.format(i)]['shorten']<lim)[0][-1]
#        globals()['slab160v60_min{}'.format(i)]=min(globals()['dfslab160v60_{}'.format(i)]['Ftot'][0:limit60])/1e12
#        globals()['slab160v60min{}'.format(i)][i,0] = (globals()['slab160v60_min{}'.format(i)])
#        slab160_vel60[i,0] = (globals()['slab160v60_min{}'.format(i)])
#        jj2 = np.where(globals()['dfslab160v60_{}'.format(i)]['Ftot']==min(globals()['dfslab160v60_{}'.format(i)]['Ftot'][0:limit60]))[0][-1]
#        slab160_vel60[i,1] =np.round(globals()['dfslab160v60_{}'.format(i)]['time'][jj2],1)

#        limit40=np.where(globals()['dfslab160v40_{}'.format(i)]['shorten']<lim)[0][-1]
#        globals()['slab160v40_min{}'.format(i)]=min(globals()['dfslab160v40_{}'.format(i)]['Ftot'][0:limit40])/1e12
#        globals()['slab160v40min{}'.format(i)][i,0] = (globals()['slab160v40_min{}'.format(i)])
#        slab160_vel40[i,0] = (globals()['slab160v40_min{}'.format(i)])
#        jj3 = np.where(globals()['dfslab160v40_{}'.format(i)]['Ftot']==min(globals()['dfslab160v40_{}'.format(i)]['Ftot'][0:limit40]))[0][-1]
#        slab160_vel40[i,1] =np.round(globals()['dfslab160v40_{}'.format(i)]['time'][jj3],1)
        
        limit20=np.where(globals()['dfslab160v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v20_min{}'.format(i)]=min(globals()['dfslab160v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['slab160v20min{}'.format(i)][i,0] = (globals()['slab160v20_min{}'.format(i)])
        slab160_vel20[i,0] = (globals()['slab160v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfslab160v20_{}'.format(i)]['Ftot']==min(globals()['dfslab160v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        slab160_vel20[i,1] =globals()['dfslab160v20_{}'.format(i)]['time'][jj4]
        
        limit4=np.where(globals()['dfslab160v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v4_min{}'.format(i)]=min(globals()['dfslab160v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['slab160v4min{}'.format(i)][i,0] = (globals()['slab160v4_min{}'.format(i)])
        slab160_vel4[i,0] = (globals()['slab160v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfslab160v4_{}'.format(i)]['Ftot']==min(globals()['dfslab160v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        slab160_vel4[i,1] =globals()['dfslab160v4_{}'.format(i)]['time'][jj5]
    
    
    def ismin(df_file):
        limit=np.where(df_file['shorten']<400)[0][-1]
        jj=np.where(df_file['Ftot']==min(df_file['Ftot'][0:limit]))[0][-1]
        
        if df_file['Ftot'][jj] < df_file['Ftot'][jj+3]:
            isitmin=1
        else:
            isitmin=0
            
        return isitmin
        
    isitmin80 = np.zeros((len(delrho),len(vel2)))
    isitmin160 = np.zeros((len(delrho),len(vel2)))
    
#    for i in range(len(vel2)):
    for j in range(len(delrho)):
        if ismin(globals()['dfslab80v4_{}'.format(j)])==1:
            isitmin80[j,0]=1
        if ismin(globals()['dfslab80v20_{}'.format(j)])==1:
            isitmin80[j,1]=1
        if ismin(globals()['dfslab80v80_{}'.format(j)])==1:
            isitmin80[j,2]=1
        if ismin(globals()['dfslab160v4_{}'.format(j)])==1:
            isitmin160[j,0]=1
        if ismin(globals()['dfslab160v20_{}'.format(j)])==1:
            isitmin160[j,1]=1
        if ismin(globals()['dfslab160v80_{}'.format(j)])==1:
            isitmin160[j,2]=1
            
            
    def find3(df,delrho,th,k):
        zzz=np.round(df['Ftot'])
        idx=max(np.where(zzz>=-3e+12)[0])
        th[:,0] = delrho[:]
        if np.shape(np.where(zzz<=-3e+12))[1]==len(zzz):
            th[k,1] = 0
        else:
            th[k,1] = df['time'][idx]
#        th[k,1] = df['time'][idx]
        return th

    th80v4_ = np.zeros((len(delrho),2))
    th80v20_ = np.zeros((len(delrho),2))
    th80v80_ = np.zeros((len(delrho),2))
    th160v4_ = np.zeros((len(delrho),2))
    th160v20_ = np.zeros((len(delrho),2))
    th160v80_ = np.zeros((len(delrho),2))
    
    for i in range(len(delrho)):
        th80v4=find3(globals()['dfslab80v4_{}'.format(i)],delrho,th80v4_,i)
        th80v20=find3(globals()['dfslab80v20_{}'.format(i)],delrho,th80v20_,i)
        th80v80=find3(globals()['dfslab80v80_{}'.format(i)],delrho,th80v80_,i)
        th160v4=find3(globals()['dfslab160v4_{}'.format(i)],delrho,th160v4_,i)
        th160v20=find3(globals()['dfslab160v20_{}'.format(i)],delrho,th160v20_,i)
        th160v80=find3(globals()['dfslab160v80_{}'.format(i)],delrho,th160v80_,i)
    
#    zzz=np.round(dfslab160v4_4['Ftot'])
#    th160v4[4,1]=dfslab160v4_4['time'][max(np.where(zzz<-3e+12)[0])]
    ## Line fitting for 30mm/yr set
#    def exponenial_func(x, a, b, c):
#        return a*np.exp(-b*x)+c
    contrast = np.linspace(min(delrho), max(delrho), 11)
    fitdeg = 7
    fitdeg2=6
    
#    popt80_v80, pcov80_v80 = curve_fit(exponenial_func, delrho, np.array(slab80_vel80[:,0]), p0=(1, 1e-6, 1))
#    popt80_v20, pcov80_v20 = curve_fit(exponenial_func, delrho, np.array(slab80_vel20[:,0]), p0=(1, 1e-6, 1))
#    popt80_v4, pcov80_v4 = curve_fit(exponenial_func, delrho, np.array(slab80_vel4[:,0]), p0=(1, 1e-6, 1))
#    
#    popt160_v80, pcov160_v80 = curve_fit(exponenial_func, delrho, np.array(slab160_vel80[:,0]))
#    popt160_v20, pcov160_v20 = curve_fit(exponenial_func, delrho, np.array(slab160_vel20[:,0]))
#    popt160_v4, pcov160_v4 = curve_fit(exponenial_func, delrho, np.array(slab160_vel4[:,0]))
######----------------------------------------------------------###
    p80_v80 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab80_vel80[:,0]), fitdeg)))
#    p80_v60 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab80_vel60[:,0]), fitdeg)))
#    p80_v40 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab80_vel40[:,0]), fitdeg)))
    p80_v20 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab80_vel20[:,0]), fitdeg)))
    p80_v4 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab80_vel4[:,0]), fitdeg)))
######----------------------------------------------------------###
    p160_v80 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab160_vel80[:,0]),15)))
#    p160_v60 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab160_vel60[:,0]), fitdeg2)))
#    p160_v40 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab160_vel40[:,0]), fitdeg2)))
    p160_v20 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab160_vel20[:,0]), fitdeg2)))
    p160_v4 = np.poly1d(np.squeeze(np.polyfit(delrho,np.array(slab160_vel4[:,0]), fitdeg2)))
######----------------------------------------------------------###    
    xp = np.linspace(min(delrho), 150, 100)
    xp2=np.linspace(min(delrho),80, 100)
    xp3=np.linspace(min(delrho),70, 100)
    fig111=plt.figure(111,figsize=(11,8))
    fig111.clf()
    ax1 = fig111.add_subplot(111)
    linWid=1.5
    dot=50
    star=160
    xk =['xkcd:goldenrod','xkcd:orange','xkcd:orangered','xkcd:plum','xkcd:magenta']
    
    a4=np.where(isitmin80[:,0]==1)[0]
    a20=np.where(isitmin80[:,1]==1)[0]
    a80=np.where(isitmin80[:,2]==1)[0]
    b4=np.where(isitmin160[:,0]==1)[0]
    b20=np.where(isitmin160[:,1]==1)[0]
    b80=np.where(isitmin160[:,2]==1)[0]
    aa4=np.where(isitmin80[:,0]==0)[0]
    aa20=np.where(isitmin80[:,1]==0)[0]
    aa80=np.where(isitmin80[:,2]==0)[0]
    bb4=np.where(isitmin160[:,0]==0)[0]
    bb20=np.where(isitmin160[:,1]==0)[0]
    bb80=np.where(isitmin160[:,2]==0)[0]
    
    def bind(xp2,minfile):
        mix=np.zeros((len(xp2),2))
        mix[:,0]=xp2[:]
        for i in range(9):
            mix[np.where(np.round(mix[:,0])<=(8-i)*10),1]=minfile[8-i,1]
            
        return mix
    
    ax1.plot([], [], ' ', label=" ")
    ax1.annotate("80km", xy=(63.8,-6.3),xycoords='data',fontsize=16)
    fig111.texts.append(ax1.texts.pop())
    culor=[110,170,500]
    for i in range(len(vel2)):
        ax1.plot(xp,globals()['p80_v{}'.format(vel2[i])](xp), \
                color=plt.cm.winter_r(1.*i/(no_points)), linewidth=linWid, label=' ')
                
    ax1.plot([], [], ' ', label=" ")
    ax1.annotate("160km", xy=(69.6,-6.3),xycoords='data',fontsize=16)           
    fig111.texts.append(ax1.texts.pop())
    for i in range(len(vel2)):
        ax1.plot(xp,globals()['p160_v{}'.format(vel2[i])](xp), '--', \
                color=plt.cm.winter_r(1.*i/(no_points)), linewidth=linWid, label=str(vel2[i])) 
##        ax1.scatter(delrho,globals()['slab80_vel{}'.format(vel2[i])],s=dot, \
##                color=plt.cm.winter_r(1.*i/(no_points)), label=str(vel2[i]))
#        yy80=exponenial_func(xp,*globals()['popt80_v{}'.format(vel2[i])])
#        yy160=exponenial_func(xp,*globals()['popt160_v{}'.format(vel2[i])])
#        ax1.plot(xp,yy80,color=plt.cm.winter_r(1.*i/(no_points)), linewidth=linWid, label=str(vel2[i]))
#        ax1.plot(xp,yy160, '--', color=xk[i], linewidth=linWid, label=str(vel2[i])) 
#        
#                
#    for i in range(len(vel2)): 
#        if i in [0,1]:
#            ax1.plot(xp,globals()['p160_v{}'.format(vel2[i])](xp), '--', \
#                    color=xk[i], linewidth=linWid, label=str(vel2[i]))  
#            ax1.plot(delrho,globals()['slab160_vel{}'.format(vel2[i])], '--o', \
#                    color=xk[i], linewidth=linWid, label=str(vel2[i]))   
##            ax1.scatter(delrho,globals()['slab160_vel{}'.format(vel2[i])],s=dot, \
##                   color=xk[i], label=str(vel2[i]))
#        elif i in [2]:
#            ax1.plot(xp2,globals()['p160_v{}'.format(vel2[i])](xp2), '--', \
#                        color=xk[i], linewidth=linWid)    
#            ax1.plot(delrho[35:],globals()['slab160_vel{}'.format(vel2[i])][35:,0]*0, '--',\
#                        color=xk[i], linewidth=linWid, label=str(vel2[i]))  
#            ax1.scatter(delrho,globals()['slavel{}'.fb160_vel{}'.format(vel2[i])][i,0],s=dot, \
#                        c=globals()['slab160_ormat(vel2[i])][i,1],cmap=plt.cm.viridis, label=str(vel2[i]))
#    ax1.scatter(delrho[2:],slab160_vel4[2:,0],s=dot-20,color='k',label='160 km thick')
#    ax1.scatter(delrho[1:],slab80_vel4[1:,0],s=dot-20,marker='^',color='k',label='80 km thick')
#    ax1.scatter(delrho[1],slab80_vel4[1,0],s=star-20,marker='*', color='gray',label='Not Min.')
    zord=5
    for i in [a4][0]:
        ax1.scatter(delrho[i],slab80_vel4[i,0],s=dot,marker='^',zorder=zord,color=plt.cm.winter_r(1.*0/(no_points)))
    for i in [aa4][0]:
        ax1.scatter(delrho[i],slab80_vel4[i,0],s=star,marker='*',zorder=zord, color='gray')
    for i in [a20][0]:
        ax1.scatter(delrho[i],slab80_vel20[i,0],s=dot,marker='^',zorder=zord,color=plt.cm.winter_r(1.*1/(no_points)))
    for i in [aa20][0]:
        ax1.scatter(delrho[i],slab80_vel20[i,0],s=star,marker='*',zorder=zord, color='gray')
    for i in [a80][0]:
        ax1.scatter(delrho[i],slab80_vel80[i,0],s=dot,marker='^',zorder=zord,color=plt.cm.winter_r(1.*2/(no_points)))
    for i in [aa80][0]:
        ax1.scatter(delrho[i],slab80_vel80[i,0],s=star,marker='*',zorder=zord, color='gray')  
    
    for i in [b4][0]:
        ax1.scatter(delrho[i],slab160_vel4[i,0],s=dot,zorder=zord,color=plt.cm.winter_r(1.*0/(no_points)))
    for i in [bb4][0]:
        ax1.scatter(delrho[i],slab160_vel4[i,0],s=star,marker='*',zorder=zord, color='gray') 
    for i in [b20][0]:
        ax1.scatter(delrho[i],slab160_vel20[i,0],s=dot,zorder=zord,color=plt.cm.winter_r(1.*1/(no_points)))
    for i in [bb20][0]:
        ax1.scatter(delrho[i],slab160_vel20[i,0],s=star,marker='*',zorder=zord, color='gray') 
    for i in [b80][0]:
        ax1.scatter(delrho[i],slab160_vel80[i,0],s=dot,zorder=zord,color=plt.cm.winter_r(1.*2/(no_points)))
    for i in [bb80][0]:
        ax1.scatter(delrho[i],slab160_vel80[i,0],s=star,marker='*',zorder=zord, color='gray')         
        
    
    
#    ax1.text(delrho[1]*(1-0.1),(slab80_vel4[1,0])*(1-0.2),slab80_vel4[1,1],fontsize=17)
#    ax1.text(delrho[2]*(1-0.05),(slab80_vel4[2,0])*(1-0.25),slab80_vel4[2,1],fontsize=17)
#    ax1.text(delrho[3]*(1-0.05),(slab80_vel4[3,0])*(1-0.28),slab80_vel4[3,1],fontsize=17)
#    ax1.text(delrho[4]*(1-0.04),(slab80_vel4[4,0])*(1-0.6),slab80_vel4[4,1],fontsize=17)
#    ax1.text(delrho[5]*(1-0.03),(slab80_vel4[5,0])*(1-1.1),slab80_vel4[5,1],fontsize=17)
#    ax1.text(delrho[6]*(1-0.02),0.2,slab80_vel4[6,1],fontsize=17)
#    ax1.text(delrho[7]*(1-0.015),0.2,slab80_vel4[7,1],fontsize=17)
#    ax1.text(delrho[8]*(1-0.03),0.2,slab80_vel4[8,1],fontsize=17)
#    
#    ax1.text(delrho[2]*(1+0.05),(slab80_vel20[2,0])*(1-0.25),slab80_vel20[2,1],fontsize=17)
#    ax1.text(delrho[3]*(1-0.05),(slab80_vel20[3,0])*(1-0.28),slab80_vel20[3,1],fontsize=17)
#    ax1.text(delrho[4]*(1-0.04),(slab80_vel20[4,0])*(1-0.6),slab80_vel20[4,1],fontsize=17)
#    ax1.text(delrho[5]*(1-0.03),(slab80_vel20[5,0])*(1-1.1),slab80_vel20[5,1],fontsize=17)
#    ax1.text(delrho[6]*(1-0.02),0.2,slab80_vel20[6,1],fontsize=17)
#    ax1.text(delrho[7]*(1-0.015),0.2,slab80_vel20[7,1],fontsize=17)
#    ax1.text(delrho[8]*(1-0.03),0.2,slab80_vel20[8,1],fontsize=17)
    


#    ax1.fill([8,29,29,8],[-9,-9,2,2],'r',alpha=0.2)
#    ax1.fill([12,61,61,12],[-9,-9,2,2],'b',alpha=0.2)
#    ax1.fill([43,90,90,43],[-9,-9,2,2],'g',alpha=0.2)
    
    ax1.axvspan(8, 29,color = 'red', alpha = 0.08,zorder=0)
    ax1.axvspan(12, 61, color = 'green', alpha = 0.08,zorder=0)
    ax1.axvspan(43, 90, color = 'blue', alpha = 0.08,zorder=0)
    
#    def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):
#    
#        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
#                arrowprops={'arrowstyle': '|-|', 'color':linecolor, 'linewidth':linewidth},)
#        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
#                arrowprops={'arrowstyle': '<|-|>', 'color':linecolor, 'linewidth':linewidth})
#        xcenter = xmin + (xmax-xmin)/2
#        if ytext==0:
#            ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20
#        ax.annotate( text, xy=(xcenter,ytext), ha='center', va='center', fontsize=fontsize)
#            
#    def annotation_line2( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):
#    
#        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
#                arrowprops={'arrowstyle': '|-|', 'color':linecolor, 'linewidth':linewidth})

                          
#    annotation_line(ax=ax1, text=' ', xmin=8, xmax=29.1, \
#                    y=0.2, ytext=0.4, linewidth=2, linecolor='red', fontsize=16 )
#    annotation_line(ax=ax1, text='Avg. Proton', xmin=12, xmax=61.1, \
#                    y=0.4, ytext=0.2, linewidth=2, linecolor='green', fontsize=16 )
##    annotation_line2( ax=ax1, text='Avg. Archon', xmin=43, xmax=80, \
##                    y=0.2, ytext=0.4, linewidth=2, linecolor='blue', fontsize=16 )
#    ax1.annotate('', xy=(42.9, 0.2), xytext=(60, 0.2), xycoords='data', textcoords='data',
#                arrowprops={'arrowstyle': '-|>', 'color':'blue', 'linewidth':2})
#    ax1.annotate('', xy=(60, 0.2), xytext=(80, 0.2), xycoords='data', textcoords='data',
#                arrowprops={'arrowstyle': '-', 'ls': 'dashed', 'color':'blue', 'linewidth':2})
##    ax1.annotate( 'Avg. Archon', xy=(66,-7.6), ha='center', va='center', fontsize=18)  
#    ax1.annotate( 'Avg. Archon', xy=(70,0.38), ha='center', va='center', fontsize=16) 
#    ax1.annotate( '|', xy=(43.1,0.2),fontweight='bold', color='blue', ha='center', va='center', fontsize=23)   
#    ax1.annotate( 'Avg. Tecton', xy=(18,0), ha='center', va='center', fontsize=16)                  
#    ax1.text(67, -13.7, '80km', fontsize=22, rotation=90)
#    ax1.text(67, -17.1, '160km', fontsize=22, rotation=90)
    
    ax1.text(14,0.05, 'Avg. Tecton',fontsize=15,color='red')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(31,0.1, 'Avg. Proton',fontsize=15,color='green')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(57,0.05, 'Avg. Archon',fontsize=15,color='purple')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)

    ax1.fill([8,29,29,8],[0.3,0.3,0.4,0.4],color='red',alpha=0.5)
    ax1.fill([12,61,61,12],[0.4,0.4,0.5,0.5],color='green',alpha=0.5)
    ax1.fill([43,80,80,43],[0.3,0.3,0.4,0.4],color='purple',alpha=0.5)
                    
    hline=np.linspace(min(delrho),max(delrho))
    ax1.plot(hline,(hline*0)-3,'k',alpha=0.4)
    ax1.plot(hline,(hline*0),'gray',alpha=0.4)
    
    ax1.legend(title="v (mm/yr)",ncol=2,columnspacing=-0.2,loc='lower right', fontsize=16, frameon=True)._legend_box.align = "center"
    ax1.get_legend().get_title().set_fontsize('16')
    
  
    ax1.set_xlim(min(delrho),max(delrho))
    ax1.set_ylim(-8,0.5)
    
    ax1.set_title("$/bigtriangleup /rho_{LAB}$ vs Minimum $F_{buoy}$ (within d=400km)", fontsize=20, fontweight='bold', loc='center',y=1.02)
    ax1.set_xlabel('$/bigtriangleup /rho_{LAB} = /rho_{asth} - /rho_{lith}$ ($kg/m^{3}$)',fontsize=20,fontweight='bold',y=-1.5)
    ax1.set_ylabel('Slab pull / -F$_{bouy}$ ($10^{12}N/m$)',fontsize=20,fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='major', length=12,width=1)
    ax1.tick_params(which='minor', length=6)
    xminorLocator1   = MultipleLocator(2)
    yminorLocator1   = MultipleLocator(0.5)
    ax1.xaxis.set_minor_locator(xminorLocator1)
    ax1.yaxis.set_minor_locator(yminorLocator1)
    
    ax1.grid(linestyle='dotted')
    ax1.xaxis.grid() 
    plt.setp(ax1.spines.values(), color='k', linewidth=2)


#    cax, _ = matplotlib.colorbar.make_axes(ax1)
#    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
#    cbar.ax.tick_params(labelsize=16)
#    cbar.ax.set_ylabel('Convergence Rate ($mm/yr$)', rotation=270, fontsize=18)
#    cbar.ax.get_yaxis().labelpad = 30
#    tick_locs = (np.arange(min(vel2)-.5,max(vel2)+.5,len(vel)))
#    cbar.set_ticks(tick_locs)
#    fig111.tight_layout()
    
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig111.savefig('rhocontrast_Fb.png', format='png', dpi=300)
    os.chdir('..')
    
if (rho_contrast2>0.):
    lim = 400
#    vel2=np.array([4,20,40,60,80])
    vel2=np.array([4,20,80])
    delrho = np.arange(0,81,2) # [0,10,20,30,40,50,60,70,80,90,100]
    delrho2=np.arange(-10,32,2)
    
    slab80_vel80=np.zeros((len(delrho),2))
    slab80_vel20=np.zeros((len(delrho),2))
    slab80_vel4=np.zeros((len(delrho),2))
    
    slab160_vel80=np.zeros((len(delrho),2))
    slab160_vel20=np.zeros((len(delrho),2))
    slab160_vel4=np.zeros((len(delrho),2))
    
    oc60_vel80=np.zeros((len(delrho2),2))
    oc60_vel20=np.zeros((len(delrho2),2))
    oc60_vel4=np.zeros((len(delrho2),2))
    
    oc110_vel80=np.zeros((len(delrho2),2))
    oc110_vel20=np.zeros((len(delrho2),2))
    oc110_vel4=np.zeros((len(delrho2),2))
    
    no_points=len(vel2)

    for i in range(len(delrho)):
        globals()['slab80v80min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab80v20min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab80v4min{}'.format(i)]=np.zeros((len(delrho),2))        
        
        globals()['slab160v80min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab160v20min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab160v4min{}'.format(i)]=np.zeros((len(delrho),2))
   
    for i in range(len(delrho2)):     
        globals()['oc60v80min{}'.format(i)]=np.zeros((len(delrho2),2))
        globals()['oc60v20min{}'.format(i)]=np.zeros((len(delrho2),2))
        globals()['oc60v4min{}'.format(i)]=np.zeros((len(delrho2),2))
        
        globals()['oc110v80min{}'.format(i)]=np.zeros((len(delrho2),2))
        globals()['oc110v20min{}'.format(i)]=np.zeros((len(delrho2),2))
        globals()['oc110v4min{}'.format(i)]=np.zeros((len(delrho2),2))
        
    for i in range(len(delrho)):
        globals()['dfslab80v80_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_80.csv')
        globals()['slab80v80min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab80v20_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_20.csv')
        globals()['slab80v20min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab80v4_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_4.csv')
        globals()['slab80v4min{}'.format(i)]=np.zeros((len(delrho),2))        
        
        globals()['dfslab160v80_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_80.csv')
        globals()['slab160v80min{}'.format(i)]=np.zeros((len(delrho),2))      
        
        globals()['dfslab160v20_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_20.csv')
        globals()['slab160v20min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab160v4_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_4.csv')
        globals()['slab160v4min{}'.format(i)]=np.zeros((len(delrho),2))
    
    for i in range(len(delrho2)):    
        globals()['dfoc60v80_{}'.format(i)]=pd.read_csv('oc60_delRho'+str(delrho2[i])+'_vel_80.csv')
        globals()['oc60v80min{}'.format(i)]=np.zeros((len(delrho2),2))      
        
        globals()['dfoc60v20_{}'.format(i)]=pd.read_csv('oc60_delRho'+str(delrho2[i])+'_vel_20.csv')
        globals()['oc60v20min{}'.format(i)]=np.zeros((len(delrho2),2))
        
        globals()['dfoc60v4_{}'.format(i)]=pd.read_csv('oc60_delRho'+str(delrho2[i])+'_vel_4.csv')
        globals()['oc60v4min{}'.format(i)]=np.zeros((len(delrho2),2))
        
        globals()['dfoc110v80_{}'.format(i)]=pd.read_csv('oc110_delRho'+str(delrho2[i])+'_vel_80.csv')
        globals()['oc110v80min{}'.format(i)]=np.zeros((len(delrho2),2))      
        
        globals()['dfoc110v20_{}'.format(i)]=pd.read_csv('oc110_delRho'+str(delrho2[i])+'_vel_20.csv')
        globals()['oc110v20min{}'.format(i)]=np.zeros((len(delrho2),2))
        
        globals()['dfoc110v4_{}'.format(i)]=pd.read_csv('oc110_delRho'+str(delrho2[i])+'_vel_4.csv')
        globals()['oc110v4min{}'.format(i)]=np.zeros((len(delrho2),2))
        
        
## SLAB 80 ####        
    for i in range(len(delrho)):
        limit80=np.where(globals()['dfslab80v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v80_min{}'.format(i)]=min(globals()['dfslab80v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['slab80v80min{}'.format(i)][i,0] = (globals()['slab80v80_min{}'.format(i)])
        slab80_vel80[i,0] = (globals()['slab80v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfslab80v80_{}'.format(i)]['Ftot']==min(globals()['dfslab80v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        slab80_vel80[i,1] =np.round(globals()['dfslab80v80_{}'.format(i)]['time'][jj1],1)
        
        limit20=np.where(globals()['dfslab80v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v20_min{}'.format(i)]=min(globals()['dfslab80v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['slab80v20min{}'.format(i)][i,0] = (globals()['slab80v20_min{}'.format(i)])
        slab80_vel20[i,0] = (globals()['slab80v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfslab80v20_{}'.format(i)]['Ftot']==min(globals()['dfslab80v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        slab80_vel20[i,1] =np.round(globals()['dfslab80v20_{}'.format(i)]['time'][jj4],1)
        
        limit4=np.where(globals()['dfslab80v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v4_min{}'.format(i)]=min(globals()['dfslab80v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['slab80v4min{}'.format(i)][i,0] = (globals()['slab80v4_min{}'.format(i)])
        slab80_vel4[i,0] = (globals()['slab80v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfslab80v4_{}'.format(i)]['Ftot']==min(globals()['dfslab80v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        slab80_vel4[i,1] =np.round(globals()['dfslab80v4_{}'.format(i)]['time'][jj5],1)
 
## SLAB 160 ####           
    for i in range(len(delrho)):
        limit80=np.where(globals()['dfslab160v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v80_min{}'.format(i)]=min(globals()['dfslab160v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['slab160v80min{}'.format(i)][i,0] = (globals()['slab160v80_min{}'.format(i)])
        slab160_vel80[i,0] = (globals()['slab160v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfslab160v80_{}'.format(i)]['Ftot']==min(globals()['dfslab160v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        slab160_vel80[i,1] =np.round(globals()['dfslab160v80_{}'.format(i)]['time'][jj1],1)

        limit20=np.where(globals()['dfslab160v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v20_min{}'.format(i)]=min(globals()['dfslab160v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['slab160v20min{}'.format(i)][i,0] = (globals()['slab160v20_min{}'.format(i)])
        slab160_vel20[i,0] = (globals()['slab160v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfslab160v20_{}'.format(i)]['Ftot']==min(globals()['dfslab160v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        slab160_vel20[i,1] =np.round(globals()['dfslab160v20_{}'.format(i)]['time'][jj4],1)
        
        limit4=np.where(globals()['dfslab160v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v4_min{}'.format(i)]=min(globals()['dfslab160v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['slab160v4min{}'.format(i)][i,0] = (globals()['slab160v4_min{}'.format(i)])
        slab160_vel4[i,0] = (globals()['slab160v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfslab160v4_{}'.format(i)]['Ftot']==min(globals()['dfslab160v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        slab160_vel4[i,1] =np.round(globals()['dfslab160v4_{}'.format(i)]['time'][jj5],1)
    
    
## OC 60 ####           
    for i in range(len(delrho2)):
        limit80=np.where(globals()['dfoc60v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['oc60v80_min{}'.format(i)]=min(globals()['dfoc60v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['oc60v80min{}'.format(i)][i,0] = (globals()['oc60v80_min{}'.format(i)])
        oc60_vel80[i,0] = (globals()['oc60v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfoc60v80_{}'.format(i)]['Ftot']==min(globals()['dfoc60v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        oc60_vel80[i,1] =np.round(globals()['dfoc60v80_{}'.format(i)]['time'][jj1],1)

        limit20=np.where(globals()['dfoc60v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['oc60v20_min{}'.format(i)]=min(globals()['dfoc60v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['oc60v20min{}'.format(i)][i,0] = (globals()['oc60v20_min{}'.format(i)])
        oc60_vel20[i,0] = (globals()['oc60v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfoc60v20_{}'.format(i)]['Ftot']==min(globals()['dfoc60v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        oc60_vel20[i,1] =np.round(globals()['dfoc60v20_{}'.format(i)]['time'][jj4],1)
        
        limit4=np.where(globals()['dfoc60v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['oc60v4_min{}'.format(i)]=min(globals()['dfoc60v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['oc60v4min{}'.format(i)][i,0] = (globals()['oc60v4_min{}'.format(i)])
        oc60_vel4[i,0] = (globals()['oc60v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfoc60v4_{}'.format(i)]['Ftot']==min(globals()['dfoc60v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        oc60_vel4[i,1] =np.round(globals()['dfoc60v4_{}'.format(i)]['time'][jj5],1)
    
    
## OC 110 ####           
    for i in range(len(delrho2)):
        limit80=np.where(globals()['dfoc110v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['oc110v80_min{}'.format(i)]=min(globals()['dfoc110v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['oc110v80min{}'.format(i)][i,0] = (globals()['oc110v80_min{}'.format(i)])
        oc110_vel80[i,0] = (globals()['oc110v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfoc110v80_{}'.format(i)]['Ftot']==min(globals()['dfoc110v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        oc110_vel80[i,1] =np.round(globals()['dfoc110v80_{}'.format(i)]['time'][jj1],1)

        limit20=np.where(globals()['dfoc110v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['oc110v20_min{}'.format(i)]=min(globals()['dfoc110v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['oc110v20min{}'.format(i)][i,0] = (globals()['oc110v20_min{}'.format(i)])
        oc110_vel20[i,0] = (globals()['oc110v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfoc110v20_{}'.format(i)]['Ftot']==min(globals()['dfoc110v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        oc110_vel20[i,1] =np.round(globals()['dfoc110v20_{}'.format(i)]['time'][jj4],1)
        
        limit4=np.where(globals()['dfoc110v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['oc110v4_min{}'.format(i)]=min(globals()['dfoc110v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['oc110v4min{}'.format(i)][i,0] = (globals()['oc110v4_min{}'.format(i)])
        oc110_vel4[i,0] = (globals()['oc110v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfoc110v4_{}'.format(i)]['Ftot']==min(globals()['dfoc110v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        oc110_vel4[i,1] =np.round(globals()['dfoc110v4_{}'.format(i)]['time'][jj5],1)
        
        
#fig111=plt.figure(111,figsize=(11,8))
#for i in range(len(delrho2)):
#    plt.plot(globals()['dfoc60v80_{}'.format(i)]['time'],globals()['dfoc60v80_{}'.format(i)]['Ftot'])
#    hline=np.linspace(0,5)
#    plt.plot(hline,(hline*0)-3e12,'k',alpha=0.4)
#    plt.xlim(0,5)
#    plt.show()    
        
    def ismin(df_file):
        limit=np.where(df_file['shorten']<400)[0][-1]
        jj=np.where(df_file['Ftot']==min(df_file['Ftot'][0:limit]))[0][-1]
        
        if df_file['Ftot'][jj] < df_file['Ftot'][jj+3]:
            isitmin=1
        else:
            isitmin=0
            
        return isitmin
            
    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    
    def round_sig(x, sig=2):
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)
    
    def find3(df,delrho,th,k):
        limit=np.where(df['shorten']<400)[0][-1]
        if ismin(df)==1:
            lims=np.where(df['Ftot']==min(df['Ftot'][0:limit]))[0][-1]
        else:
            lims=np.where(df['shorten']<limit)[0][-1]

        zzz=np.array(df['Ftot'][0:lims])
        mmm=np.array([round_sig(mm,3) for mm in zzz])

        if lims == 0:
            th[k,1] = 0
        else:
            idx=max(np.where(mmm>=-3e+12)[0])
            th[k,1] = df['time'][idx]

        th[:,0] = delrho[:]
        return th

#df=dfslab160v4_0
#plt.plot(df['time'],df['Ftot']/1e12)  
#zzz=np.round(df['Ftot'])
#max(np.where(zzz<-3e+12)[0])
#len(zzz)
#zzz[max(np.where(zzz<-3e+12)[0])]
#df['time'][max(np.where(zzz<-3e+12)[0])]


    th80v4_ = np.zeros((len(delrho),2))
    th80v20_ = np.zeros((len(delrho),2))
    th80v80_ = np.zeros((len(delrho),2))
    th160v4_ = np.zeros((len(delrho),2))
    th160v20_ = np.zeros((len(delrho),2))
    th160v80_ = np.zeros((len(delrho),2))
    
    oc60v4_ = np.zeros((len(delrho2),2))
    oc60v20_ = np.zeros((len(delrho2),2))
    oc60v80_ = np.zeros((len(delrho2),2))
    oc110v4_ = np.zeros((len(delrho2),2))
    oc110v20_ = np.zeros((len(delrho2),2))
    oc110v80_ = np.zeros((len(delrho2),2))
    
    for i in range(len(delrho)):
        th80v4=find3(globals()['dfslab80v4_{}'.format(i)],delrho,th80v4_,i)
        th80v20=find3(globals()['dfslab80v20_{}'.format(i)],delrho,th80v20_,i)
        th80v80=find3(globals()['dfslab80v80_{}'.format(i)],delrho,th80v80_,i)
        th160v4=find3(globals()['dfslab160v4_{}'.format(i)],delrho,th160v4_,i)
        th160v20=find3(globals()['dfslab160v20_{}'.format(i)],delrho,th160v20_,i)
        th160v80=find3(globals()['dfslab160v80_{}'.format(i)],delrho,th160v80_,i)
        
    for i in range(len(delrho2)):
        oc60v4=find3(globals()['dfoc60v4_{}'.format(i)],delrho2,oc60v4_,i)
        oc60v20=find3(globals()['dfoc60v20_{}'.format(i)],delrho2,oc60v20_,i)
        oc60v80=find3(globals()['dfoc60v80_{}'.format(i)],delrho2,oc60v80_,i)       
        oc110v4=find3(globals()['dfoc110v4_{}'.format(i)],delrho2,oc110v4_,i)
        oc110v20=find3(globals()['dfoc110v20_{}'.format(i)],delrho2,oc110v20_,i)
        oc110v80=find3(globals()['dfoc110v80_{}'.format(i)],delrho2,oc110v80_,i)            
    
    zz7=np.array(dfslab80v20_7['Ftot'][0:np.where(dfslab80v20_7['shorten']<400)[0][-1]])
    mm7=np.array([round_sig(mm,3) for mm in zz7])
    th80v20[7,1]=dfslab80v20_7['time'][max(np.where(mm7>=-2.9e+12)[0])]


    
    fig111=plt.figure(111,figsize=(11,8))
    fig111.clf()
    ax1 = fig111.add_subplot(111)
#    ax2 = ax1.twinx()
    linWid=1.5
    dot=50
    star=220
    xk =['xkcd:goldenrod','xkcd:orange','xkcd:orangered','xkcd:plum','xkcd:magenta']
    
#    for i in range(len(vel2)):
#        ax1.plot(globals()['th80v{}'.format(vel2[i])][:,0],globals()['th80v{}'.format(vel2[i])][:,1],'-o', color=plt.cm.winter_r(1.*i/(no_points)))
#        ax2.plot(globals()['th160v{}'.format(vel2[i])][:,0],globals()['th160v{}'.format(vel2[i])][:,1],'--^',color=xk[i])

#    ax1.plot([], [], ' ', label=" ")
#    ax1.annotate("80km", xy=(50.8,1.18),xycoords='data',fontsize=14)
#    fig111.texts.append(ax1.texts.pop())
    ax1.semilogy(th80v4[:4,0],th80v4[:4,1],'-',linewidth=3, color='blue',alpha=0.6,label=' ')
    ax1.semilogy(th80v20[:8,0],th80v20[:8,1],'-', linewidth=3,color='blue',alpha=0.8,label=' ')
    ax1.semilogy(th80v80[:15,0],th80v80[:15,1],'-',linewidth=3, color='blue',alpha=1,label=' ')
    
    ax1.semilogy(th160v4[:21,0],th160v4[:21,1],'-',linewidth=3,color='red',alpha=0.6,label='4')
    ax1.semilogy(th160v20[:25,0],th160v20[:25,1],'-',linewidth=3,color='red',alpha=0.8,label='20')
    ax1.semilogy(th160v80[:28,0],th160v80[:28,1],'-',linewidth=3,color='red',alpha=1,label='80') 
    
#    ax1.semilogy(oc60v4[:12,0],oc60v4[:12,1],'--',linewidth=3, color='green',alpha=0.6,label=' ')
#    ax1.semilogy(oc60v20[:17,0],oc60v20[:17,1],'--',linewidth=3, color='green',alpha=0.8,label=' ')
#    ax1.semilogy(oc60v80[:,0],oc60v80[:,1],'--',linewidth=3, color='green',alpha=1,label=' ')
#    ax1.semilogy(oc110v4[:,0],oc110v4[:,1],'--',linewidth=3, color='k',alpha=0.6,label=' ')
#    ax1.semilogy(oc110v20[:,0],oc110v20[:,1],'--',linewidth=3, color='k',alpha=0.8,label=' ')
#    ax1.semilogy(oc110v80[:,0],oc110v80[:,1],'--',linewidth=3, color='k',alpha=1,label=' ')
#    ax1.plot([], [], ' ', label=" ")
#    ax1.annotate("160km", xy=(55,1.18),xycoords='data',fontsize=14)
#    fig111.texts.append(ax1.texts.pop())

    x_geo = [2, 12, 12]
    y_geo = [35, 2, 0.85]
    xerror = [3., 3., 3.]
    yerror = [10.,0.5,0.35]

    plt.errorbar(x_geo[0], y_geo[0], xerr=xerror[0], fmt='o',ecolor='blue',linewidth=2,markersize=12,markerfacecolor='blue',markeredgecolor='blue',capsize=5)
    plt.errorbar(x_geo[1], y_geo[1], xerr=xerror[1], fmt='o',ecolor='red',linewidth=2,markersize=12,markerfacecolor='red',markeredgecolor='red',capsize=5)
    plt.errorbar(x_geo[2], y_geo[2], xerr=xerror[2], fmt='o',ecolor='red',linewidth=2,markersize=12,markerfacecolor='red',markeredgecolor='red',capsize=5)
#    plt.errorbar(x_geo[2], y_geo[2], xerr=xerror, yerr=yerror, fmt='o',ecolor='sienna',linewidth=2,markersize=12,markerfacecolor='sienna',capsize=5)
#    ax1.text(x_geo[0],y_geo[0]+7, 'Alboran',fontsize=18,color='blue')
#    ax1.text(x_geo[1],y_geo[1]+0.7, 'Zagros',fontsize=18,color='red')
#    ax1.text(x_geo[2],y_geo[2]-0.2, 'Tibet',fontsize=18,color='red')
   
    
    txt001=ax1.annotate("Never reach $F_{b,ref}$", xy=(56,46),xycoords='data',ha='center',fontsize=18,color='k')
#    txt001.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
    txt002=ax1.annotate("Never reach $F_{b,ref}$", xy=(25,46),xycoords='data',ha='center',fontsize=18,color='k')

#    fig111.texts.append(ax1.texts.pop())
    
#    ax1.axvspan(8, 29, facecolor = 'red', alpha = 0.1,zorder=0)
#    ax1.axvspan(12, 61, facecolor = 'green', alpha = 0.1,zorder=0)
#    ax1.axvspan(43, 90, facecolor = 'blue', alpha = 0.1,zorder=0)

    xxx =np.linspace(0,500,400)
    xx=np.linspace(0,80,100)
#    yyy1=p_thick(np.linspace(0,80,100))
#    yyy2=p_thin(np.linspace(0,80,100))
    yyy1= 0.8693009 + (56839.67*np.exp(xxx[:]*-0.178609))  #160
    yyy2= 4.523505+(934.7117*np.exp(xxx[:]*-0.3739652))  # 80
    yyy3 = (yyy1*0)
    ax1.fill(xxx, yyy1, 'r', alpha=0.4,zorder=1)
    ax1.fill_between(xxx, yyy1,yyy2,where=yyy1 >yyy2, facecolor='blue', alpha=0.4,zorder=1)

#    ax1.fill_between(xxx, yyy2, yyy3, where=(xxx>=8) & (xxx<=29) , facecolor='red', alpha=0.2,zorder=1)
#    ax1.fill_between(xxx, yyy2, yyy3, where=(xxx>=12) & (xxx<=54) , facecolor='green', alpha=0.2,zorder=1)
#    ax1.fill_between(xxx, yyy1, yyy3, where=(xxx>=53) & (xxx<=61.5) , facecolor='green', alpha=0.2,zorder=1)
#    ax1.fill_between(xxx, yyy1, yyy3, where=(xxx>53) & (xxx<=80) , facecolor='orange', alpha=0.2,zorder=1)
#    ax1.fill_between(xxx, yyy2, yyy3, where=(xxx>43) & (xxx<=54) , facecolor='orange', alpha=0.2,zorder=1)

    
    def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):
    
        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
                arrowprops={'arrowstyle': '|-|', 'color':linecolor, 'linewidth':linewidth},)
        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
                arrowprops={'arrowstyle': '<|-|>', 'color':linecolor, 'linewidth':linewidth})
        xcenter = xmin + (xmax-xmin)/2
        if ytext==0:
            ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20
        ax.annotate( text, xy=(xcenter,ytext), ha='center', va='center', fontsize=fontsize)
#            

    ax1.text(22,0.158, 'Tecton',fontsize=15,color='red',bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)   
#    ax1.text(19,0.428, 'Tecton',fontsize=15,color='red',bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(32,0.131, 'Proton',fontsize=15,color='green',bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(52,0.158, 'Archon',fontsize=15,color='purple',bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(7,0.192, 'Oceanic',fontsize=15,color='magenta',bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)

    ax1.fill([8,29,29,8],[0.124,0.124,0.15,0.15],color='red',alpha=0.4)
    ax1.fill([12,61,61,12],[0.1,0.1,0.124,0.124],color='green',alpha=0.4)
    ax1.fill([43,70,70,43],[0.124,0.124,0.15,0.15],color='purple',alpha=0.4)
    ax1.fill([0,17,17,0],[0.15,0.15,0.185,0.185],color='magenta',alpha=0.5)
    

    hline=np.linspace(min(delrho),max(delrho))
    ax1.plot(hline,hline*0,'k',alpha=0.4)

    txt1=ax1.text(32,18, '4',fontsize=18,color='red')
    txt1.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    txt2=ax1.text(42,4.7, '20',fontsize=18,color='red')
    txt2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    ax1.text(48,1.08, '80  $mm$ $yr^{-1}$',fontsize=18,color='red',ha='center')
    
    ax1.text(3.5,50, '4',fontsize=18,color='blue')
    ax1.text(6,5, '20',fontsize=18,color='blue')
    ax1.text(28,2., '80 $mm$ $yr^{-1}$',fontsize=18,color='blue',ha='center')
    
    ax1.plot(17,0.197, marker=r'$\downarrow$',markersize=12,color='magenta')
    ax1.plot(19,0.165, marker=r'$\downarrow$',markersize=12,color='red')
    ax1.plot(39,0.137, marker=r'$\downarrow$',markersize=12,color='green')
    ax1.plot(68,0.165, marker=r'$\downarrow$',markersize=12,color='purple')

#    ax1.legend(title="v (mm/yr)",ncol=2,columnspacing=-0.2,labelspacing=-0.01,loc='lower right', fontsize=15,frameon=True,bbox_to_anchor=(0.98,0.08))
#    ax1.get_legend().get_title().set_fontsize('16')
  
    ax1.set_xlim(min(delrho),70)
    ax1.set_ylim((0.1, 110))
#    ax2.set_ylim(0,15)
    
#    ax1.set_title("$F_{buoy}$ = -3e12 $N/m$", fontsize=20, fontweight='bold', loc='center',y=1.02)
    ax1.set_xlabel('$/bigtriangleup /rho_{LAB} = /rho_{asth} - /rho_{lith}$ ($kg m^{-3}$)',fontsize=20,fontweight='bold')
    ax1.set_ylabel('Time needed for $F_{b,ref}$ = -3 $TNm^{-1}$ ($Myr$)',fontsize=20,fontweight='bold')

    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='major', length=12,width=1,labelsize=20)
    ax1.tick_params(which='minor', length=6)
    
#    ax2.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=False, labelright=True,
#             bottom=True, top=True, left=False, right=True,color='r')
#    ax2.tick_params(which='major', length=12,width=1,labelsize=20)
#    ax2.tick_params(which='minor', length=6)
    
    xminorLocator1   = MultipleLocator(2)
#    yminorLocator1   = MultipleLocator(10)
    ax1.xaxis.set_minor_locator(xminorLocator1)
#    ax1.yaxis.set_minor_locator(MultipleLocator(1))
#    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    
    ax1.grid(linestyle='dotted')
    ax1.xaxis.grid() 
    plt.setp(ax1.spines.values(), color='k', linewidth=2)
    ax1.set_yticklabels(['10','0.1','1','10','100'])

#    cax, _ = matplotlib.colorbar.make_axes(ax1)
#    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
#    cbar.ax.tick_params(labelsize=16)
#    cbar.ax.set_ylabel('Convergence Rate ($mm/yr$)', rotation=270, fontsize=18)
#    cbar.ax.get_yaxis().labelpad = 30
#    tick_locs = (np.arange(min(vel2)-.5,max(vel2)+.5,len(vel)))
#    cbar.set_ticks(tick_locs)
#    fig111.tight_layout()
    
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig111.savefig('rhocontrast_Fb_2.png', format='png', dpi=300)
    os.chdir('..')    
    
    
if (rho_contrast3>0.):
    lim = 400
#    vel2=np.array([4,20,40,60,80])
    vel2=np.array([4,20,80])
    delrho = np.arange(0,81,2) # [0,10,20,30,40,50,60,70,80,90,100]

    slab80_vel80=np.zeros((len(delrho),2))
    slab80_vel60=np.zeros((len(delrho),2))
    slab80_vel40=np.zeros((len(delrho),2))
    slab80_vel20=np.zeros((len(delrho),2))
    slab80_vel4=np.zeros((len(delrho),2))
    
    slab160_vel80=np.zeros((len(delrho),2))
    slab160_vel60=np.zeros((len(delrho),2))
    slab160_vel40=np.zeros((len(delrho),2))
    slab160_vel20=np.zeros((len(delrho),2))
    slab160_vel4=np.zeros((len(delrho),2))
    
    no_points=len(vel2)
## SLAB 80 ####
    for i in range(len(delrho)):
        globals()['slab80v80min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab80v20min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab80v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        globals()['dfslab80v80_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_80.csv')
        globals()['slab80v80min{}'.format(i)]=np.zeros((len(delrho),2))

        globals()['dfslab80v20_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_20.csv')
        globals()['slab80v20min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab80v4_{}'.format(i)]=pd.read_csv('slab80_delRho'+str(delrho[i])+'_vel_4.csv')
        globals()['slab80v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        limit80=np.where(globals()['dfslab80v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v80_min{}'.format(i)]=min(globals()['dfslab80v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['slab80v80min{}'.format(i)][i,0] = (globals()['slab80v80_min{}'.format(i)])
        slab80_vel80[i,0] = (globals()['slab80v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfslab80v80_{}'.format(i)]['Ftot']==min(globals()['dfslab80v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        slab80_vel80[i,1] =np.round(globals()['dfslab80v80_{}'.format(i)]['time'][jj1],1)
        
        limit20=np.where(globals()['dfslab80v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v20_min{}'.format(i)]=min(globals()['dfslab80v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['slab80v20min{}'.format(i)][i,0] = (globals()['slab80v20_min{}'.format(i)])
        slab80_vel20[i,0] = (globals()['slab80v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfslab80v20_{}'.format(i)]['Ftot']==min(globals()['dfslab80v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        slab80_vel20[i,1] =np.round(globals()['dfslab80v20_{}'.format(i)]['time'][jj4],1)
        
        limit4=np.where(globals()['dfslab80v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab80v4_min{}'.format(i)]=min(globals()['dfslab80v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['slab80v4min{}'.format(i)][i,0] = (globals()['slab80v4_min{}'.format(i)])
        slab80_vel4[i,0] = (globals()['slab80v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfslab80v4_{}'.format(i)]['Ftot']==min(globals()['dfslab80v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        slab80_vel4[i,1] =np.round(globals()['dfslab80v4_{}'.format(i)]['time'][jj5],1)
 
## SLAB 160 ####           
    for i in range(len(delrho)):
        globals()['slab160v80min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab160v20min{}'.format(i)]=np.zeros((len(delrho),2))
        globals()['slab160v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        globals()['dfslab160v80_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_80.csv')
        globals()['slab160v80min{}'.format(i)]=np.zeros((len(delrho),2))

        globals()['dfslab160v20_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_20.csv')
        globals()['slab160v20min{}'.format(i)]=np.zeros((len(delrho),2))
        
        globals()['dfslab160v4_{}'.format(i)]=pd.read_csv('slab160_delRho'+str(delrho[i])+'_vel_4.csv')
        globals()['slab160v4min{}'.format(i)]=np.zeros((len(delrho),2))
    for i in range(len(delrho)):
        limit80=np.where(globals()['dfslab160v80_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v80_min{}'.format(i)]=min(globals()['dfslab160v80_{}'.format(i)]['Ftot'][0:limit80])/1e12
        globals()['slab160v80min{}'.format(i)][i,0] = (globals()['slab160v80_min{}'.format(i)])
        slab160_vel80[i,0] = (globals()['slab160v80_min{}'.format(i)])
        jj1 = np.where(globals()['dfslab160v80_{}'.format(i)]['Ftot']==min(globals()['dfslab160v80_{}'.format(i)]['Ftot'][0:limit80]))[0][-1]
        slab160_vel80[i,1] =np.round(globals()['dfslab160v80_{}'.format(i)]['time'][jj1],1)
             
        limit20=np.where(globals()['dfslab160v20_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v20_min{}'.format(i)]=min(globals()['dfslab160v20_{}'.format(i)]['Ftot'][0:limit20])/1e12
        globals()['slab160v20min{}'.format(i)][i,0] = (globals()['slab160v20_min{}'.format(i)])
        slab160_vel20[i,0] = (globals()['slab160v20_min{}'.format(i)])
        jj4 = np.where(globals()['dfslab160v20_{}'.format(i)]['Ftot']==min(globals()['dfslab160v20_{}'.format(i)]['Ftot'][0:limit20]))[0][-1]
        slab160_vel20[i,1] =np.round(globals()['dfslab160v20_{}'.format(i)]['time'][jj4],1)
        
        limit4=np.where(globals()['dfslab160v4_{}'.format(i)]['shorten']<lim)[0][-1]
        globals()['slab160v4_min{}'.format(i)]=min(globals()['dfslab160v4_{}'.format(i)]['Ftot'][0:limit4])/1e12
        globals()['slab160v4min{}'.format(i)][i,0] = (globals()['slab160v4_min{}'.format(i)])
        slab160_vel4[i,0] = (globals()['slab160v4_min{}'.format(i)])
        jj5 = np.where(globals()['dfslab160v4_{}'.format(i)]['Ftot']==min(globals()['dfslab160v4_{}'.format(i)]['Ftot'][0:limit4]))[0][-1]
        slab160_vel4[i,1] =np.round(globals()['dfslab160v4_{}'.format(i)]['time'][jj5],1)
    
    
    def round_sig(x, sig=2):
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)
    
    def find3(df,delrho,th,k):
        limit=np.where(df['shorten']<400)[0][-1]
        if ismin(df)==1:
            lims=np.where(df['Ftot']==min(df['Ftot'][0:limit]))[0][-1]
        else:
            lims=np.where(df['shorten']<limit)[0][-1]

        zzz=np.array(df['Ftot'][0:lims])
        mmm=np.array([round_sig(mm,3) for mm in zzz])

        if lims == 0:
            th[k,1] = 0
        else:
            idx=max(np.where(mmm>=-3e+12)[0])
            th[k,1] = df['time'][idx]

        th[:,0] = delrho[:]
        return th

    th80v4_ = np.zeros((len(delrho),2))
    th80v20_ = np.zeros((len(delrho),2))
    th80v80_ = np.zeros((len(delrho),2))
    th160v4_ = np.zeros((len(delrho),2))
    th160v20_ = np.zeros((len(delrho),2))
    th160v80_ = np.zeros((len(delrho),2))
    
    for i in range(len(delrho)):
        th80v4=find3(globals()['dfslab80v4_{}'.format(i)],delrho,th80v4_,i)
        th80v20=find3(globals()['dfslab80v20_{}'.format(i)],delrho,th80v20_,i)
        th80v80=find3(globals()['dfslab80v80_{}'.format(i)],delrho,th80v80_,i)
        th160v4=find3(globals()['dfslab160v4_{}'.format(i)],delrho,th160v4_,i)
        th160v20=find3(globals()['dfslab160v20_{}'.format(i)],delrho,th160v20_,i)
        th160v80=find3(globals()['dfslab160v80_{}'.format(i)],delrho,th160v80_,i)
    
    zz7=np.array(dfslab80v20_7['Ftot'][0:np.where(dfslab80v20_7['shorten']<400)[0][-1]])
    mm7=np.array([round_sig(mm,3) for mm in zz7])
    th80v20[7,1]=dfslab80v20_7['time'][max(np.where(mm7>=-2.9e+12)[0])]

    contrast = np.linspace(min(delrho), max(delrho), 11)

    xp = np.linspace(min(delrho), max(delrho), 100)
    xp2=np.linspace(min(delrho),80, 100)
    
    
    fig131=plt.figure(131,figsize=(15,8))
    fig131.clf()
    ax1 = fig131.add_subplot(121)
    ax2 = fig131.add_subplot(122)
    linWid=1.5
    dot=50
    star=220
    xk =['xkcd:goldenrod','xkcd:orange','xkcd:orangered','xkcd:plum','xkcd:magenta']
    
    th160v80[29::,1]=100
    th80v20[9::,1]=100
    th80v80[16::,1]=100
    
#    for i in range(len(vel2)):
#        ax1.plot(globals()['th80v{}'.format(vel2[i])][:,0],globals()['th80v{}'.format(vel2[i])][:,1],'-o', color=plt.cm.winter_r(1.*i/(no_points)))
#        ax2.plot(globals()['th160v{}'.format(vel2[i])][:,0],globals()['th160v{}'.format(vel2[i])][:,1],'--^',color=xk[i])

    ax1.semilogy(th80v4[:4,0],th80v4[:4,1],'-',markersize=7, color=plt.cm.gray_r(110),label=' ',alpha=0)
    ax1.fill_between(th80v4[:,0],th80v20[:,1],th80v4[:,1],color='blue',alpha=0.2,zorder=0)
    ax1.semilogy(th80v20[:8,0],th80v20[:8,1],'-',markersize=7, color=plt.cm.gray_r(170),label=' ',alpha=0)
    ax1.fill_between(th80v20[:,0],th80v80[:,1],th80v20[:,1],color='blue',alpha=0.4,zorder=0)
    ax1.semilogy(th80v80[:15,0],th80v80[:15,1],'-',markersize=7, color=plt.cm.gray_r(500),label=' ',alpha=0)
    ax1.fill_between(th80v80[:,0],0,th80v80[:,1],color='blue',alpha=0.6,zorder=0)

    ax2.semilogy(th160v4[:21,0],th160v4[:21,1],'--',markersize=7,color=plt.cm.gray_r(110),label=' ',alpha=0)
    ax2.fill_between(th160v4[:,0],th160v20[:,1],th160v4[:,1],color='red',alpha=0.2,zorder=0)
    ax2.semilogy(th160v20[:25,0],th160v20[:25,1],'--',markersize=7,color=plt.cm.gray_r(170),label=' ',alpha=0)
    ax2.fill_between(th160v20[:,0],th160v80[:,1],th160v20[:,1],color='red',alpha=0.4,zorder=0)
    ax2.semilogy(th160v80[:28,0],th160v80[:28,1],'--',markersize=7,color=plt.cm.gray_r(500),label=' ',alpha=0)
    ax2.fill_between(th160v80[:,0],0,th160v80[:,1],color='red',alpha=0.6,zorder=0)
    
    ax2.annotate("no delamination \n $h_L = 160 km$", xy=(52.7,46),xycoords='data',ha='center',fontsize=16,bbox=dict(facecolor='white', alpha=1, edgecolor='white'))
    ax1.annotate("no delamination \n $h_L = 80 km$", xy=(38,46),xycoords='data',ha='center',fontsize=16,bbox=dict(facecolor='white', alpha=1, edgecolor='white'))
#    fig111.texts.append(ax1.texts.pop())
    
#    ax1.axvspan(8, 29, facecolor = 'red', alpha = 0.1,zorder=0)
#    ax1.axvspan(12, 61, facecolor = 'green', alpha = 0.1,zorder=0)
#    ax1.axvspan(43, 90, facecolor = 'blue', alpha = 0.1,zorder=0)

    xxx =np.linspace(0,500,400)
    xx=np.linspace(0,80,100)
#    yyy1=p_thick(np.linspace(0,80,100))
#    yyy2=p_thin(np.linspace(0,80,100))
    yyy1= 0.8693009 + (56839.67*np.exp(xxx[:]*-0.178609))  #160
    yyy2= 4.523505+(934.7117*np.exp(xxx[:]*-0.3739652))  # 80
    yyy3 = (yyy1*0)
#    ax2.fill(xxx, yyy1, 'grey', alpha=1,zorder=2,hatch='/')
    ax2.fill(xxx, yyy1, 'grey', alpha=1,zorder=1,hatch='//')
    ax1.fill(xxx, yyy2, facecolor='grey', alpha=1,zorder=1,hatch='//')
    
    ffont=16
    ax2.text(45,0.7,'v>80 mm/yr',fontsize=ffont)
    ax2.text(25,1.7,'20<v<80 mm/yr',fontsize=ffont)
    ax2.text(15,8,'4<v<20 mm/yr',fontsize=ffont)
    
    ax1.text(38,1,'v>80 mm/yr',fontsize=ffont)
    ax1.text(5,3,'20<v<80 mm/yr',fontsize=ffont)
    ax1.text(2.5,10.8,'4<v<20 \n mm/yr',fontsize=ffont)

    
    ax1.text(12,0.428, 'Avg. Tecton',fontsize=15,color='black')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(29.3,0.428, 'Avg. Proton',fontsize=15,color='black')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.text(47,0.428, 'Avg. Archon',fontsize=15,color='black')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax1.fill([8,29,29,8],[0.38,0.38,0.41,0.41],color='red',alpha=1)
    ax1.fill([12,61,61,12],[0.35,0.35,0.38,0.38],color='green',alpha=1)
    ax1.fill([43,64,64,43],[0.38,0.38,0.41,0.41],color='purple',alpha=1)
    
    
    ax2.text(12,0.428, 'Avg. Tecton',fontsize=15,color='black')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax2.text(29.3,0.428, 'Avg. Proton',fontsize=15,color='black')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax2.text(47,0.428, 'Avg. Archon',fontsize=15,color='black')#,bbox=dict(facecolor='white', alpha=1, edgecolor='white'),zorder=0)
    ax2.fill([8,29,29,8],[0.38,0.38,0.41,0.41],color='red',alpha=1)
    ax2.fill([12,61,61,12],[0.35,0.35,0.38,0.38],color='green',alpha=1)
    ax2.fill([43,64,64,43],[0.38,0.38,0.41,0.41],color='purple',alpha=1)
  
    hline=np.linspace(min(delrho),max(delrho))
    ax1.plot(hline,hline*0,'k',alpha=0.4)


#    ax1.legend(title="v (mm/yr)",ncol=2,columnspacing=-0.2,labelspacing=-0.01,loc='lower right', fontsize=15,frameon=True,bbox_to_anchor=(0.98,0.08))
#    ax1.get_legend().get_title().set_fontsize('16')
  
    ax1.set_xlim(min(delrho),64)
    ax2.set_xlim(min(delrho),64)
    ax1.set_ylim((0.35, 110))
    ax2.set_ylim((0.35, 110))

    
    ax1.set_title("$F_{buoy}$ = -3e12 $N/m$, wihin d=400km", fontsize=20, fontweight='bold', loc='center',y=1.02)
    ax1.set_xlabel('$/bigtriangleup /rho_{LAB} = /rho_{asth} - /rho_{lith}$ ($kg/m^{3}$)',fontsize=20,fontweight='bold')
    ax1.set_ylabel('log(Time) ($Myr$)',fontsize=20,fontweight='bold')
    
    ax2.set_title("$F_{buoy}$ = -3e12 $N/m$, wihin d=400km", fontsize=20, fontweight='bold', loc='center',y=1.02)
    ax2.set_xlabel('$/bigtriangleup /rho_{LAB} = /rho_{asth} - /rho_{lith}$ ($kg/m^{3}$)',fontsize=20,fontweight='bold')

    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True,zorder=9)
    ax1.tick_params(which='major', length=12,width=1,labelsize=20)
    ax1.tick_params(which='minor', length=6)
    
    ax2.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True,zorder=9)
    ax2.tick_params(which='major', length=12,width=1,labelsize=20)
    ax2.tick_params(which='minor', length=6)
    

    
    xminorLocator1   = MultipleLocator(2)
    ax1.xaxis.set_minor_locator(xminorLocator1)
    ax2.xaxis.set_minor_locator(xminorLocator1)
    ax2.xaxis.set_major_locator(MultipleLocator(10))

    
#    ax1.grid(linestyle='dotted')
#    ax1.xaxis.grid() 
#    plt.setp(ax1.spines.values(), color='k', linewidth=2)
    ax1.set_yticklabels(['1','10','1','10','100','10'])
    ax2.set_yticklabels(['1','10','1','10','100','10'])


    fig131.tight_layout()
    
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig131.savefig('rhocontrast_Fb_3.png', format='png', dpi=300)
    os.chdir('..')    
    
if(rho_contrast_rho>0.):
#    os.chdir('/home/kittiphon/ownCloud/PhD/delam_csv/')
    T0=loadtxt('Trho_slab80_delRho0_vel_60.txt', delimiter=",")
    T1=loadtxt('Trho_slab80_delRho10_vel_60.txt', delimiter=",")
    T2=loadtxt('Trho_slab80_delRho20_vel_60.txt', delimiter=",")
    T3=loadtxt('Trho_slab80_delRho30_vel_60.txt', delimiter=",")
    T4=loadtxt('Trho_slab80_delRho40_vel_60.txt', delimiter=",")
    T5=loadtxt('Trho_slab80_delRho50_vel_60.txt', delimiter=",")
    T6=loadtxt('Trho_slab80_delRho60_vel_60.txt', delimiter=",")
    T7=loadtxt('Trho_slab80_delRho70_vel_60.txt', delimiter=",")
    T8=loadtxt('Trho_slab80_delRho80_vel_60.txt', delimiter=",")
    T9=loadtxt('Trho_slab80_delRho90_vel_60.txt', delimiter=",")
    T10=loadtxt('Trho_slab80_delRho100_vel_60.txt', delimiter=",")

#    T0=loadtxt('Trho_slab160_delRho0_vel_20.txt', delimiter=",")
#    T1=loadtxt('Trho_slab160_delRho10_vel_20.txt', delimiter=",")
#    T2=loadtxt('Trho_slab160_delRho20_vel_20.txt', delimiter=",")
#    T3=loadtxt('Trho_slab160_delRho30_vel_20.txt', delimiter=",")
#    T4=loadtxt('Trho_slab160_delRho40_vel_20.txt', delimiter=",")
#    T5=loadtxt('Trho_slab160_delRho50_vel_20.txt', delimiter=",")
#    T6=loadtxt('Trho_slab160_delRho60_vel_20.txt', delimiter=",")
#    T7=loadtxt('Trho_slab160_delRho70_vel_20.txt', delimiter=",")
#    T8=loadtxt('Trho_slab160_delRho80_vel_20.txt', delimiter=",")
#    T9=loadtxt('Trho_slab160_delRho90_vel_20.txt', delimiter=",")
#    T10=loadtxt('Trho_slab160_delRho100_vel_20.txt', delimiter=",")
    delrho = [0,10,20,30,40,50,60,70,80,90,100]
    fig8 = plt.figure(8, figsize=(9,7))
    fig8.clf()
    
    xmajorLocator1   = FixedLocator(np.arange(200,1600,300))
    xminorLocator1   = MultipleLocator(50)
    xminorLocator2   = MultipleLocator(20)
    ymajorLocator1   = FixedLocator(np.arange(0,601,100))
    yminorLocator1   = MultipleLocator(10)
    
#    set_style2()
    
#    ax1 = plt.subplot2grid((2,2), (0,0), rowspan=2, colspan=1)
#    ax2 = plt.subplot2grid((2,2), (0,1), rowspan=2, colspan=2)
    ax1=plt.subplot(121)
    ax2=plt.subplot(122, sharey=ax1)
    
    hline1=np.linspace(min(T10[:,1]),max(T10[:,1]))
    hline2=np.linspace(min(T10[:,2]),max(T10[:,2]))
    
#    dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)] 
#    up_lim = 0. 
#    up_lim1 = -10.  
#    up_lim2 = -40. 
#    uplim=np.int(np.where(np.round(T0[:,0])==up_lim)[0])
#    uplim1=np.int(np.where(np.floor(T0[:,0])==up_lim1)[0])+1
#    uplim2=np.int(np.where(np.round(T0[:,0])==up_lim2)[0])+1
    opaq=0.3    
    
#    ax1.plot(hline1,hline1*0+10,color = 'k',linestyle='-', linewidth=1,alpha=0.2)  

    ax1.set_xlabel('Temperature ($^{\circ}C$)',fontweight='normal',fontsize=20)
    ax1.set_ylabel('Depth ($km$)',fontweight='normal',fontsize=20)
    ax1.invert_yaxis()
#    ax1.set_ylim(0,600e3)
#    ax1.set_xlim(200,max(T0[:,1]))
#    ax1.legend(loc='lower left',fontsize=16)

    ax1.plot(hline1,hline1*0+40,color = 'k',linestyle='-', linewidth=1,alpha=0.2) 
    ax2.plot(hline2,hline2*0+40,color = 'k',linestyle='-', linewidth=1,alpha=0.2) 
    ax1.plot(hline1,hline1*0+120,color = 'b',linestyle='--', linewidth=1,alpha=opaq)
    ax2.plot(hline2,hline2*0+120,color = 'b',linestyle='--', linewidth=1,alpha=opaq)
    
    no_points=len(delrho)
    for i in range(len(delrho)):
        ax1.plot(globals()['T{}'.format(i)][:,1],(globals()['T{}'.format(i)][:,0])/1e3, \
                color=plt.cm.jet(1.*i/(no_points)),linewidth=1,label=str(delrho[i]))
        ax2.plot(globals()['T{}'.format(i)][:,2],(globals()['T{}'.format(i)][:,0])/1e3, \
                color=plt.cm.jet(1.*i/(no_points)),linewidth=1,label=str(delrho[i]))

    
    ax2.set_xlabel('Density ($kg/m^3}$)',fontweight='normal',fontsize=20)

    ax2.set_ylim(0,600)
    ax2.set_xlim(min(T10[0:np.where(np.round(T10[:,0])==600e3)[0][-1],2])-10, \
                max(T10[0:np.where(np.round(T10[:,0])==600e3)[0][-1],2])+10)
#    ax2.set_xlim(3250,max(Toc1[:,2]))
    ax2.invert_yaxis()
#    ax2.legend(loc='lower left',fontsize=16)
    
#    ax2.text(3790, 225, r'200km',fontsize=19, color='b')
#    ax2.text(3790, 175, r'150km',fontsize=19, color='g')
#    ax2.text(3790, 145, r'120km',fontsize=19, color='r')
#    ax2.text(3790, 108, r'110km',fontsize=19, color='m')
#    ax2.text(3790, 68, r'70km',fontsize=19, color='xkcd:sienna')
#    ax2.text(3720, 25, r'LAB depth',fontsize=19, color='k')
    
    ax1.legend(title="$/bigtriangleup /rho_{LAB} (kg/m^3)$",loc='lower left', fontsize=15)
    ax1.get_legend().get_title().set_fontsize('15')
    
    ax2.text(3680,138, '120 km - LAB',fontsize=15, color='k')
#    ax2.text(3680,218, '200 km - LAB',fontsize=15, color='k')
    ax2.text(3680,58, '40 km - MOHO',fontsize=15, color='k')
    
    
    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='major', length=12,width=1)
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax2.tick_params(which='major', length=10,width=1)
    ax2.tick_params(which='minor', length=6)
    
    ax1.xaxis.set_major_locator(xmajorLocator1)
    ax1.xaxis.set_minor_locator(xminorLocator1)
    ax1.yaxis.set_major_locator(ymajorLocator1)
    ax1.yaxis.set_minor_locator(yminorLocator1)
    ax2.xaxis.set_minor_locator(xminorLocator2)
  
    
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    plt.setp(ax1.spines.values(), color='k', linewidth=1.5)
    plt.setp(ax2.spines.values(), color='k', linewidth=1.5)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=15)
    ax2.yaxis.set_tick_params(labelsize=15)
#    ax1.legend(loc='lower left', fontsize=15)
    ax2.set_title('slab thickness 80 km', fontsize=16, fontweight='bold')   
#    fig8.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig8.subplots_adjust(wspace=0.05, hspace=0.15)
    ax1.grid(False)
    ax2.grid(False)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig8.savefig('InitTrho_DelRho_slab80.png', format='png', dpi=300)
    os.chdir('..')

#==============================================================================
if (buoyancy_plot >0.):
    # tmyr_array,  short_array,  y_plot,  Tslice2,  rho_slice2
#    count = 0
    plt.rc('axes', linewidth=2)
    df=pd.read_csv('Proton_review_vel_30.csv')
    shorten = df['shorten'].values
    time = df['time'].values
    Ftot = df['Ftot'].values
    aa=np.where(np.ceil(shorten)==400)[0][0]
    
    fig11=plt.figure(11,figsize=(15,10))
    fig11.clf()
    ax1 = fig11.add_subplot(111)
    
    # this is an inset axes over the main axes
#    inset_axes = inset_axes(ax1, 
#                    width="25%", # width = 30% of parent_bbox
#                    height=2.0, # height : 1 inch
#                    loc=10)
    
    
    diff=(df['Ftot'])-(df['Fsum'])
    adv = ((df['FadvA'])+(df['FadvB']))+diff
    ax1.plot(df['time'],df['Ftot']/1e12 , 'k', linewidth=3,label='$F_{b}$')
    ax1.plot(df['time'],adv/1e12, 'g', linewidth=3,label='$F_{a}$')
    ax1.plot(df['time'],df['Fdiffus']/1e12, 'r', linewidth=3,label='$F_{d}$')
#    ax1.set_title('Buoyancy Plot, Proton 30 mm/yr', fontsize=20)
    ax1.set_xlabel('Time ($Myr$)', fontsize=30)
    ax1.set_ylabel('Buoyancy force ($TNm^{-1}$)', fontsize=30)
    ax1.plot(df['time'],df['Ftot']*0,'k',alpha=0.4,label='_nolabel_')
    ax1.set_ylim(-12,12)
    ax1.set_xlim(0,time[aa])
#    ax1.grid(linestyle='dotted')
#    ax1.xaxis.grid() 

    ax2 = ax1.twiny()
    ax2.plot(df['shorten'],df['Ftot']/1e12,'k', alpha=0.0)
#    ax2.plot(range(400),np.zeros(400))
#    ax2.set_xlim(0,400)
    ax2.set_xlim(0,shorten[aa])
    ax2.set_xlabel('Shortening ($km$)', fontsize=30,labelpad=20)
    
    ax3 = plt.axes([0.17, 0.57, .25, .25])
    ax4 = ax3.twiny()
    ax3.plot(df['time'],df['Ftot']/1e12,'k')
    ax4.plot(df['shorten'],df['Ftot']/1e12,'k', alpha=0.0)
    ax3.set_xlim(0,max(df['time']))
    ax4.set_xlim(0,max(df['shorten']))
#    ax3.grid(linestyle='dotted')
#    ax3.xaxis.grid() 
    ax3.plot(df['time'],df['Ftot']*0,'k',alpha=0.4,label='_nolabel_')
#    ax3.set_xlabel('Time [$Myr$]', fontsize=15)
#    ax3.set_ylabel('F$_{buoy}$ [$10^{12} N/m$]', fontsize=15)
    ax3.xaxis.set_tick_params(labelsize=20)
    ax3.yaxis.set_tick_params(labelsize=20)
    ax4.xaxis.set_tick_params(labelsize=20)
    ax3.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                 bottom=True, top=False, left=True, right=True)
    ax3.tick_params(which='major', length=5,width=1)
    ax3.tick_params(which='minor', length=5)
    ax4.tick_params(direction='in',which='both',labelbottom=False, labeltop=True, labelleft=True, labelright=False,
                 bottom=False, top=True, left=True, right=True)
    ax4.tick_params(which='major', length=5,width=1)
    ax4.tick_params(which='minor', length=5)
    markers_on = [np.where(Ftot==min(Ftot))[0][0],
                  np.where(np.ceil(time)==17)[0][4],
                  np.where(Ftot==max(Ftot))[0][0] ]
    ax3.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax3.xaxis.set_minor_locator(MultipleLocator(5))
    ax4.xaxis.set_minor_locator(MultipleLocator(50))

    ax3.plot(df['time'],df['Ftot']/1e12, 'bD', markevery=markers_on)
    ax3.text(3.5, -1.5, '6.4 Myr', fontsize=20)
    ax3.text(9.5, 0.15, '17 Myr', fontsize=20)
    ax3.text(21, 0.1, '23.6 \n Myr', fontsize=20)
#    ax3.minorticks_on()
    
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax2.xaxis.set_tick_params(labelsize=30)
    
    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                 bottom=True, top=False, left=True, right=True)
    ax1.tick_params(which='major', length=14,width=2)
    ax1.tick_params(which='minor', length=7)
    ax1.minorticks_on()
    
    ax2.tick_params(direction='in',which='major', length=14,width=2)
    ax2.tick_params(direction='in',which='minor', length=7)
    ax2.minorticks_on()
    
    ax1.legend(title="Force Components",loc='lower left', fontsize=28)
    ax1.get_legend().get_title().set_fontsize('28')
    
    
#    plt.tight_layout()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig11.savefig('FbTimeShorten_manu.png', format='png', dpi=600)
    os.chdir('..')
    
if (Trho_evol >0.):
    # tmyr_array,  short_array,  y_plot,  Tslice2,  rho_slice2
#    count = 0
    dens_data = np.loadtxt("colormap/lajolla.txt")
    CBdens_map = LinearSegmentedColormap.from_list('CBdens', dens_data[::-1])
    df=np.loadtxt('review_Proton30_slices.txt', dtype=float)
    xx=np.int(np.shape(df)[0])
    yy=np.int(np.round(np.shape(df)[1]/5-2))
    age=np.zeros((xx,yy))
    shorten=np.zeros((xx,yy))
    yplot=np.zeros((xx,yy))
    Tslice=np.zeros((xx,yy))
    rhoslice=np.zeros((xx,yy))
    col=[]
    for i in range (1,round(np.shape(df)[1]/5)-1):
        age_ = df[:,(i*5)-4]
        shorten_ = df[:,(i*5)-3]
        yplot_ = df[:,(i*5)-2]
        Tslice_ = df[:,(i*5)-1]
        rhoslice_ = df[:,i*5]
        
        age[:,i-1] = np.ceil(age_) 
        shorten[:,i-1]=shorten_
        yplot[:,i-1]=yplot_
        Tslice[:,i-1]=Tslice_
        rhoslice[:,i-1]=rhoslice_
        
        age[:,0] = age_*0        
        
        init_T = df[:,4]    
        init_rho = df[:,5]   
        
    hline1=np.linspace(min(Tslice[:,1]),max(Tslice[:,1]))
    hline2=np.linspace(min(rhoslice[:,1]),max(rhoslice[:,1]))
        
    s=[0.0,2.0,6.0,12.0,16.0,21.0]
#    if any(s in np.round(age[1],-1) for s in [2.0,5.0,10.0,15.0,20.0]):
#        s=np.round(age[1],-2)
    for i in range (0,np.size(s)):   
        col.append(max(np.where(age==s[i])[1]))
#        col.append((np.where(age==s[i])[1]))
        
        

    fig99.clf() 
    for i in range (0,np.size(col)):
        fig99 = plt.figure(13,figsize=(15,10))
        ax1=plt.subplot(121)
        ax2=plt.subplot(122, sharey=ax1)
    #    plt.plot(df['Tslice2'],df['y_plot'])
        no_points = len(col)
        ax1.plot(Tslice[:,col[i]],yplot[:,col[i]],color=plt.cm.inferno(1.*i/(no_points)),\
                label=(str(np.int(age[col[0]][col[i]])*30))+' km')
#        ax1.plot(Tslice[:,col[i]],yplot[:,col[i]],color=plt.cm.copper(1.*i/(no_points-1)),\
#                label=(str(np.int(age[col[0]][col[i]])*30))+' km')
        ax1.plot(hline1,hline1*0+110e3,color = 'k',linestyle='--', linewidth=1,alpha=0.5)
    #    plt.gca().invert_yaxis()
        
        ax1.legend(title="Shortening",loc='lower left', fontsize=23,fancybox=True, framealpha=0.3)
        ax1.get_legend().get_title().set_fontsize('23')
        ax1.grid(linestyle='dotted')
#        ax1.set_title('Temperature-Depth profile', fontsize=25)
        ax1.set_xlabel('Temperature ($^{\circ}C$)', fontsize=25)
        ax1.text(1190,-10e3,'$T$ - profile',fontsize=30)
        ax1.set_ylabel('Depth ($km$)', fontsize=25)
        ax1.set_ylim(600e3,0)
        ax1.set_xlim(500,max(Tslice[:,1]))
        ax1.invert_yaxis()
        ytick_prof =np.arange(40e3/1000, 650e3 /1000, 100)
        ax1.set_yticklabels(ytick_prof)            
        ax1.xaxis.set_ticks(np.arange(min(Tslice[:,1]), max(Tslice[:,1]), 250))
        ax1.tick_params(labelsize=22)
        ax1.tick_params(axis='x', pad=12)
        ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                 bottom=True, top=True, left=True, right=True)
        ax1.tick_params(which='major', length=14)
        ax1.tick_params(which='minor', length=7)
        ax1.minorticks_on()
        
        ax2.plot(rhoslice[:,col[i]],yplot[:,col[i]],color=plt.cm.inferno(1.*i/(no_points)) ,\
                label=(str(np.int(age[col[0]][col[i]])))+' Myr')
        ax2.plot(hline2,hline2*0+110e3,color = 'k',linestyle='--', linewidth=1,alpha=0.5)
    #    plt.gca().invert_yaxis()
        ax2.invert_yaxis()
        ax2.text(3730,-10e3,'$/rho$ - profile',fontsize=30)
        ax2.legend(title="Time",loc='lower left', fontsize=23,fancybox=True, framealpha=0.3)
        ax2.get_legend().get_title().set_fontsize('23')
        ax2.grid(linestyle='dotted')
#        ax2.set_title('Density-Depth profile', fontsize=25)
        ax2.set_xlabel('Density ($kg$ $m^{-3}$)',fontweight='normal',fontsize=25)
#        ax2.set_ylabel('Depth [$km$]', fontsize=25)
#            ax2.set_ylim(600e3,0)
#            ytick_prof = np.arange(40e3/1000, 1500e3 /1000, 100)
#            ax2.set_yticklabels(ytick_prof)
        ax2.set_xlim(min(rhoslice[:,1]),max(rhoslice[:,1]))
        ax2.xaxis.set_ticks(np.arange(min(rhoslice[:,1]), max(rhoslice[:,1]), 200))
        ax2.tick_params(labelsize=22)
        ax2.tick_params(axis='x', pad=12)
        ax2.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                 bottom=True, top=True, left=True, right=True)
        ax2.tick_params(which='major', length=14)
        ax2.tick_params(which='minor', length=7)
        ax2.minorticks_on()
        
        ax1.plot(init_T,yplot[:,1],'--k',linewidth=1)
        ax2.plot(init_rho,yplot[:,1],'--k',linewidth=1)
        plt.pause(0.0005)
        plt.show()
#        count=count+1
    fig99.subplots_adjust(wspace=0.25, hspace=0.15)
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig99.savefig('Trho_evolution_Proton30.png', format='png', dpi=600)
    os.chdir('..')

if (FbouyShorten>0.):
    fig1 = plt.figure(figsize=(15,10))
    fig1.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    
    for i in range(len(vel)):  
#        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['shorten'], \
#                  globals()['dfa{}'.format(i+1)]['Ftot']/1e12, 'b',label="Archon", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['Ftot']/1e12, 'g',label="Proton", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['Ftot']/1e12, 'r',label="Tecton", linewidth=3)
#        globals()['ax{}'.format(i+1)].plot(globals()['dfoc1{}'.format(i+1)]['shorten'], \
#                  globals()['dfoc1{}'.format(i+1)]['Ftot']/1e12, 'c',label="OC30ma", linewidth=3)
#        globals()['ax{}'.format(i+1)].plot(globals()['dfoc2{}'.format(i+1)]['shorten'], \
#                  globals()['dfoc2{}'.format(i+1)]['Ftot']/1e12, 'm',label="OC120ma", linewidth=3)
#        globals()['ax{}'.format(i+1)].plot(globals()['dfoc2{}'.format(i+1)]['shorten'], \
#                  globals()['dfoc2{}'.format(i+1)]['Ftot']*0,'--k', label='_nolegend_', linewidth=3)
        
        globals()['ax{}'.format(i+1)].set_xlim(0,600)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
                  verticalalignment='top', bbox=props)
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Shortening (km)',fontsize=20, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12} N/m$)',fontsize=20, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
            
    ax4.legend(loc='best',fontsize=22)      
    fig1.suptitle('F$_{bouy}$ vs Shortening', fontsize=20, fontweight='bold')   
    fig1.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig1.savefig('FbShorten_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')

if(FbouyTime>0.):

    fig2 = plt.figure(figsize=(15,10))
    fig2.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1,  sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2,  sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1,  sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2,  sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2,  sharey=ax4)

    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['time'], \
                  globals()['dfa{}'.format(i+1)]['Ftot']/1e12, 'b',label="Archon", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['time'], \
                  globals()['dfp{}'.format(i+1)]['Ftot']/1e12, 'g',label="Proton", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['Ftot']/1e12, 'r',label="Tecton", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfoc1{}'.format(i+1)]['time'], \
                  globals()['dfoc1{}'.format(i+1)]['Ftot']/1e12, 'c',label="OC30ma", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfoc2{}'.format(i+1)]['time'], \
                  globals()['dfoc2{}'.format(i+1)]['Ftot']/1e12, 'm',label="OC120ma", linewidth=3)

        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  (globals()['dft{}'.format(i+1)]['time']*0)-3,'-k', label='_nolegend_',alpha=0.5)
        globals()['ax{}'.format(i+1)].set_xlim(0,100)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
#        if i==1 or i==len(vel)-1:
#            globals()['ax{}'.format(i+1)].set_xlabel('Time (Myr)',fontsize=16, fontweight='bold')
#            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
#        else:
#            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
#        if i<=1:
#            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
#        else:
#            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
      
    ax1.set_xlim(0,400)
    ax2.set_xlim(0,100)
    ax3.set_xlim(0,40)
    ax4.set_xlim(0,20)
    ax5.set_xlim(0,10)
    ax6.set_xlim(0,5)
    
    fig2.suptitle('F$_{bouy}$ vs Time', fontsize=16, fontweight='bold')   
    fig2.tight_layout(rect=[0, 0.03, 1, 0.97])
#    fig2.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig2.savefig('FbTime_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')
    

if(FbouyVels>0.):
    fig3 = plt.figure(3,figsize=(10,6))
    fig3.clf()
    set_style2()
#    with sns.axes_style("whitegrid"):
    ax5 = fig3.add_subplot(111)
    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    minFa=[]
    minFp=[]
    minFt=[]
    minFoc1=[]
    minFoc2=[]
    
    lim=400
    
    for i in range(len(vel)):
        limit=np.where(globals()['dfa{}'.format(i+1)]['shorten']<lim)[0][-1]
        globals()['minF_a{}'.format(i+1)]=min(globals()['dfa{}'.format(i+1)]['Ftot'][0:limit])*0/1e12
        globals()['minF_p{}'.format(i+1)]=min(globals()['dfp{}'.format(i+1)]['Ftot'][0:limit])/1e12
        globals()['minF_t{}'.format(i+1)]=min(globals()['dft{}'.format(i+1)]['Ftot'][0:limit])/1e12
        globals()['minF_oc1{}'.format(i+1)]=min(globals()['dfoc1{}'.format(i+1)]['Ftot'][0:limit])/1e12
        globals()['minF_oc2{}'.format(i+1)]=min(globals()['dfoc2{}'.format(i+1)]['Ftot'][0:limit])/1e12
        minFa.append(globals()['minF_a{}'.format(i+1)])
        minFp.append(globals()['minF_p{}'.format(i+1)])
        minFt.append(globals()['minF_t{}'.format(i+1)])
        minFoc1.append(globals()['minF_oc1{}'.format(i+1)])
        minFoc2.append(globals()['minF_oc2{}'.format(i+1)])
        
    vel = [int(i) for i in vel]
    fitdeg = 5
    z_a = np.polyfit(vel,minFa, fitdeg)
    z_p = np.polyfit(vel,minFp, fitdeg)
    z_t = np.polyfit(vel,minFt, fitdeg)
    z_oc1 = np.polyfit(vel,minFoc1, fitdeg)
    z_oc2 = np.polyfit(vel,minFoc2, fitdeg)
    p_a = np.poly1d(z_a)
    p_p = np.poly1d(z_p)
    p_t = np.poly1d(z_t)
    p_oc1 = np.poly1d(z_oc1)
    p_oc2 = np.poly1d(z_oc2)

    xp = np.linspace(0, 80, 100)

    ax5.plot(xp, p_a(xp), 'b-',linewidth=3,label='Archon')
    ax5.plot(xp, p_p(xp), 'g-',linewidth=3,label='Proton')
    ax5.plot(xp, p_t(xp), 'r-',linewidth=3,label='Tecton')
    ax5.plot(xp, p_oc1(xp), 'c-.',linewidth=3,label='OC30ma')
    ax5.plot(xp, p_oc2(xp), 'm--',linewidth=3,label='OC120ma')
    ax5.plot(vel,minFa, 'b.',markersize=20)
    ax5.plot(vel,minFp, 'g.',markersize=20)
    ax5.plot(vel,minFt, 'r.',markersize=20)
    ax5.plot(vel,minFoc1, 'c.',markersize=20)
    ax5.plot(vel,minFoc2, 'm.',markersize=20)
    ax5.plot(xp, p_a(xp)*0,color="0.5", label='_nolegend_', linewidth=1)  
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax5.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax5.set_xlim(0,max(vel))
    ax5.legend(loc='center right', frameon=True,  fontsize=15)
    ax5.set_xlabel('Convergence rate $ /nu $ (mm/year)',fontsize=18,fontweight='bold')
    ax5.set_ylabel('F$_{bouy}$ ($10^{12}N/m$)',fontsize=18,fontweight='bold')
    ax5.tick_params(direction='in',which='minor', length=7, color='k') 
    plt.setp(ax5.spines.values(), color='k', linewidth=1.5)
    ax5.set_title('Maximum Slab-pull (within '+str(lim)+'km shortening)', fontsize=15, fontweight='bold') 
#    ax5.text(0.02, 0.02, '(e)',color='k', transform=ax5.transAxes, fontsize=18, \
#                  fontweight='bold',verticalalignment='bottom')
#    fig3.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.tight_layout()
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig3.savefig('FbVels_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')


if(Archon_FbouyTime>0.):
    fig4 = plt.figure(figsize=(15,10))
    fig4.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    minorLocator = AutoMinorLocator()
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['time'], \
                  globals()['dfa{}'.format(i+1)]['Ftot']/1e12, label="F$_{tot}$", linewidth=3,color='k')
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['time'], \
                  globals()['dfa{}'.format(i+1)]['FadvA']/1e12, label="F$_{adv}$^LAB", linewidth=3,color='b')
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['time'], \
                  globals()['dfa{}'.format(i+1)]['FadvB']/1e12, label="F$_{adv}$vLAB", linewidth=3,color='g')
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['time'], \
                  globals()['dfa{}'.format(i+1)]['Fdiffus']/1e12 ,'--m', label="F$_{diffus}$vLAB", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['time'], \
                  globals()['dfa{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')
        globals()['ax{}'.format(i+1)].set_xlim(0,100)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Time (Myr)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    
    fig4.suptitle('Archon: F$_{bouy}$ vs Time', fontsize=16, fontweight='bold')   
    fig4.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig4.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig4.savefig('ArchonFbTime_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')


if(Proton_FbouyTime>0.):
    fig5 = plt.figure(figsize=(15,10))
    fig5.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    minorLocator = AutoMinorLocator()
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['time'], \
                  globals()['dfp{}'.format(i+1)]['Ftot']/1e12, label="F$_{tot}$", linewidth=3,color='k')
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['time'], \
                  globals()['dfp{}'.format(i+1)]['FadvA']/1e12, label="F$_{adv}$^LAB", linewidth=3,color='b')
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['time'], \
                  globals()['dfp{}'.format(i+1)]['FadvB']/1e12, label="F$_{adv}$vLAB", linewidth=3,color='g')
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['time'], \
                  globals()['dfp{}'.format(i+1)]['Fdiffus']/1e12 ,'--m', label="F$_{diffus}$vLAB", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['time'], \
                  globals()['dfp{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')
        globals()['ax{}'.format(i+1)].set_xlim(0,100)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Time (Myr)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    plt.setp(ax1.spines.values(), linewidth=5)
    fig5.suptitle('Proton: F$_{bouy}$ vs Time', fontsize=16, fontweight='bold')   
    fig5.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig5.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig5.savefig('ProtonFbTime_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')
    
if(Tecton_FbouyTime>0.):
    fig6 = plt.figure(6,figsize=(15,10))
    fig6.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    minorLocator = AutoMinorLocator()
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['Ftot']/1e12, label="F$_{tot}$", linewidth=3,color='k')
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['FadvA']/1e12, label="F$_{adv}$^LAB", linewidth=3,color='b')
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['FadvB']/1e12, label="F$_{adv}$vLAB", linewidth=3,color='g')
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['Fdiffus']/1e12 ,'--m', label="F$_{diffus}$vLAB", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['time'], \
                  globals()['dft{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')
        globals()['ax{}'.format(i+1)].set_xlim(0,100)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Time (Myr)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    
    fig6.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig6.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig6.savefig('TectonFbTime_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')


if(Tecton_FbouyShorten>0.):
    thick=['60km','80km (default)','110km']
    tec1=pd.read_csv('Tecton_lab100_vel_30.csv')
    tec2=pd.read_csv('Tecton_lab120_vel_30.csv')
    tec3=pd.read_csv('Tecton_lab150_vel_30.csv')
    shorten = tec1['shorten'].values
    time = tec1['time'].values
    aa=np.where(np.round(shorten)==400)[0][-1]
    
    fig6 = plt.figure(6,figsize=(14,5))
    fig6.clf()
    
#    set_style()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)#, sharey=ax1)
    ax3 = ax1.twiny()
    ax4 = ax2.twiny()
#    ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1, sharex=ax1)
#    ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax1)

#    minorLocator = AutoMinorLocator()
    xminorLocator1   = MultipleLocator(10)
    xminorLocator2   = MultipleLocator(10)
    yminorLocator1   = MultipleLocator(1)
    yminorLocator2   = MultipleLocator(1)
    xminorLocator3   = MultipleLocator(1)
    
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    diff1=(tec1['Ftot'])-(tec1['Fsum'])
    adv1 = ((tec1['FadvA'])+(tec1['FadvB']))+diff1
    
    diff2=(tec2['Ftot'])-(tec2['Fsum'])
    adv2 = ((tec2['FadvA'])+(tec2['FadvB']))+diff2
    
    diff3=(tec3['Ftot'])-(tec3['Fsum'])
    adv3 = ((tec3['FadvA'])+(tec3['FadvB']))+diff3
    Tecthick=[60e3,80e3,110e3]    

    mark=['*','v','o','.']
    for i in range(3):     
        ax1.plot(globals()['tec{}'.format(i+1)]['shorten'],(globals()['tec{}'.format(i+1)]['Ftot']/1e12), \
                 color=plt.cm.copper_r(i*90), label=thick[i], linewidth=2)
   
#        ax2.plot(tec1['shorten'],tec1['FadvA']/1e12 , \
#                 mark[i], color='b',label=thick[i]+'km', linewidth=2,markersize=8,markevery=20)
#    ax3.plot(tec1['time'],tec1['Ftot'], alpha=0.0)
    ax3.set_xlim(0,13)
#    ax4.plot(tec1['time'],tec1['Ftot'], alpha=0.0)
    ax4.set_xlim(0,13)
#    ax3x = ax1.get_xticks()
#    ax3.set_xticks([0,100,200,300,400,500,600])       
#    ax3.set_xticks(ax3x)
#    ax3.set_xlim(0,tec1['shorten'][aa])
    
    
#    ax2.plot(tec1['shorten'],tec1['FadvA']/1e12 , \
#             'b-*',label='_nolegend_', linewidth=2,markersize=6,markevery=60)
#    ax2.plot(tec2['shorten'],tec2['FadvA']/1e12 , \
#             'b-v',label='_nolegend_', linewidth=2,markersize=6,markevery=60)
#    ax2.plot(tec3['shorten'],tec3['FadvA']/1e12 , \
#             'b-',label='Advection above LAB', linewidth=2,markersize=6,markevery=60)
#    ax2.plot(tec3['shorten'],tec3['FadvA']/1e12 , \
#             'bo',label='_nolegend_', linewidth=2,markersize=6,markevery=60)
    
    ax2.plot(tec1['shorten'],adv1/1e12, color=plt.cm.copper_r(0*90),\
             marker='v', linewidth=2,markersize=7,markevery=50)
    ax2.plot(tec2['shorten'],adv2/1e12,color=plt.cm.copper_r(1*90), \
             marker='v', linewidth=2,markersize=7,markevery=50)
    ax2.plot(tec3['shorten'],adv3/1e12, color=plt.cm.copper_r(2*90),\
             marker='v', linewidth=2,markersize=6,markevery=50)
#    ax2.plot(tec3['shorten'],adv3/1e12,color=plt.cm.copper_r(3*90), \
#             'go', linewidth=2,markersize=7,markevery=5)
    
    ax2.plot(tec1['shorten'],tec1['Fdiffus']/1e12, color=plt.cm.copper_r(0*90),\
             label='_nolegend_',marker='o', linewidth=2,markersize=7,markevery=50)
    ax2.plot(tec2['shorten'],tec2['Fdiffus']/1e12,color=plt.cm.copper_r(1*90), \
             label='_nolegend_',marker='o', linewidth=2,markersize=7,markevery=50)
    ax2.plot(tec3['shorten'],tec3['Fdiffus']/1e12, color=plt.cm.copper_r(2*90),\
             marker='o', linewidth=2,markersize=6,markevery=50)
    
#    ax2.plot(tec3['shorten'],tec3['Fdiffus']/1e12, \
#             'ro', label='_nolegend_',linewidth=2,markersize=7,markevery=5)

#    ax2.text(0.03, 0.33, r'$\bigstar$ 60km',transform=ax2.transAxes, fontsize=14, \
#              fontweight='normal',verticalalignment='top')
#    ax2.text(0.03, 0.27, r'$ \blacktriangledown$  80km (default)',transform=ax2.transAxes, fontsize=14, \
#              fontweight='normal',verticalalignment='top')
#    ax2.text(0.03, 0.23, r'$\bullet$',transform=ax2.transAxes, fontsize=22, \
#              fontweight='normal',verticalalignment='top')
#    ax2.text(0.05, 0.2, '  110km',transform=ax2.transAxes, fontsize=14, \
#              fontweight='normal',verticalalignment='top')
    
    ax1.plot(np.linspace(0,600,len(tec1)), \
              tec1['shorten']*0,color="0.5", label='_nolegend_', linewidth=1)
    ax2.plot(np.linspace(0,600,len(tec1)), \
              tec1['shorten']*0,color="0.5", label='_nolegend_', linewidth=1)
    ax1.set_xlim(0,400)
    ax2.set_xlim(0,400)
    
    lab_size=17
    ax2.xaxis.set_tick_params(labelsize=lab_size)
    ax1.xaxis.set_tick_params(labelsize=lab_size)
    ax1.yaxis.set_tick_params(labelsize=lab_size)
    ax2.yaxis.set_tick_params(labelsize=lab_size)
    ax3.xaxis.set_tick_params(labelsize=lab_size)
    ax4.xaxis.set_tick_params(labelsize=lab_size)
    
    ax1.xaxis.set_minor_locator(xminorLocator1)
    ax1.yaxis.set_minor_locator(yminorLocator1)
    ax2.xaxis.set_minor_locator(xminorLocator2)
    ax2.yaxis.set_minor_locator(yminorLocator2)
    ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax3.xaxis.set_major_locator(MultipleLocator(2))
    ax4.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax4.xaxis.set_major_locator(MultipleLocator(2))
    
    ax1.text(0.9, 0.12, '(a)',color='k', \
             transform=ax1.transAxes, fontsize=25, \
             fontweight='bold',verticalalignment='top')
    ax2.text(0.02, 0.12, '(b)',color='k', \
             transform=ax2.transAxes, fontsize=25, \
             fontweight='bold',verticalalignment='top')
    
    ax1.set_ylim(-10,0)
    ax2.set_ylim(-15,10)
#    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax1.spines.values(), color='k', linewidth=1.5)
    plt.setp(ax2.spines.values(), color='k', linewidth=1.5)
    ax1.set_xlabel('Shortening ($km$)',fontsize=20, fontweight='bold')
    ax2.set_xlabel('Shortening ($km$)',fontsize=20, fontweight='bold')
    ax1.set_ylabel('F$_{b}$ ($TNm^{-1}$)',fontsize=20, fontweight='bold')
    ax2.set_ylabel('Buoyancy Force ($TNm^{-1}$)',fontsize=20, fontweight='bold') 
    ax3.set_xlabel(r"Time ($Myr$)",fontsize=20, fontweight='bold')
    ax4.set_xlabel(r"Time ($Myr$)",fontsize=20, fontweight='bold')
    
    ax1.legend(title='Slab thickness',fontsize=16, fancybox=True, loc='lower left')#, bbox_to_anchor=(0.55,0.07))
    
    ax1.get_legend().get_title().set_fontsize('16')
    mark_diffus = mlines.Line2D([], [], color='black', marker='o',linestyle = 'None',
                          markersize=13, label='Diffusion')
    mark_adv = mlines.Line2D([], [], color='black', marker='v',linestyle = 'None',
                          markersize=13, label='Advection')
    ax2.legend(handles=[mark_diffus, mark_adv],fontsize=17)
#    ax2.legend(loc='upper left',frameon=True,fontsize=17)
    
    
    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=False, left=True, right=True)
    ax1.tick_params(which='major', length=8,width=1)
    ax1.tick_params(which='minor', length=4)
    ax2.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=False, left=True, right=True)
    ax2.tick_params(which='major', length=8,width=1)
    ax2.tick_params(which='minor', length=4)
    ax3.tick_params(direction='in',which='both',labelbottom=False, labeltop=True, labelleft=False, labelright=False,
             bottom=False, top=True, left=False, right=False)
    ax3.tick_params(which='major', length=8,width=1)
    ax3.tick_params(which='minor', length=4)
    ax4.tick_params(direction='in',which='both',labelbottom=False, labeltop=True, labelleft=True, labelright=False,
             bottom=False, top=True, left=True, right=True)
    ax4.tick_params(which='major', length=8,width=1)
    ax4.tick_params(which='minor', length=4)
    
    ax1.grid(linestyle='dotted')
    ax1.xaxis.grid()
    ax2.grid(linestyle='dotted')
    ax2.xaxis.grid()
#    ax1.text(15,0.1, 'Tecton, v=30mm/yr', fontsize=20)
    
    
#    st = fig6.suptitle("Effect of lithospheric thickness",y=1.05, fontsize=15)
#    ax1.set_title('Effect of lithospheric thickness', fontsize=20, fontweight='bold', loc='left',y=1.17)
#    ax1.set_title('Tecton, 'r'$ v=30mm/yr$', fontsize=20, fontweight='normal', loc='right',y=1.17)
#    ax2.set_title('Tecton, 'r'$ v=30mm/yr$', fontsize=20, fontweight='normal', loc='right',y=1.17)
    
#    fig6.tight_layout()
#    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig6.savefig('TectonFbShorten_'+exp_name+'_vels.png', format='png', dpi=600)
    os.chdir('..')

if(Effect_advA>0.):
    fig9 = plt.figure(figsize=(15,10))
    fig9.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['FadvA']/1e12,'--b', label="advA_Arc", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['FadvA']/1e12,'+b', label="advA_Pro", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['FadvA']/1e12,'*b', label="advA_Tec", linewidth=3)
        
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')   
        globals()['ax{}'.format(i+1)].set_xlim(0,900)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Shortening (km)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    fig9.suptitle('Advection above LAB', fontsize=16, fontweight='bold')   
    fig9.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig9.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig9.savefig('AdvA_'+exp_name+'_vels.png', format='png', dpi=600)
    os.chdir('..')

if(Effect_advB>0.):
    fig10 = plt.figure(figsize=(15,10))
    fig10.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['FadvB']/1e12,'--g', label="advB_Arc", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['FadvB']/1e12,'+g', label="advB_Pro", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['FadvB']/1e12,'*g', label="advB_Tec", linewidth=3)
        
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')   
        globals()['ax{}'.format(i+1)].set_xlim(0,900)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Shortening (km)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    fig10.suptitle('Advection below LAB', fontsize=16, fontweight='bold')   
    fig10.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig10.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig10.savefig('AdvB_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')

#if(Effect_diffusA>0.):
#    fig11 = plt.figure(figsize=(15,10))
#    fig11.clf()
#    
#    if(len(vel)==4):
#        with sns.axes_style("darkgrid"):
#            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
#            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
#            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
#            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
#    if(len(vel)==6):
#        with sns.axes_style("darkgrid"):
#            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
#            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
#            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
#            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
#            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
#            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
#
#    minorLocator = AutoMinorLocator()
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    
#    for i in range(len(vel)):
#        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['shorten'], \
#                  globals()['dfa{}'.format(i+1)]['AA']/1e12,'--r', label="diffusA_Arc", linewidth=3)
#        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
#                  globals()['dfp{}'.format(i+1)]['FdiffusA']/1e12,'+r', label="diffusA_Pro", linewidth=3)
#        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
#                  globals()['dft{}'.format(i+1)]['FdiffusA']/1e12,'*r', label="diffusA_Tec", linewidth=3)
#        
#        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
#                  globals()['dft{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')   
#        globals()['ax{}'.format(i+1)].set_xlim(0,900)
#        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
#        sns.despine(right=True,top=True)
#        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
#        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
#                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
#                  verticalalignment='top', bbox=props)
#         
#    
#        if i==1 or i==len(vel)-1:
#            globals()['ax{}'.format(i+1)].set_xlabel('Shortening (km)',fontsize=16, fontweight='bold')
#            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
#        else:
#            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
#        if i<=1:
#            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
#        else:
#            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
#      
#    fig11.suptitle('Diffusion above LAB', fontsize=16, fontweight='bold')   
#    fig11.tight_layout(rect=[0, 0.03, 1, 0.97])
#    fig11.subplots_adjust(wspace=0.05, hspace=0.15)
#    plt.show()
#    print("Images in directory %s" % dir)
#    os.chdir(dir)
#    fig11.savefig('DiffusA_'+exp_name+'_vels.png', format='png', dpi=300)
#    os.chdir('..')
    

if(Effect_diffus>0.):
    fig12 = plt.figure(figsize=(15,10))
    fig12.clf()
    
    if(len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if(len(vel)==6):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)

    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    
    for i in range(len(vel)):
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['Fdiffus']/1e12,'--m', label="diffus_Arc", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['Fdiffus']/1e12,'+m', label="diffus_Pro", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['Fdiffus']/1e12,'*m', label="diffus_Tec", linewidth=3)
        
        
        globals()['ax{}'.format(i+1)].plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['time']*0,'--k', label='_nolegend_')  
        
    #    globals()['ax{}'.format(i+1)].axvline(x=bottma,color='k', linestyle='--',linewidth=2)
    #    globals()['ax{}'.format(i+1)].axvline(x=bottmp,color='r', linestyle='--',linewidth=2)
    #    globals()['ax{}'.format(i+1)].axvline(x=bottmt,color='m', linestyle='--',linewidth=2)
        
        globals()['ax{}'.format(i+1)].set_xlim(0,900)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Shortening (km)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    fig12.suptitle('Diffusion below LAB', fontsize=16, fontweight='bold')   
    fig12.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig12.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig12.savefig('Diffus_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')

#==============================================================================
if(FtotShorten>0.):
#    plt.close(fig13)
    fig13 = plt.figure(13,figsize=(14,8))
    fig13.clf()
#    set_style()
#    with sns.set_style("white", {
#        "font.family": "Times"}):
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
#    ax5 = fig13.add_subplot(313)  
#    ax1 = plt.subplot2grid((11, 6), (0, 0), colspan=3, rowspan=3)
#    ax2 = plt.subplot2grid((11, 6), (0, 3), colspan=3, rowspan=3)
#    ax3 = plt.subplot2grid((11, 6), (3, 0), colspan=3, rowspan=3)
#    ax4 = plt.subplot2grid((11, 6), (3, 3), colspan=3, rowspan=3)
#    ax5 = plt.subplot2grid((11, 6), (7, 1), colspan=4, rowspan=4)
    
    color=['k','k','k','k','k']
    mtype=['Archon','Proton', 'Tecton','Oceanic']
    abc=['(a)','(b)','(c)','(d)']
    lim=400
    

    #props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#    lines1 = []
#    lines2=[]


    for i in range(len(vel)):
        globals()['fp{}'.format(i+1)] = interpolate.interp1d(globals()['dfp{}'.format(i+1)]['shorten'], \
                                  globals()['dfp{}'.format(i+1)]['Ftot'], kind='cubic')
        globals()['xp{}'.format(i+1)] = np.linspace(np.min(globals()['dfp{}'.format(i+1)]['shorten']), \
                  np.max(globals()['dfp{}'.format(i+1)]['shorten']),100)
        globals()['sp{}'.format(i+1)] = splrep(globals()['dfp{}'.format(i+1)]['shorten'], globals()['dfp{}'.format(i+1)]['Ftot'])
        globals()['yp{}'.format(i+1)] = splev(globals()['xp{}'.format(i+1)], globals()['sp{}'.format(i+1)])
        
#        globals()['ft{}'.format(i+1)] = interpolate.interp1d(globals()['dft{}'.format(i+1)]['shorten'], \
#                                  globals()['dft{}'.format(i+1)]['Ftot'], kind='cubic')
#        globals()['xt{}'.format(i+1)] = np.linspace(np.min(globals()['dft{}'.format(i+1)]['shorten']), \
#                  np.max(globals()['dft{}'.format(i+1)]['shorten']),100)
#        globals()['st{}'.format(i+1)] = splrep(globals()['dft{}'.format(i+1)]['shorten'], globals()['dft{}'.format(i+1)]['Ftot'])
#        globals()['yt{}'.format(i+1)] = splev(globals()['xt{}'.format(i+1)], globals()['st{}'.format(i+1)])
        
#        globals()['fa{}'.format(i+1)] = interpolate.interp1d(globals()['dfa{}'.format(i+1)]['shorten'], \
#                                  globals()['dfa{}'.format(i+1)]['Ftot'], kind='cubic')
#        globals()['xa{}'.format(i+1)] = np.linspace(np.min(globals()['dfa{}'.format(i+1)]['shorten']), \
#                  np.max(globals()['dfa{}'.format(i+1)]['shorten']),100)
#        globals()['sa{}'.format(i+1)] = splrep(globals()['dfa{}'.format(i+1)]['shorten'], globals()['dfa{}'.format(i+1)]['Ftot'])
#        globals()['ya{}'.format(i+1)] = splev(globals()['xa{}'.format(i+1)], globals()['sa{}'.format(i+1)])
#        
#        globals()['foc1{}'.format(i+1)] = interpolate.interp1d(globals()['dfoc1{}'.format(i+1)]['shorten'], \
#                                  globals()['dfoc1{}'.format(i+1)]['Ftot'], kind='cubic')
#        globals()['xoc1{}'.format(i+1)] = np.linspace(np.min(globals()['dfoc1{}'.format(i+1)]['shorten']), \
#                  np.max(globals()['dfoc1{}'.format(i+1)]['shorten']),100)
#        globals()['soc1{}'.format(i+1)] = splrep(globals()['dfoc1{}'.format(i+1)]['shorten'], globals()['dfoc1{}'.format(i+1)]['Ftot'])
#        globals()['yoc1{}'.format(i+1)] = splev(globals()['xoc1{}'.format(i+1)], globals()['soc1{}'.format(i+1)])
#        
#        globals()['foc2{}'.format(i+1)] = interpolate.interp1d(globals()['dfoc2{}'.format(i+1)]['shorten'], \
#                                  globals()['dfoc2{}'.format(i+1)]['Ftot'], kind='cubic')
#        globals()['xoc2{}'.format(i+1)] = np.linspace(np.min(globals()['dfoc2{}'.format(i+1)]['shorten']), \
#                  np.max(globals()['dfoc2{}'.format(i+1)]['shorten']),100)
#        globals()['soc2{}'.format(i+1)] = splrep(globals()['dfoc2{}'.format(i+1)]['shorten'], globals()['dfoc2{}'.format(i+1)]['Ftot'])
#        globals()['yoc2{}'.format(i+1)] = splev(globals()['xoc2{}'.format(i+1)], globals()['soc2{}'.format(i+1)])


#    ax1.plot(xa1,ya1/1e12, color=plt.cm.copper_r(0*40), label=str(vel[0]), linewidth=2)
#    ax1.plot(xa2,ya2/1e12, color=plt.cm.copper_r(1*40), label=str(vel[1]), linewidth=2)
#    ax1.plot(xa3,ya3/1e12, color=plt.cm.copper_r(2*40), label=str(vel[2]), linewidth=2)
#    ax1.plot(xa4,ya4/1e12, color=plt.cm.copper_r(3*40), label=str(vel[3]), linewidth=2)
#    ax1.plot(xa5,ya5/1e12, color=plt.cm.copper_r(4*40), label=str(vel[4]), linewidth=2)
#    ax1.plot(xa6,ya6/1e12, color=plt.cm.copper_r(5*40), label=str(vel[5]), linewidth=2)

    for i in range(len(vel)):
#        xnew = np.linspace(globals()['xa{}'.format(i+1)].min(),globals()['xa{}'.format(i+1)].max(),2000)
#        spl = make_interp_spline(globals()['xa{}'.format(i+1)], globals()['ya{}'.format(i+1)]/1e12, k=5) 
#        power_smooth = spl(xnew)
#        ax1.plot(xnew,power_smooth, color=plt.cm.copper_r(i*40), label=str(vel[i]), linewidth=2)
#        ax1.plot(globals()['xa{}'.format(i+1)],globals()['ya{}'.format(i+1)]/1e12 , \
#                 color=plt.cm.copper_r(i*40), label=str(vel[i]), linewidth=2)
        ax2.plot(globals()['xp{}'.format(i+1)],globals()['yp{}'.format(i+1)]/1e12 , \
                 color=plt.cm.copper_r(i*40), label=str(vel[i]), linewidth=2)
#        ax3.plot(globals()['xt{}'.format(i+1)],globals()['yt{}'.format(i+1)]/1e12 , \
#                color=plt.cm.copper_r(i*40), label='_nolegend_', linewidth=2)
#        ax4.plot(globals()['xoc1{}'.format(i+1)],globals()['yoc1{}'.format(i+1)]/1e12 , \
#                 '--', color=plt.cm.copper_r(i*40), label='_nolegend_', linewidth=2)
#        ax4.plot(globals()['xoc2{}'.format(i+1)],globals()['yoc2{}'.format(i+1)]/1e12 , \
#                 color=plt.cm.copper_r(i*40), linewidth=2)
#    

                 
    for i in range(4):   
        globals()['ax{}'.format(i+1)].plot(np.linspace(0,600,len(globals()['dfp{}'.format(i+1)])), \
                  globals()['dfp{}'.format(i+1)]['time']*0,color="0.5", label='_nolegend_', linewidth=1)  
        globals()['ax{}'.format(i+1)].set_xlim(0,lim)
#        globals()['ax{}'.format(i+1)].set_ylim(-8,8)
#        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].text(0.02, 0.1, mtype[i],color=color[i], \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=22, \
                  fontweight='bold',verticalalignment='top')
        globals()['ax{}'.format(i+1)].text(0.02, 0.96, abc[i],color=color[i], \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=22, \
                  fontweight='bold',verticalalignment='top')
        plt.setp(globals()['ax{}'.format(i+1)].spines.values(), color='k', linewidth=1.5)
#        globals()['ax{}'.format(i+1)].minorticks_on()
        globals()['ax{}'.format(i+1)].xaxis.set_minor_locator(MultipleLocator(20))

    ax4.text(0.12, 0.95, 'OC30ma   - dash' , \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
                  fontweight='normal',verticalalignment='top')
    ax4.text(0.12, 0.87, 'OC120ma - solid' , \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
                  fontweight='normal',verticalalignment='top')
                  
  
                  
##    ax5.map(plt.axhline, y=0, ls=":", c=".5")
#    ax3.plot(np.linspace(0,600,62),dfa4['time']*0,'k', label='_nolegend_', linewidth=1)
##    ax4.legend(lines1[::], ['1 mm/yr', '10 mm/yr', '20 mm/yr', '40 mm/yr'],
##          loc='upper left', frameon=False ,fontsize=13)
##    leg = Legend(ax4, lines2[::], ['1 mm/yr', '10 mm/yr', '20 mm/yr', '40 mm/yr'],
##             loc='upper right', frameon=False ,fontsize=13)
##    ax4.add_artist(leg);
    lab_size=20
    ax1.xaxis.set_tick_params(labelsize=lab_size)
    ax2.xaxis.set_tick_params(labelsize=lab_size)
    ax3.xaxis.set_tick_params(labelsize=lab_size)
    ax4.xaxis.set_tick_params(labelsize=lab_size)
    ax1.yaxis.set_tick_params(labelsize=lab_size)
    ax2.yaxis.set_tick_params(labelsize=lab_size)
    ax3.yaxis.set_tick_params(labelsize=lab_size)
    ax4.yaxis.set_tick_params(labelsize=lab_size)
    
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
#    ax3.yaxis.set_major_locator(MultipleLocator(1))
    ax4.yaxis.set_minor_locator(MultipleLocator(1))
    
    
    ax1.tick_params(direction='in',which='both',labelbottom=False, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='major', length=12,width=1)
    ax1.tick_params(which='minor', length=5)
    ax2.tick_params(direction='in',which='both',labelbottom=False, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax2.tick_params(which='major', length=12,width=1)
    ax2.tick_params(which='minor', length=5)
    ax3.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax3.tick_params(which='major', length=12,width=1)
    ax3.tick_params(which='minor', length=5)
    ax4.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax4.tick_params(which='major', length=12,width=1)
    ax4.tick_params(which='minor', length=5)
    
    ax2.legend(title="v ($mm$$yr^{-1}$)",ncol=2,columnspacing=0.1,labelspacing=0.3,loc='lower right', fontsize=16,frameon=True)
    ax2.get_legend().get_title().set_fontsize('17')

    ax4.set_xlabel('Shortening ($km$)',fontsize=lab_size, fontweight='bold')
    ax3.set_xlabel('Shortening ($km$)',fontsize=lab_size, fontweight='bold')
    ax1.set_ylabel('F$_{b}$ ($T$ $Nm^{-1}$)',fontsize=lab_size, fontweight='bold')
    ax3.set_ylabel('F$_{b}$ ($T$ $Nm^{-1}$)',fontsize=lab_size, fontweight='bold')
    ax1.grid(linestyle='dotted')
    ax1.xaxis.grid()
    ax2.grid(linestyle='dotted')
    ax2.xaxis.grid()
    ax3.grid(linestyle='dotted')
    ax3.xaxis.grid()
    ax4.grid(linestyle='dotted')
    ax4.xaxis.grid()
    
    ax1.set_ylim(-2,8)
    ax2.set_ylim(-4,6)
    ax3.set_ylim(-6,1)
    ax4.set_ylim(-8,2)
    

#    plt.setp(ax2.get_yticklabels(), visible=False)
#    plt.setp(ax4.get_yticklabels(), visible=False)
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    plt.setp(ax2.get_xticklabels(), visible=False)
#    plt.setp(ax3.get_xticklabels(), visible=False)
#    plt.setp(ax4.get_xticklabels(), visible=False)
    ax1.set_title("Effect of convergence velocity", fontsize=20, fontweight='bold', loc='left',y=1.05)
    
    fig13.tight_layout()
#    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
#    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig13.savefig('FtotShort_'+exp_name+'_vels.png', format='png', dpi=600)
    os.chdir('..')
    

    
if (minFbuoy>0.):
#    os.chdir('..')
    lim=400    
    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    minFa=[]
    minFp=[]
    minFt=[]
    minFoc1=[]
    minFoc2=[]
    vel2=[1,4,10,20,30,40,50,60,70,80]
    for i in range(len(vel2)):
        globals()['dfa{}'.format(i+1)]=pd.read_csv('Archon_'+exp_name+'_vel_'+str(vel2[i])+'.csv')
        globals()['dfp{}'.format(i+1)]=pd.read_csv('Proton_'+exp_name+'_vel_'+str(vel2[i])+'.csv')
        globals()['dft{}'.format(i+1)]=pd.read_csv('Tecton_'+exp_name+'_vel_'+str(vel2[i])+'.csv')
        globals()['dfoc1{}'.format(i+1)]=pd.read_csv('Ocean30ma_'+exp_name+'_vel_'+str(vel2[i])+'.csv')
        globals()['dfoc2{}'.format(i+1)]=pd.read_csv('Ocean120ma_'+exp_name+'_vel_'+str(vel2[i])+'.csv')
    for i in range(len(vel2)):
        globals()['df_a{}'.format(i+1)]=(globals()['dfa{}'.format(i+1)]).values
        globals()['df_p{}'.format(i+1)]=(globals()['dfp{}'.format(i+1)]).values
        globals()['df_t{}'.format(i+1)]=(globals()['dft{}'.format(i+1)]).values
        globals()['df_oc1{}'.format(i+1)]=(globals()['dfoc1{}'.format(i+1)]).values
        globals()['df_oc2{}'.format(i+1)]=(globals()['dfoc2{}'.format(i+1)]).values 
    for i in range(len(vel2)):
        limit=np.where(globals()['dfa{}'.format(i+1)]['shorten']<lim)[0][-1]
        globals()['minF_a{}'.format(i+1)]=min(globals()['dfa{}'.format(i+1)]['Ftot'][0:limit])*0/1e12
        globals()['minF_p{}'.format(i+1)]=min(globals()['dfp{}'.format(i+1)]['Ftot'][0:limit])/1e12
        globals()['minF_t{}'.format(i+1)]=min(globals()['dft{}'.format(i+1)]['Ftot'][0:limit])/1e12
        globals()['minF_oc1{}'.format(i+1)]=min(globals()['dfoc1{}'.format(i+1)]['Ftot'][0:limit])/1e12
        globals()['minF_oc2{}'.format(i+1)]=min(globals()['dfoc2{}'.format(i+1)]['Ftot'][0:limit])/1e12
        minFa.append(globals()['minF_a{}'.format(i+1)])
        minFp.append(globals()['minF_p{}'.format(i+1)])
        minFt.append(globals()['minF_t{}'.format(i+1)])
        minFoc1.append(globals()['minF_oc1{}'.format(i+1)])
        minFoc2.append(globals()['minF_oc2{}'.format(i+1)])
        
    vel = [int(i) for i in vel2]
    fitdeg = 5
    z_a = np.polyfit(vel,minFa, fitdeg)
    z_p = np.polyfit(vel,minFp, fitdeg)
    z_t = np.polyfit(vel,minFt, fitdeg)
    z_oc1 = np.polyfit(vel,minFoc1, fitdeg)
    z_oc2 = np.polyfit(vel,minFoc2, fitdeg)
    p_a = np.poly1d(z_a)
    p_p = np.poly1d(z_p)
    p_t = np.poly1d(z_t)
    p_oc1 = np.poly1d(z_oc1)
    p_oc2 = np.poly1d(z_oc2)

    xp = np.linspace(0, max(vel), 100)
    
    

    fig43 = plt.figure(43,figsize=(14,9))
    fig43.clf()
#    set_style()

#    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#    fig43.clf()
#    set_style()
    ax1=plt.subplot(211)
    ax2=plt.subplot(212, sharex=ax1)
    
    ax1.plot(xp, p_a(xp), 'b-',linewidth=3,label='Archon')
    ax1.plot(xp, p_p(xp), 'g-',linewidth=3,label='Proton')
    ax1.plot(xp, p_t(xp), 'r-',linewidth=3,label='Tecton')
    ax1.plot(xp, p_oc1(xp), 'c-.',linewidth=3,label='OC30ma')
    ax2.plot(xp, p_a(xp), 'b-',linewidth=3,label='Archon')
    ax2.plot(xp, p_p(xp), 'g-',linewidth=3,label='Proton')
    ax2.plot(xp, p_t(xp), 'r-',linewidth=3,label='Tecton')
    ax2.plot(xp, p_oc1(xp), 'c-.',linewidth=3,label='OC30ma')
    ax2.plot(xp, p_oc2(xp), 'm--',linewidth=3,label='OC120ma')
    ax1.plot(vel,minFa, 'b.',markersize=20)
    ax1.plot(vel,minFp, 'g.',markersize=20)
    ax1.plot(vel,minFt, 'r.',markersize=20)
    ax1.plot(vel,minFoc1, 'c.',markersize=20)
    ax2.plot(vel,minFoc2, 'm.',markersize=20)
    ax1.plot(xp, p_a(xp)*0,color="0.5", label='_nolegend_', linewidth=1) 
    
    ax1.set_ylim(-5.2, 0.5)  # outliers only
    ax2.set_ylim(-21, -9)  # most of the data
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.setp(ax1.spines.values(), color='k', linewidth=2)
    plt.setp(ax2.spines.values(), color='k', linewidth=2)

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlim(0,max(vel))

    ax2.set_xlabel('Convergence rate  V  [$mm/year$]',fontsize=35,fontweight='bold')
    ax2.set_ylabel('Minimum F$_{b}$ [$10^{12}N/m$]',fontsize=35,fontweight='bold')
    ax2.yaxis.set_label_coords(-0.05,1.1)
    
#    plt.setp(ax1.spines.values(), color='k', linewidth=1.5)
    ax1.set_title('Maximum Slab-pull (within '+str(lim)+'km shortening)', fontsize=35, fontweight='bold', y=1.03) 
    ax2.text(0.03, 0.095, '(e)',color='k', transform=ax2.transAxes, fontsize=40, \
                  fontweight='bold',verticalalignment='bottom')
                  
    ax1.tick_params(direction='in',which='both',labelsize=30,labelbottom=False, labeltop=False, labelleft=True, labelright=False,
             bottom=False, top=True, left=True, right=True)
    ax1.tick_params(which='major', length=12,width=1)
    ax1.tick_params(which='minor', length=6)
    
    ax2.tick_params(direction='in',which='both',labelsize=30,labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=False, left=True, right=True)
    ax2.tick_params(which='major', length=12,width=1)
    ax2.tick_params(which='minor', length=6)
    
#    ax2.legend(title="Lithosphere type",loc='upper right', fontsize=27)
#    ax2.get_legend().get_title().set_fontsize('27')
#    ax1.grid(linestyle='dotted')
#    ax1.xaxis.grid()
#    ax2.grid(linestyle='dotted')
#    ax2.xaxis.grid()

    fig43.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig43.savefig('MinFbuoy_'+exp_name+'_vels.png', format='png', dpi=600)
    os.chdir('..')
    

    
if(InitTrho>0.):
    Ta=loadtxt('Archon_to_plot.txt', delimiter=" ")
    Tp=loadtxt('Proton_to_plot.txt', delimiter=" ")
    Tt=loadtxt('Tecton_to_plot.txt', delimiter=" ")
    Toc1=loadtxt('Ocean_70km_to_plot.txt', delimiter=" ") 
    Toc2=loadtxt('Ocean_110km_to_plot.txt', delimiter=" ")
    fig8 = plt.figure(8, figsize=(9,7))
    fig8.clf()
    
    xmajorLocator1   = FixedLocator(np.arange(200,1600,300))
    xminorLocator1   = MultipleLocator(50)
    xminorLocator2   = MultipleLocator(20)
    ymajorLocator1   = FixedLocator(np.arange(0,601,100))
    yminorLocator1   = MultipleLocator(10)
    
#    set_style2()
    
#    ax1 = plt.subplot2grid((2,2), (0,0), rowspan=2, colspan=1)
#    ax2 = plt.subplot2grid((2,2), (0,1), rowspan=2, colspan=2)
    ax1=plt.subplot(121)
    ax2=plt.subplot(122, sharey=ax1)
    
    hline1=np.linspace(min(Toc1[:,1]),max(Toc1[:,1]))
    hline2=np.linspace(min(Ta[:,2])-100000,max(Toc1[:,2]))
    
    dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)] 
    up_lim = 0. 
    up_lim1 = -10.  
    up_lim2 = -40. 
    uplim=np.int(np.where(np.round(Ta[:,0])==up_lim)[0])
    uplim1=np.int(np.where(np.floor(Ta[:,0])==up_lim1)[0])+1
    uplim2=np.int(np.where(np.round(Ta[:,0])==up_lim2)[0])+1
    opaq=0.3    
    
    ax1.plot(hline1,hline1*0+10,color = 'k',linestyle='-', linewidth=1,alpha=0.2)  
    ax1.plot(hline1,hline1*0+40,color = 'k',linestyle='-', linewidth=1,alpha=0.2) 
    ax2.plot(hline2,hline2*0+10,color = 'k',linestyle='-', linewidth=1,alpha=0.2)  
    ax2.plot(hline2,hline2*0+40,color = 'k',linestyle='-', linewidth=1,alpha=0.2) 
    
    ax1.plot(Ta[uplim2-1::,1],-(Ta[uplim2-1::,0]),'-b',linewidth=1,label='Archon')
    ax1.plot(hline1,hline1*0+200,color = 'b',linestyle='--', linewidth=1,alpha=opaq)
    ax1.plot(Tp[uplim2-1::,1],-(Tp[uplim2-1::,0]),'-g',linewidth=1,label='Proton')
    ax1.plot(hline1,hline1*0+150,color = 'g',linestyle='--', linewidth=1,alpha=opaq)
    ax1.plot(Tt[uplim2-1::,1],-(Tt[uplim2-1::,0]),'-r', linewidth=1,label='Tecton')
    ax1.plot(hline1,hline1*0+120,color = 'r',linestyle='--', linewidth=1,alpha=opaq)
    ax1.plot(Toc1[uplim1-1::,1],-(Toc1[uplim1-1::,0]),'xkcd:sienna', linewidth=1,label='OC30ma')
    ax1.plot(hline1,hline1*0+70,color='xkcd:sienna',linestyle='--', linewidth=1,alpha=opaq)
    ax1.plot(Toc2[uplim1-1::,1],-(Toc2[uplim1-1::,0]),'-m', linewidth=1,label='OC120ma')
    ax1.plot(hline1,hline1*0+110,color = 'm',linestyle='--', linewidth=1,alpha=opaq)
    
    ax1.set_xlabel('Temperature ($^{\circ}C$)',fontweight='normal',fontsize=18)
    ax1.set_ylabel('Depth ($km$)',fontweight='normal',fontsize=18)
    ax1.invert_yaxis()
    ax1.set_ylim(0,400)
    ax1.set_xlim(200,max(Toc1[:,1]))
#    ax1.legend(loc='lower left',fontsize=16)
    
    
    ax2.plot(Ta[uplim2::,2],-(Ta[uplim2::,0]),'-b',linewidth=1,label='Archon')
    ax2.plot(hline2,hline2*0+200,color = 'b',linestyle='--', linewidth=1,alpha=opaq)
    ax2.plot(Tp[uplim2::,2],-(Tp[uplim2::,0]),'-g',linewidth=1,label='Proton')
    ax2.plot(hline2,hline2*0+150,color = 'g',linestyle='--', linewidth=1,alpha=opaq)
    ax2.plot(Tt[uplim2::,2],-(Tt[uplim2::,0]),'-r', linewidth=1,label='Tecton')
    ax2.plot(hline2,hline2*0+120,color = 'r',linestyle='--', linewidth=1,alpha=opaq)
    ax2.plot(Toc1[uplim1::,2],-(Toc1[uplim1::,0]),'xkcd:sienna', linewidth=1,label='OC30ma')
    ax2.plot(hline2,hline2*0+70,color='xkcd:sienna',linestyle='--', linewidth=1,alpha=opaq)
    ax2.plot(Toc2[uplim1::,2],-(Toc2[uplim1::,0]),'-m', linewidth=1,label='OC120ma')
    ax2.plot(hline2,hline2*0+110,color = 'm',linestyle='--', linewidth=1,alpha=opaq)
    ax2.set_xlabel('Density ($kg$ $m^{-3}$)',fontweight='normal',fontsize=20)
#    ax2.set_ylabel('Depth ($km$)',fontweight='bold',fontsize=22)
#    ax2.set_ylim(0,600)
    ax2.set_xlim(min(Toc1[uplim1::,2])-10,max(Toc1[:,2]))
#    ax2.set_xlim(3250,max(Toc1[:,2]))
    ax2.invert_yaxis()
#    ax2.legend(loc='lower left',fontsize=16)
    
#    ax2.text(3790, 225, r'200km',fontsize=19, color='b')
#    ax2.text(3790, 175, r'150km',fontsize=19, color='g')
#    ax2.text(3790, 145, r'120km',fontsize=19, color='r')
#    ax2.text(3790, 108, r'110km',fontsize=19, color='m')
#    ax2.text(3790, 68, r'70km',fontsize=19, color='xkcd:sienna')
#    ax2.text(3720, 25, r'LAB depth',fontsize=19, color='k')
    
    ax2.text(3400,22, '$10 km - MOHO_{oceanic}$',fontsize=13, color='k')
    ax2.text(3400,52, '$40 km - MOHO_{continental}$',fontsize=13, color='k')
    
    
    ax1.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='major', length=12,width=1)
    ax1.tick_params(which='minor', length=6)
    ax2.tick_params(direction='in',which='both',labelbottom=True, labeltop=False, labelleft=True, labelright=False,
             bottom=True, top=True, left=True, right=True)
    ax2.tick_params(which='major', length=10,width=1)
    ax2.tick_params(which='minor', length=6)
    
#    ax1.xaxis.set_major_locator(xmajorLocator1)
    ax1.xaxis.set_minor_locator(xminorLocator1)
    ax1.yaxis.set_major_locator(ymajorLocator1)
    ax1.yaxis.set_minor_locator(yminorLocator1)
    ax2.xaxis.set_minor_locator(xminorLocator2)
  
    
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    plt.setp(ax1.spines.values(), color='k', linewidth=1.5)
    plt.setp(ax2.spines.values(), color='k', linewidth=1.5)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=15)
    ax1.legend(loc='lower left', fontsize=15)
    #fig8.suptitle('Tecton: F$_{bouy}$ vs Time', fontsize=16, fontweight='bold')   
    fig8.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig8.subplots_adjust(wspace=0.05, hspace=0.15)
    ax1.grid(False)
    ax2.grid(False)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig8.savefig('InitTrho_'+exp_name+'.png', format='png', dpi=300)
    os.chdir('..')

if(ComponentsTime>0.):
    fig15 = plt.figure(figsize=(15,10))
    fig15.clf()
    
    if (len(vel)==4):
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((4,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((4,2), (0,1), rowspan=2, colspan=1)
            ax4 = plt.subplot2grid((4,2), (2,1), rowspan=2, colspan=2, sharex=ax3, sharey=ax3)
    if (len(vel)==6):   
        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2, colspan=1)
            ax2 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=1, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot2grid((6,2), (0,1), rowspan=2, colspan=2)
            ax5 = plt.subplot2grid((6,2), (2,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
            ax6 = plt.subplot2grid((6,2), (4,1), rowspan=2, colspan=2, sharex=ax4, sharey=ax4)
        
    minorLocator = AutoMinorLocator()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    for i in range(len(vel)):
#        newdiffus=globals()['dfp{}'.format(i+1)]['Ftot'] \
#                    -globals()['dfp{}'.format(i+1)]['FadvA']
                    #+globals()['dfp{}'.format(i+1)]['FadvB'] 
                    #+globals()['dfp{}'.format(i+1)]['FdiffusB']
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['Ftot']/1e12, label="F$_{tot}$", linewidth=3,color='k')
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['FadvA']/1e12, label="F$_{adv}$^LAB", linewidth=3,color='b')
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['FadvB']/1e12, label="F$_{adv}$vLAB", linewidth=3,color='g')
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['Fdiffus']/1e12 ,'--m', label="F$_{diffus}$vLAB", linewidth=3)
        globals()['ax{}'.format(i+1)].plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['shorten']*0,'--k', label='_nolegend_')
        
    #    globals()['ax{}'.format(i+1)].axvline(x=bottma,color='k', linestyle='--',linewidth=2)
    #    globals()['ax{}'.format(i+1)].axvline(x=bottmp,color='r', linestyle='--',linewidth=2)
    #    globals()['ax{}'.format(i+1)].axvline(x=bottmt,color='m', linestyle='--',linewidth=2)
        
        globals()['ax{}'.format(i+1)].set_xlim(0,900)
        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].legend(loc='best',fontsize=12)
        globals()['ax{}'.format(i+1)].text(0.02, 0.11, 'v='+str(vel[i])+'mm/yr', \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=16, \
                  verticalalignment='top', bbox=props)
         
    
        if i==1 or i==len(vel)-1:
            globals()['ax{}'.format(i+1)].set_xlabel('Shortening (km)',fontsize=16, fontweight='bold')
            globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_xticklabels(), visible=False)
        if i<=1:
            globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=16, fontweight='bold')
        else:
            plt.setp(globals()['ax{}'.format(i+1)].get_yticklabels(), visible=False)
      
    #fig15.suptitle('Diffusion below LAB', fontsize=16, fontweight='bold')   
    fig15.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig15.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig15.savefig('Components_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')


#==============================================================================
#if(FdiffusAShortenVel>0.):
#    fig16 = plt.figure(figsize=(10,15))
#    fig16.clf()
#    
#    with sns.axes_style("darkgrid"):
#        ax1 = plt.subplot(311)
#        ax2 = plt.subplot(312)
#        ax3 = plt.subplot(313)
#    
#    color=['b','g','r']
#    color2=['--b','+b','*b','b']
#    color3=['--g','+g','*g','g']
#    color4=['--r','+r','*r','r']
#    mtype=['Archon','Proton', 'Tecton']
#    lim=600
#    
#    minorLocator = AutoMinorLocator()
##    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#    
#    for i in range(4):
#        ax1.plot(globals()['dfa{}'.format(i+1)]['shorten'], \
#                  globals()['dfa{}'.format(i+1)]['FdiffusA']/1e12, str(color2[i]), label="vel"+str(vel[i]), linewidth=3)
#        ax2.plot(globals()['dfp{}'.format(i+1)]['shorten'], \
#                  globals()['dfp{}'.format(i+1)]['FdiffusA']/1e12, str(color3[i]), label="vel"+str(vel[i]), linewidth=3)
#        ax3.plot(globals()['dft{}'.format(i+1)]['shorten'], \
#                  globals()['dft{}'.format(i+1)]['FdiffusA']/1e12, str(color4[i]), label="vel"+str(vel[i]), linewidth=3)
#
#
#    for i in range(3):   
#        props = dict(boxstyle='round', facecolor=color[i], alpha=0.15)
#        globals()['ax{}'.format(i+1)].set_xlim(0,lim)
##        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
#        sns.despine(right=True,top=True)
#        globals()['ax{}'.format(i+1)].text(0.04, 0.95, mtype[i],color=color[i], \
#                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
#                  verticalalignment='top', bbox=props)
#        globals()['ax{}'.format(i+1)].legend(loc='southeast',fontsize=20)
#        globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
#        globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=15, fontweight='bold')
#        
#    ax3.set_xlabel('Shortening (km)',fontsize=22, fontweight='bold')
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    plt.setp(ax2.get_xticklabels(), visible=False)
#    fig16.suptitle('Diffusion above LAB at different velocity', fontsize=16, fontweight='bold')   
#    fig16.tight_layout(rect=[0, 0.03, 1, 0.97])
#    fig16.subplots_adjust(wspace=0.0, hspace=0.05)
#    plt.show()
#    print("Images in directory %s" % dir)
#    os.chdir(dir)
#    fig16.savefig('FdiffusAVel_'+exp_name+'_vels.png', format='png', dpi=300)
#    os.chdir('..')
#    
    
    #==============================================================================
if(FdiffusShortenVel>0.):
    fig17 = plt.figure(figsize=(10,15))
    fig17.clf()
    
    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
    
    color=['b','g','r']
    color2=['--b','+b','*b','b']
    color3=['--g','+g','*g','g']
    color4=['--r','+r','*r','r']
    mtype=['Archon','Proton', 'Tecton']
    lim=600
    
    minorLocator = AutoMinorLocator()
#    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    for i in range(4):
        ax1.plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['Fdiffus']/1e12, str(color2[i]), label="vel"+str(vel[i]), linewidth=3)
        ax2.plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['Fdiffus']/1e12, str(color3[i]), label="vel"+str(vel[i]), linewidth=3)
        ax3.plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['Fdiffus']/1e12, str(color4[i]), label="vel"+str(vel[i]), linewidth=3)


    for i in range(3):   
        props = dict(boxstyle='round', facecolor=color[i], alpha=0.15)
        globals()['ax{}'.format(i+1)].set_xlim(0,lim)
#        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].text(0.04, 0.95, mtype[i],color=color[i], \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
                  verticalalignment='top', bbox=props)
        globals()['ax{}'.format(i+1)].legend(loc='southeast',fontsize=20)
        globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=15, fontweight='bold')
        
    ax3.set_xlabel('Shortening (km)',fontsize=22, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig17.suptitle('Diffusion below LAB at different velocity', fontsize=16, fontweight='bold')   
    fig17.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig17.subplots_adjust(wspace=0.0, hspace=0.05)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig17.savefig('FdiffusVel_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')
    
        #==============================================================================
if(FadvAShortenVel>0.):
    fig18 = plt.figure(figsize=(10,15))
    fig18.clf()
    
    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
    
    color=['b','g','r']
    color2=['--b','+b','*b','b']
    color3=['--g','+g','*g','g']
    color4=['--r','+r','*r','r']
    mtype=['Archon','Proton', 'Tecton']
    lim=600
    
    minorLocator = AutoMinorLocator()
#    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    for i in range(4):
        ax1.plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['FadvA']/1e12, str(color2[i]), label="vel"+str(vel[i]), linewidth=3)
        ax2.plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['FadvA']/1e12, str(color3[i]), label="vel"+str(vel[i]), linewidth=3)
        ax3.plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['FadvA']/1e12, str(color4[i]), label="vel"+str(vel[i]), linewidth=3)


    for i in range(3):   
        props = dict(boxstyle='round', facecolor=color[i], alpha=0.15)
        globals()['ax{}'.format(i+1)].set_xlim(0,lim)
#        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].text(0.04, 0.95, mtype[i],color=color[i], \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
                  verticalalignment='top', bbox=props)
        globals()['ax{}'.format(i+1)].legend(loc='southeast',fontsize=20)
        globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=15, fontweight='bold')
        
    ax3.set_xlabel('Shortening (km)',fontsize=22, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig18.suptitle('Advection above LAB at different velocity', fontsize=16, fontweight='bold')   
    fig18.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig18.subplots_adjust(wspace=0.0, hspace=0.05)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig18.savefig('FadvAVel_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')
    
        #==============================================================================
if(FadvBShortenVel>0.):
    fig19 = plt.figure(figsize=(10,15))
    fig19.clf()
    
    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
    
    color=['b','g','r']
    color2=['--b','+b','*b','b']
    color3=['--g','+g','*g','g']
    color4=['--r','+r','*r','r']
    mtype=['Archon','Proton', 'Tecton']
    lim=600
    
    minorLocator = AutoMinorLocator()
#    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    for i in range(4):
        ax1.plot(globals()['dfa{}'.format(i+1)]['shorten'], \
                  globals()['dfa{}'.format(i+1)]['FadvB']/1e12, str(color2[i]), label="vel"+str(vel[i]), linewidth=3)
        ax2.plot(globals()['dfp{}'.format(i+1)]['shorten'], \
                  globals()['dfp{}'.format(i+1)]['FadvB']/1e12, str(color3[i]), label="vel"+str(vel[i]), linewidth=3)
        ax3.plot(globals()['dft{}'.format(i+1)]['shorten'], \
                  globals()['dft{}'.format(i+1)]['FadvB']/1e12, str(color4[i]), label="vel"+str(vel[i]), linewidth=3)


    for i in range(3):   
        props = dict(boxstyle='round', facecolor=color[i], alpha=0.15)
        globals()['ax{}'.format(i+1)].set_xlim(0,lim)
#        globals()['ax{}'.format(i+1)].set_ylim(-10,10)
        sns.despine(right=True,top=True)
        globals()['ax{}'.format(i+1)].text(0.04, 0.95, mtype[i],color=color[i], \
                  transform=globals()['ax{}'.format(i+1)].transAxes, fontsize=20, \
                  verticalalignment='top', bbox=props)
        globals()['ax{}'.format(i+1)].legend(loc='southeast',fontsize=20)
        globals()['ax{}'.format(i+1)].tick_params(which='minor', length=7, color='k')  
        globals()['ax{}'.format(i+1)].set_ylabel('F$_{buoy}$ ($10^{12}$N/m)',fontsize=15, fontweight='bold')
        
    ax3.set_xlabel('Shortening (km)',fontsize=22, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig19.suptitle('Advection below LAB at different velocity', fontsize=16, fontweight='bold')   
    fig19.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig19.subplots_adjust(wspace=0.0, hspace=0.05)
    plt.show()
    print("Images in directory %s" % dir)
    os.chdir(dir)
    fig19.savefig('FadvBVel_'+exp_name+'_vels.png', format='png', dpi=300)
    os.chdir('..')
    
    