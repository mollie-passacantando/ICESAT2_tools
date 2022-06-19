## final script mock-up that calls the ICESAT_tools module

from dask.distributed import Client, progress
import modules.ICESAT_tools as tool
from dask_jobqueue import SLURMCluster
import os
import numpy as np
import dask.array as da
import glob
import csv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc
import h5py
from scipy.interpolate import griddata
import pandas as pd


ATL03_target_dir = 'ATL03_data/Chukchi_project/2019/04'
ATL10_target_dir = 'icethick_data/Chukchi_project/2019/04'
track = 'gt1l'

# initiate the cluster for parallel processing 

cluster1 = SLURMCluster(cores=1, 
                        processes=1,
                        memory="250GB",
                        walltime="01:00:00",
                        local_directory = '/data/looselab/mollie',
                        queue="general")
cluster1.adapt(maximum_jobs=36)
client1=Client(cluster1)

# find the surface return from ATL03
kwargs = {'target_directory':ATL03_target_dir,'hbin':7,'vbin':.15,'track':'gt1l','surface':'ocean'}
file_list = sorted(os.listdir('/data/looselab/mollie/data/'+ATL03_target_dir))

f = client1.map(tool.get_surface,file_list,**kwargs) # return a list of futures that has only the photons under the surface

# find ice thickness from ATL10
kwargs1 = {'target_directory':ATL10_target_dir}
file_list1 = sorted(os.listdir('/data/looselab/mollie/data/'+ATL10_target_dir))
# file_list1 = [os.path.basename(x) for x in sorted(glob.glob(r'/data/looselab/mollie/'+ATL10_target_dir+'/*gt1l*.nc'))]
l = client1.map(tool.get_thickness,sorted(file_list1),**kwargs1) # return a list of futures that has the ice thicknesses

ATL03_time_lists = client1.map(lambda a:a['time'],f) # pull out ATL03 times
ATL03_time_range = client1.map(tool.min_max_of_lists,ATL03_time_lists) # we need the min and max time from each ATL10 file, to match it with the appropriate ATL03 files
ATL03_times = client1.gather(ATL03_time_range)
kwargs2 = {'list_of_times':ATL03_times,'time_constraint':600,'binsize':.1,'target_directory':ATL10_target_dir}
# fzdata = np.loadtxt('homemade_Fz_data.csv',delimiter=',')
# fzdepth = np.loadtxt('homemade_Fz_depth.csv',delimiter=',')

# match up the files - here we are looking at a specific file
ATL10_future = l[0].result()
ATL03_ind = tool.match_ATL03_ATL10_file_num(ATL10_future,**kwargs2) # find the matching ATL03 file

ATL03_future = f[int(ATL03_ind)].result()
kwargs3 = {'ATL10data':ATL10_future,'binsize':20}
full_data = tool.match_ATL03_ATL10_data(ATL03_future,**kwargs3)
twt = tool.twt(full_data)

ATL03_profs = tool.get_ATL03_profiles(ATL03_future,100000)

vbins= np.arange(-30,30,.05)
Bc,z,bin_edges,fzdata,fzdepth,Bm = ATL03_deconvolve(ATL03_profs['heights'][0][0],vbins,tep_hist,tep_hist_time)

##################################################### that is all our code stuff, now lets plot and test things out

##### order of plot code follows order of analysis process ####


############# get some basic variables out ##############
ATL10_future = l[0].result()
ATL03_ind = tool.match_ATL03_ATL10_file_num(ATL10_future,**kwargs2) # find 
icefile = ATL10_future['filename'][0]
phfile = f[int(ATL03_ind)].result()['filename'][0]

f2 = nc.Dataset('/data/looselab/mollie/data/'+ATL10_target_dir+'/'+str(icefile)) # open h5 files
lat10 = f2['latitude'][:]
lon10 = f2['longitude'][:]
time10 = f2['gps_seconds'][:]

f1 = h5py.File('/data/looselab/mollie/data/'+ATL03_target_dir+'/'+str(phfile),'r') # open h5 files
h_ph = f1[track+'/heights/h_ph'][:]
pulse = f1[track+'/heights/ph_id_pulse'][:]
dist = f1[track+'/heights/dist_ph_along'][:]
seglen = f1[track+'/geolocation/segment_length'][:]
lat03 = f1[track+'/heights/lat_ph'][:]
lon03 = f1[track+'/heights/lon_ph'][:]
deltime03 = f1[track+'/heights/delta_time'][:]
bandheight = f1[track+'/bckgrd_atlas/tlm_top_band1'][:]
banddelt = f1[track+'/bckgrd_atlas/delta_time'][:]
total_dist = sum(seglen)+dist[-1]
photon_num = lon03.shape[0]
dist_corrected = np.arange(0,total_dist,total_dist/photon_num)
tep_hist = f1['/atlas_impulse_response/pce1_spot1/tep_histogram/tep_hist'][:]
time_offset = f1['/ancillary_data/atlas_sdp_gps_epoch']

tep_hist_time = f1['/atlas_impulse_response/pce1_spot1/tep_histogram/tep_hist_time'][:]




# Plotting Code Follows
###########################################################



###### match ATL03 and ATL10 plot #####
plt.cla()
plt.clf()
fig = plt.figure(figsize=(2,3)) #initiate plot
axs = plt.axes(projection=ccrs.NorthPolarStereo())
proj = ccrs.NorthPolarStereo()
axs.coastlines(resolution='10m')
axs.gridlines()
axs.set_extent([-180,-150,70,76],crs = ccrs.PlateCarree())
axs.scatter(lon10,lat10,s=2,c='red',transform=ccrs.PlateCarree(),label='ATL10')
axs.scatter(lon03[0::1000],lat03[0::1000],s=1,c='green',transform=ccrs.PlateCarree(),label='ATL03')
axs.legend()
axs.scatter(lon10,lat10,s=2,c='red',transform=ccrs.PlateCarree(),label='ATL10')
axs.scatter(lon03[0::1000],lat03[0::1000],s=1,c='green',transform=ccrs.PlateCarree(),label='ATL03')
plt.savefig('match_tracks.png')
##################################



############## raw ATL03 plots ###############
plt.cla()
plt.clf()
fig = plt.figure(figsize=(20,10))
plt.scatter(dist_corrected,h_ph,s=.25)
plt.xlabel('Along-track distance (m)',fontsize=18,fontweight='bold')
plt.ylabel('Altitude (m)',fontsize=18,fontweight='bold')
plt.xticks(fontsize=14,fontweight='bold')
plt.yticks(fontsize=14,fontweight='bold')
# plt.ylim([-10,5])
plt.savefig('h_ph1.png')


plt.cla()
plt.clf()
fig = plt.figure(figsize=(6,10)) #initiate plot
axs = plt.axes(projection=ccrs.NorthPolarStereo())
proj = ccrs.NorthPolarStereo()
axs.coastlines(resolution='10m')
axs.gridlines()
sc1=axs.scatter(lon03[0::1000],lat03[0::1000],s=1,c=dist_corrected[0::1000],transform=ccrs.PlateCarree())
cbar = fig.colorbar(sc1)
cbar.ax.tick_params(labelsize=14,labelweight='bold')
cbar.set_label('distance (m)',fontsize=18,fontweight='bold')
plt.savefig('h_ph_space.png')
###########################################


############# ATL03 plots to identify telemetry window ##########
inds = tool.get_rid_of_land_photons(h_ph,dist_corrected,bandheight,binsize,deltime03,banddelt)
plt.cla()
plt.clf()
fig,axs = plt.subplots(2,1,figsize=(20,10))
axs[0].scatter(deltime03,h_ph,s=.25)
axs[0].scatter(banddelt,bandheight,c='cyan',s=1,label='telemetry window height')
axs[0].set_title('All photons with telemetry height',fontsize=18,fontweight='bold')
axs[0].tick_params(labelsize=14)
# axs[0].set_ylim([-50,50])
# axs[1].scatter(f[0].result()['time'][0],f[0].result()['heights'][0],s=1)
axs[1].scatter(deltime03[inds],h_ph[inds],s=1)
axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_xlabel('time (s since 2018-01-01)',fontsize=18,fontweight='bold')
axs[1].set_title('Photons with -100 m < height < 100 m , and telemetry height < 30 m',fontsize=18,fontweight='bold')
axs[1].tick_params(labelsize=14)
plt.savefig('h_only_ice_and_ocean.png')
##############################################################

############## histogram plots to identify surface #################
workingdist = dist_corrected[3660000:5000000]
workingheight = h_ph[3660000:5000000]
plt.cla()
plt.clf()
fig,axs = plt.subplots(2,1,figsize=(10,10))
axs[0].scatter(workingdist,workingheight,s=1)
axs[0].set_ylim([-20,5])
axs[0].set_ylabel('Altitude (m)',fontsize=14)
axs[0].set_title('Raw photon data',fontsize=18,fontweight='bold')
# axs[0].set_xlim([600000,1000000])
axs[1].pcolormesh(xedges[1::], yedges[1::], h.T)
axs[1].set_ylim([-20,5])
axs[1].set_title('Photon Histogram, 7m horizontal and 15cm vertical bins',fontsize=18,fontweight='bold')
axs[1].set_xlabel('Along track distance (m)',fontsize=14)
axs[1].set_ylabel('Altitude (m)',fontsize=14)
plt.savefig('histplot.png')

###########################################################


###### ATL03 surface-corrected plots #####
## pull out the corrected data, corrected through the get_surface function stored in the f futures
var = f[int(ATL03_ind)].result()
x = var['dist'][0]
y=var['heights'][0]
lats = var['lat'][0]
lons = var['lon'][0]

#plots
plt.cla()
plt.clf()
fig = plt.figure(figsize=(20,10))
plt.scatter(x,y,s=.25)
plt.ylim([-10,5])
plt.xticks(fontsize=14,fontweight='bold')
plt.yticks(fontsize=14,fontweight='bold')
plt.xlabel('Distance Along Track (m)',fontsize=18,fontweight='bold')
plt.ylabel('Altitude(m)',fontsize=18,fontweight='bold')
plt.savefig('h_ph_surface_corrected.png')

plt.cla()
plt.clf()
fig = plt.figure(figsize=(7,7)) #initiate plot
axs = plt.axes(projection=ccrs.NorthPolarStereo())
proj = ccrs.NorthPolarStereo()
axs.coastlines(resolution='10m')
axs.gridlines()
sc1=axs.scatter(lons[0::1000],lats[0::1000],s=1,c=x[0::1000],transform=ccrs.PlateCarree())
fig.colorbar(sc1)
plt.savefig('h_ph_surface_space.png')
#############################################




############### ATL03 profile plots #############


vbins = np.arange(-30,30,.05) 
plotnum = len(ATL03_profs['heights'][0])
binres = [20,20000,280000]
labels = ['a','b','c']
plotnum=[0,1,2]
plt.cla()
plt.clf()
fig,axs = plt.subplots(1,len(plotnum),figsize=(10,7))
for index in plotnum:
    ATL03_profs = tool.get_ATL03_profiles(ATL03_future,binres[index])
    [h1,bin_edges] = np.histogram(ATL03_profs['heights'][0][0],vbins) # bin h_ph into hbins and vbins
    h_norm1 = (h1-np.min(h1)) / np.ptp(h1)
    axs[index].plot(np.log(h_norm1),bin_edges[1::])
    axs[index].set_ylim([-15,5])
    axs[index].set_xlabel('Normalized Photons per bin')
    # axs[count].set_xlim([-10,.608])
    axs[index].set_title(str(binres[index]) +'m binsize')
    axs[index].set_ylabel('Altitude(meter)')
plt.savefig('ATL03_profs.png')
##############################################


###### match ATL03 and ATL10 plot #####
plt.cla()
plt.clf()
fig = plt.figure(figsize=(7,7)) #initiate plot
axs = plt.axes(projection=ccrs.NorthPolarStereo())
proj = ccrs.NorthPolarStereo()
axs.coastlines(resolution='10m')
axs.gridlines()
axs.set_extent([-180,-150,70,76],crs = ccrs.PlateCarree())
axs.scatter(lon10,lat10,s=2,c='red',transform=ccrs.PlateCarree(),label='ATL10')
axs.scatter(lon03[0::1000],lat03[0::1000],s=1,c='green',transform=ccrs.PlateCarree(),label='ATL03')
axs.legend()
plt.savefig('match_tracks.png')
##################################



########### Deconvolution with TEP plots ###############
plt.cla()
plt.clf()
fig,axs = plt.subplots(1,3,figsize=(10,10)) #initiate plot
axs[0].plot(np.log(Bm),bin_edges)
# axs[0].plot(np.log(h_norm),np.flip(bin_edges[1::]))
axs[0].set_ylim([-15,1])
axs[0].set_title('measured signal')
# axs[1].plot(np.log(fzdata),fzdepth,c='lime')
axs[1].plot(np.log(fzdata_var),fzdepth_var,c='lime')
axs[1].set_ylim([-15,1])
axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_title('TEP Echo')
axs[2].plot(np.log(Bc),z[1::])
axs[2].set_title('deconvolution result')
axs[2].set_ylim([-15,1])
# axs[2].set_xlim(axs[0].get_xlim())
plt.savefig('deconvolve_tep.png')
######################################################



