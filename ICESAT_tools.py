import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import datetime
from datetime import datetime, timezone, timedelta
import numpy.matlib
import pandas as pd
import asyncio
import csv
from geopy import distance
from scipy.interpolate import NearestNDInterpolator, griddata
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
import netCDF4 as nc
import math
import numpy.ma as ma

## use parallel processing through dask to download ATL10 files
def ATL10_download(start_date,end_date,target_directory): 

    cluster = SLURMCluster(cores=1,
                            memory="180GB",
                            processes=1,
                            walltime="01:00:00",
                            queue="general")
    cluster.adapt(maximum_jobs=20)
    client=Client(cluster)
    import modules.download_ATL10 as dl
    start_date = '2019-05-01'
    end_date = '2019-05-31'
    target_directory = 'Chukchi_project/05'
    bounding_box = '-180,70,-150,76'
    url_list = dl.main(start_date,end_date,bounding_box)
    kwargs = {'target_directory':target_directory}
    f = client.map(dl.single_cmr_download,url_list,**kwargs)
    progress(f)

def ATL03_download(start_date,end_date,target_directory): 

    cluster = SLURMCluster(cores=1, # initiate the cluster for parallel processing 
                            processes=1,
                            memory="250GB",
                            walltime="01:00:00",
                            local_directory = '/data/looselab/mollie',
                            queue="general")
    cluster.adapt(maximum_jobs=36,target_duration="10m")

    client=Client(cluster)
    import modules.download_ATL03 as dl
    start_date = '2019-07-15'
    end_date = '2019-07-17'
    target_directory = 'Chukchi_project/ATL03_chuck_07'
    kwargs = {'target_directory':target_directory}
    bounding_box = '-180,70,-150,76'
    url_list = dl.main(start_date,end_date,bounding_box)
    f1 = client.map(dl.single_cmr_download,url_list,**kwargs)
    progress(f1)

# start=14
# from dask.distributed import wait
# for k in [21,28,30]:
#     f= client.map(dl.single_cmr_download,url_list[start:k],**kwargs)
#     progress(f)
#     wait(f)
#     start = k
    

## identify surface using ATL03 backscatter, with the surface identified as the maximum return from
## the normalized photon return binned in vertical and horiztonal bins

def get_surface(filename,**kwargs):

    h_ph = np.array([])
    dist_corrected = np.array([])
    all_data = []
    ############### ONCE YOU WRITE THE FUNCTION FOR THE YAW FLIP, JUST PICK THE TIME AND FIGURE OUT THE GOOD TRACK!!!!###
    track = kwargs['track']
    target_directory = kwargs['target_directory']
    vbin = kwargs['vbin']
    hbin = kwargs['hbin']
    ATL03_df = pd.DataFrame()

    ####### EVENTUALLY LET YOU SPECIFY BINSIZE!!
    binsize=20
    # loop through all files in target directory
    if filename.endswith('h5') and filename.startswith("ATL03"):

        try:
            f1 = h5py.File('/data/looselab/mollie/data/'+target_directory+'/'+str(filename),'r') # open h5 files
            print(filename)
            # pull out photon heights, distane along track, and segment length
            h_ph = f1[track+'/heights/h_ph'][:]
            dist = f1[track+'/heights/dist_ph_along'][:]
            seglen = f1[track+'/geolocation/segment_length'][:]
            lats = f1[track+'/heights/lat_ph'][:]
            lons = f1[track+'/heights/lon_ph'][:]
            deltime = f1[track+'/heights/delta_time'][:]
            bandheight = f1[track+'/bckgrd_atlas/tlm_top_band1'][:]
            deltimeband = f1[track+'/bckgrd_atlas/delta_time'][:]
            h_ph = h_ph[np.logical_and(lats<=76,lats>=70)] # only
            deltime = deltime[np.logical_and(lats<=76,lats>=70)]


            # correct the along track distance to account for total distance including segement length
            total_dist = sum(seglen)+dist[-1]
            photon_num = lons.shape[0]
            dist_corrected = np.arange(0,total_dist,total_dist/photon_num)
            dist_corrected = dist_corrected[np.logical_and(lats<=76,lats>=70)]
            lons = lons[np.logical_and(lats<=76,lats>=70)]
            lats = lats[np.logical_and(lats<=76,lats>=70)]


            if kwargs['surface'] == 'ocean':
                surface_inds = get_rid_of_land_photons(h_ph,dist_corrected,bandheight,binsize,deltime,deltimeband)
            elif kwargs['surface'] == 'land':
                # surface_inds = get_only_land_photons(h_ph,dist_corrected,bandheight,binsize,deltime,deltimeband)
                print('not working at the moment!!!!')
                
            # pull out the photons that are not on land
            h_ph = h_ph[surface_inds]
            dist_corrected = dist_corrected[surface_inds]
            lats = lats[surface_inds]
            lons = lons[surface_inds]
            deltime = deltime[surface_inds]

            # break up the data into the ocean sections (there will be space between the ocean sections)
            coherent_sections = np.argwhere(np.diff(surface_inds)>5000)

            if len(coherent_sections>0):
                coherent_h = np.array_split(h_ph,np.squeeze(np.vstack([0,coherent_sections])).astype(int))
                coherent_dist = np.array_split(dist_corrected,np.squeeze(np.vstack([0,coherent_sections])).astype(int))
                coherent_lats = np.array_split(lats,np.squeeze(np.vstack([0,coherent_sections])).astype(int))
                coherent_lons = np.array_split(lons,np.squeeze(np.vstack([0,coherent_sections])).astype(int))
                coherent_time = np.array_split(deltime,np.squeeze(np.vstack([0,coherent_sections])).astype(int))

                # only use sections that have a significant amount of data in them

                h_ph = [a for a in coherent_h if a.shape[0]>20000]
                dist_corrected = [a for a in coherent_dist if a.shape[0]>20000]
                lats = [a for a in coherent_lats if a.shape[0]>20000]
                lons = [a for a in coherent_lons if a.shape[0]>20000]
                deltime = [a for a in coherent_time if a.shape[0]>20000]
            else: # turn these into lists so they can go through the loop as well
                h_ph = [h_ph]
                dist_corrected= [dist_corrected]
                lats = [lats]
                lons = [lons]
                deltime = [deltime]

            # set the bins for the 2d histogra
            if kwargs['surface'] == 'ocean':
                vbins = np.arange(-50,50,vbin) # create vertical bins
            elif kwargs['surface'] == 'land':
                vbins = np.arange(0,1500,vbin) 

            ATL03_df = pd.DataFrame()
            for count, data in enumerate(h_ph):
                print(dist_corrected[count].shape)
                current_df = ocean_data_hist(dist_corrected[count],h_ph[count],lats[count],lons[count],deltime[count],vbins,filename)
                ATL03_df = ATL03_df.append(current_df)
        except:
            print('file not read')
            current_df = [{'filename':'truncated_file','time':-999,'heights':-999,'dist':-999,'lat':-999,'lon':-999}]
            ATL03_df = pd.DataFrame().append(current_df)
                
    else:
        current_df = [{'filename':'truncated_file','time':-999,'heights':-999,'dist':-999,'lat':-999,'lon':-999}]
        ATL03_df = pd.DataFrame().append(current_df)
    ATL03_df = ATL03_df.set_index([pd.Index(np.arange(len(ATL03_df)))])
    return ATL03_df


def get_rid_of_land_photons(heights,distances,bandheight,binsize,deltimeph,deltimeband):
    ##### this is a super rough estimation of where there is ocean/ice based on the telemetry window
    ##### input a full photon track and get out just the photons that are within some threshold 
    ##### will sort into horizontal bins, then 
    #
    bandheight = bandheight[np.logical_and(deltimeband>=np.min(deltimeph),deltimeband<=np.max(deltimeph))] # need to get rid of extraneous points
    deltimeband = deltimeband[np.logical_and(deltimeband>=np.min(deltimeph),deltimeband<=np.max(deltimeph))] # need to get rid of extraneous points
    band_inds = find_min_between_lists(deltimeband,deltimeph) # telemetry band goes by time, find where it intersects photons
    # you found the closest indices, now create bins to split the photons around
    #
    prev_ind = 0
    bin_inds = np.empty(len(band_inds))
    for count,current_ind in enumerate(band_inds):
        back_range = current_ind - prev_ind # find total distance from last index
        range_start = current_ind - int(back_range/2) # set back range to half
        bin_inds[count] = range_start
        prev_ind = current_ind
    #
    bin_inds = bin_inds[1::]
    binned_photons = np.array_split(heights,bin_inds.astype(int)) # split photons up
    #
    # for each photon bin, label it with the telemetry band height
    total_band = [a*np.ones(b.shape[0]) for a,b in zip(bandheight,binned_photons)]
    total_band = np.concatenate(total_band)
    # now that each photon is labeled with the height, we can split it up based on our distance bins
    indices = np.arange(0,heights.shape[0])
    bin_num = np.ptp(distances)/binsize # for a bin of x meters, how many of our individual points will it take
    #
    split_bands = np.array_split(total_band,int(bin_num)) # this is the bandheight for each photon, that we now re-bin along with our photon data to our binsize
    split_inds = np.array_split(indices,int(bin_num))
    split_heights = np.array_split(heights,int(bin_num)) # total distance (distances[-1] over the number of points gives us the number of bins we need)
    split_times = np.array_split(deltimeph,int(bin_num))
    #
    # small_bands = [a for a,b in zip(split_bands,split_heights) if np.shape(np.argwhere(np.absolute(b)>60))[0]>1]
    small_bands = [a[np.absolute(b)<100] for a,b in zip(split_bands,split_heights)]
    small_inds = [a[np.absolute(b)<100] for a,b in zip(split_inds,split_heights)]
    # small_bands = [a for a,b in zip(split_bands,split_heights) if np.argwhere(b>400).shape[0]<1]
    # small_inds = [a for a,b in zip(split_inds,split_heights) if np.argwhere(b>400).shape[0]<1]
    #
    next_bands = [a for a in small_bands if a.shape[0]>0]
    next_inds = [a for a in small_inds if a.shape[0]>0]
    #
    # good_inds = [a for a,b in zip(split_inds,split_bands) if np.mean(b)<30 ]
    good_inds = [a for a,b in zip(next_inds,next_bands) if np.mean(b)<30 ]
    final_inds = np.concatenate(good_inds)
    return final_inds

def get_only_land_photons(heights,distances,bandheight,binsize,deltimeph,deltimeband):
    ##### this is a super rough estimation of where there is ocean/ice based on the telemetry window
    ##### input a full photon track and get out just the photons that are within some threshold 
    ##### will sort into horizontal bins, then 
    band_inds = find_min_between_lists(deltimeband,deltimeph) # telemetry band goes by time, find where it intersects photons
    # you found the closest indices, now create bins to split the photons around
    prev_ind = 0
    bin_inds = np.empty(len(band_inds))
    for count,current_ind in enumerate(band_inds):
        back_range = current_ind - prev_ind # find total distance from last index
        range_start = current_ind - int(back_range/2) # set back range to half
        bin_inds[count] = range_start
        prev_ind = current_ind
    bin_inds = bin_inds[1::]
    binned_photons = np.array_split(heights,bin_inds.astype(int)) # split photons up
    # for each photon bin, label it with the telemetry band height
    total_band = [a*np.ones(b.shape[0]) for a,b in zip(bandheight,binned_photons)]
    total_band = np.concatenate(total_band)
    # now that each photon is labeled with the height, we can split it up based on our distance bins
    indices = np.arange(0,heights.shape[0])
    bin_num = binsize / np.diff(distances)[0] # for a bin of x meters, how many of our individual points will it take
    split_bands = np.array_split(total_band,int(distances[-1]/bin_num)) 
    split_inds = np.array_split(indices,int(distances[-1]/bin_num)) 
    split_heights = np.array_split(heights,int(distances[-1]/bin_num)) # total distance (distances[-1] over the number of points gives us the number of bins we need)
    good_inds = [a for a,b in zip(split_inds,split_bands) if np.logical_and(np.mean(b)>180,np.mean(b)<500)]
    final_inds = np.concatenate(good_inds)
    return final_inds


def get_thickness(filename,track = 'gt1r',**kwargs):
    target_directory = kwargs['target_directory']
    all_data = []
    try:
        # fil = h5py.File('/data/looselab/mollie/ATL10_data/'+ATL10_target_dir+'/'+str(filename),'r') # open h5 files
        f2 = nc.Dataset('/data/looselab/mollie/data/'+target_directory+'/'+str(filename)) # open h5 files
        ilat = f2['latitude'][:]
        ilon = f2['longitude'][:]
        thick = f2['ice_thickness'][:]
        deltime = f2['gps_seconds'][:]
        snow_dens = f2['snow_density'][:]
        snow_depth = f2['snow_depth'][:]
        fb = f2['freeboard'][:]
        vtimedelta = np.vectorize(timedelta)
        temp = (datetime(1980,1,6) +  vtimedelta(seconds=ma.getdata(deltime)) - datetime(2018,1,1))
        for count,time in enumerate(temp):
            temp[count]=time.total_seconds()
        current_df = {'filename':filename,'time':temp,'thickness':thick,'snow_dens':snow_dens,'snow_depth':snow_depth,'freeboard':fb,'lat':ilat,'lon':ilon}
        all_data.append(current_df)
        dF = pd.DataFrame().append(all_data)
    except:
        print('file not read')
        current_df = [{'filename':'truncated_file','time':-999,'thickness':-999,'snow_dens':-999,'snow_depth':-999,'freeboard':-999,'lat':-999,'lon':-999}]
        dF = pd.DataFrame().append(current_df)
    return dF

def ocean_data_hist(dist_corrected,h_ph,lats,lons,deltime,vbins,filename):
    #
    hbin=40
    hbins  = np.arange(np.amin(dist_corrected),np.amax(dist_corrected),hbin) # create horizontal bins
    [h,xedges,yedges] = np.histogram2d(dist_corrected,h_ph,bins=(hbins,vbins)) # bin h_ph into hbins and vbins
    #
    past_photons = 0
    splitbins = np.empty(len(h))
    for count,box in enumerate(h):
        current_photons = np.sum(box)
        splitbins[count] = current_photons + past_photons
        past_photons = current_photons+past_photons
    #
    latbins = np.array_split(lats,splitbins.astype(int))
    lonbins = np.array_split(lons,splitbins.astype(int))
    timebins = np.array_split(deltime,splitbins.astype(int))
    distbins = np.array_split(dist_corrected,splitbins.astype(int))
    split_h = np.array_split(h_ph,splitbins.astype('int'))
    #
    # get rid of vertical bins with too few photons in each horizontal bin
    new_vbins = [vbins[1::][a>8] for a in h] # make a list, where for each distance bin we only include the bins with more than 8 photons (from Lu)
    new_h = [a[a>8] for a in h] 
    #
    # get rid of the horizontal bins that had NO good vertical bins
    lengths = [a.shape[0] for a in new_vbins] ## figure out how many photons made it in
    final_vbins = np.array(new_vbins,dtype='object')[(np.array(lengths).astype(int)>1)] ## get rid of the ones that had absolutely no photons
    final_h = np.array(new_h,dtype='object')[(np.array(lengths).astype(int)>1)]
    nextlons = np.array(lonbins[1::],dtype='object')[np.array(lengths).astype(int)>1]
    nextlats = np.array(latbins[1::],dtype='object')[np.array(lengths).astype(int)>1]
    nexttime = np.array(timebins[1::],dtype='object')[np.array(lengths).astype(int)>1]
    nextdist = np.array(distbins[1::],dtype='object')[np.array(lengths).astype(int)>1]
    nexth = np.array(split_h[1::],dtype='object')[np.array(lengths).astype(int)>1]
    #
    # final_lats = [np.mean(a) for a in nextlats]
    # final_lons = [np.mean(a) for a in nextlons]
    # final_time = [np.mean(a) for a in nexttime]
    #
    # normalize the histogram for each horizontal bin
    h_norm = [(a-np.min(a)) / np.ptp(a) for a in final_h]
    #get rid of the sections with only one photon
    #
    # identify surface for each horizontal bin        
    surface_index = [np.argmax(a) for a in h_norm] # maximum binned photons show you the surface return index    surface = [b[a] for b,a in zip(final_vbins,surface_index)]
    surface = [b[a] for b,a in zip(final_vbins,surface_index)]
    # split_list = tool.find_min_between_lists(xedges,dist_corrected) # find where each hbin is in the photon data
    #
     # split the photons up into the hbins
    #
    new_ys = [a-b for a,b in zip(nexth,surface)] # subtract the surface from the relevant bin from those photons
    final_ys = np.concatenate(new_ys) # tape the bins back together
    final_xs = np.concatenate(nextdist)
    final_lats = np.concatenate(nextlats)
    final_lons = np.concatenate(nextlons)
    final_time = np.concatenate(nexttime)
    #
    #
    current_df = [{'filename':filename,'time':final_time,'heights':final_ys,'dist':final_xs,'lat':final_lats,'lon':final_lons}]
    return current_df


def match_ATL03_ATL10_file_num(ATL10_future,**kwargs):
    # given a single ATL10 future, and the list of times for the ATL03 futures, match up the ATL10 and ATL03 files
    # ATL10future will be l[n].result()
    ATL10lats = ATL10_future['lat'][0]
    ATL10times = ATL10_future['time'][0]
    ATL10times = ATL10times[ATL10lats<=76]
    ATL10lats = ATL10lats[ATL10lats<=76]
    if ATL10lats.shape[0]>0:
        ATL10_timerange = np.array([np.min(ATL10times),np.max(ATL10times)]).astype('float64')
        ATL10_min_diffs = [np.abs(a-ATL10_timerange[0]) for a in kwargs['list_of_times']] # find every ATL03 averaged time
        ATL10_max_diffs = [np.abs(a-ATL10_timerange[1]) for a in kwargs['list_of_times']] # same, but for the max ATL10 time
        full_min_range = list_of_lists(ATL10_min_diffs) # thel and one for the end
        full_max_range = list_of_lists(ATL10_max_diffs)
        first_min = np.argmin(full_min_range)
        if full_min_range[first_min] < kwargs['time_constraint']:
            min_ATL03_ind = np.trunc(first_min/2) # ta
        else:
            min_ATL03_ind = -99999
    else:
        min_ATL03_ind = -999
    return min_ATL03_ind



def match_ATL03_ATL10_data(ATL03future,**kwargs):
    # given the data from one ATL10future, and the data from the matching ATL03future, give back a dataframe with all the relevant stuff:
    # thickness, snow density, snow depth, and total photon returns
    dF = pd.DataFrame()
    #
    ATL10lons = kwargs['ATL10data']['lon'][0] # pull out the ATL10 values
    ATL10lats =  kwargs['ATL10data']['lat'][0]
    ATL10times =  kwargs['ATL10data']['time'][0]
    ATL10thick =   kwargs['ATL10data']['thickness'][0]
    snow_dens =   kwargs['ATL10data']['snow_dens'][0]
    snow_depth =   kwargs['ATL10data']['snow_depth'][0]
    freeboard =   kwargs['ATL10data']['freeboard'][0]
    #
    ATL10lons = ATL10lons[ATL10lats<=76] # only take relevant latitudes for Chukchi
    ATL10times = ATL10times[ATL10lats<=76]
    ATL10thick = ATL10thick[ATL10lats<=76]
    snow_dens = snow_dens[ATL10lats<=76]
    snow_depth = snow_depth[ATL10lats<=76]
    freeboard = freeboard[ATL10lats<=76]
    ATL10lats = ATL10lats[ATL10lats<=76]
    #
    for index in np.arange(len(ATL03future)):
        try:
            ATL03data = ATL03future.iloc[index]
            ATL03lats = ATL03data['lat']
            ATL03height = ATL03data['heights']
            ATL03lons = ATL03data['lon']
            ATL03lats = ATL03data['lat']
            ATL03times = ATL03data['time']
            ATL03dist = ATL03data['dist']
        #
            minpoint = np.array([ATL03lats[0],ATL03lons[0]])
            maxpoint = np.array([ATL03lats[-1],ATL03lons[-1]])
            total_dist = distance.distance(minpoint,maxpoint).km
            total_dist = total_dist*1000
            ATL03_bin_num = np.around(total_dist/kwargs['binsize'])
        #
            multarr = np.linspace(0,int(ATL03_bin_num),num=int(ATL03_bin_num)+1) # make a vector that just has one number for each bin
            intarr = np.ones([ATL03height.shape[0]]) # a vector of ones that is the total number of ATL03 data points
            intarr1 = np.array_split(intarr,ATL03_bin_num) # split all the ATL03 ones into the number of bins we picked out
            intarr2 = [a*b for a,b in zip(multarr,intarr1)] # multiply each ATL03 bin by it's bin number - so we have n bins filled with some amount of entries that are all n
            interp = NearestNDInterpolator(list(zip(ATL03lons[0::1000],ATL03lats[0::1000])),np.concatenate(intarr2)[0::1000]) 
            print('starting interpolation')
            closest_ATL03 = interp(ATL10lons,ATL10lats) # evaluate the ATL10 points for the closest batch of ATL03, each ATL10 will be given an ATL03 bin number
            print('interpolation finished')
            temp= np.vstack([closest_ATL03,ATL10thick]).T # tape together the ATL03 bin that is closest to the ATL10 point, along with the thickness of that point
        #
            # temp = temp[temp[:, 0].argsort()] # sort so that the ATL03 bins are in order
        #
            var = np.squeeze(np.argwhere(np.diff(temp[:,0]))) #there will likely be multiple thicknesses for one ATL03 bin, find where the ATL03 closest bin switches
            binned_ind = np.array_split(np.squeeze(temp.data[:,0]),var+1) # group the closest index into the relevant bins
            binned_thickness = np.array_split(np.squeeze(temp.data[:,1]),var+1) # group the thicknesses into the relevant bins
            binned_dens = np.array_split(snow_dens,var+1) # group the 
            binned_depth = np.array_split(snow_depth,var+1) # group
            binned_fb = np.array_split(freeboard,var+1) # group
            meanthick = [np.mean(a) for a in binned_thickness]
            meanind = [np.mean(a) for a in binned_ind]
            meandens = [np.mean(a) for a in binned_dens]
            meandepth = [np.mean(a) for a in binned_depth]
            meanfb = [np.mean(a) for a in binned_fb]
        #   
            #now in thickvar we have the bin number and thickness for each bin with ice
            #same for meandens and meandepth - they just exist for the bins with ice
        #
            #heightvar, latvar,lonvar,timvar have the TOTAL number of bins that exist for the ATL03
        #
            heightvar =  np.array_split(ATL03height,ATL03_bin_num) # go through and split up our ATL03 variables into the bins
            latvar =  np.array_split(ATL03lats,ATL03_bin_num)
            lonvar =  np.array_split(ATL03lons,ATL03_bin_num)
            timvar = np.array_split(ATL03times,ATL03_bin_num)
            distvar = np.array_split(ATL03dist,ATL03_bin_num)
            meandist = [np.mean(a) for a in distvar]
        #
            current_df = [{'filename':kwargs['ATL10data']['filename'][0],'time':timvar,'lat':latvar,'lon':lonvar,'dist':distvar,'heights':heightvar,'match_index':np.array(meanind),'thickness':np.array(meanthick),'snow_dens':np.array(meandens),'snow_depth':np.array(meandepth),'meandist':np.array(meandist),'freeboard':np.array(meanfb)}]
            dF = dF.append(current_df)
        except:
            current_df = [{'filename':'tiny dataset!','time':-999,'lat':-999,'lon':-999,'dist':-999,'heights':-999,'match_index':-999,'thickness':-999,'snow_dens':-999,'snow_depth':-999,'meandist':-999,'freeboard':-999}]
            dF = dF.append(current_df)
    dF = dF.set_index([pd.Index(np.arange(len(dF)))])
    return dF


def twt(total_dataframe):
    # given a data frame with binned lats,lons,ice,snow, and photons, find the two way transmittance
    dF = pd.DataFrame()
    for ind in np.arange(len(total_dataframe)):
        print(ind)
        below = []
        above = []
        avglat =[]
        avglon = []
        avgtime = []
        try:
            for count,index in enumerate(total_dataframe['match_index'][ind]): # for each horizontal bin, along the track
                # need to check on the below/above param
                avglat.append(np.mean(total_dataframe['lat'][ind][int(index)])) # average the lats and lons so we can report one location and make plotting simpler
                avglon.append(np.mean(total_dataframe['lon'][ind][int(index)]))
                avgtime.append(np.mean(total_dataframe['time'][ind][int(index)]))
                # ice_depth = total_dataframe['thickness'][0][count] - total_dataframe['snow_depth'][0][count]
                ice_depth = total_dataframe['thickness'][ind][count] - total_dataframe['freeboard'][ind][count]
                below.append(total_dataframe['heights'][ind][int(index)]<-ice_depth) # take the heights from that bin and find where they are lower than the ice
                above.append(total_dataframe['heights'][ind][int(index)]>total_dataframe['freeboard'][ind][count]) # find where the heights are above the ice (>0)
                # above.append(total_dataframe['height'][0][int(index)]>total_dataframe['snow_depth'][0][count]) # find where the heights are above the ice (>0)
            twt = [np.sum(a)/np.sum(b) for a,b in zip(above,below)] # find twt for each bin
            current_df = [{'time':np.array(avgtime),'lon':avglon,'lat':avglat,'thickness':total_dataframe['thickness'][ind],'twt':np.array(twt),'snow_dens':total_dataframe['snow_dens'][ind],'snow_depth':total_dataframe['snow_depth'][ind]}]
            dF = dF.append(current_df)
        except:
            current_df = [{'filename':'tiny dataset!','time':-999,'lat':-999,'lon':-999,'dist':-999,'heights':-999,'match_index':-999,'thickness':-999,'snow_dens':-999,'snow_depth':-999,'meandist':-999,'freeboard':-999}]
            dF = dF.append(current_df)
    # current_df = {'time':np.array(avgtime),'lat':avglat,'lon':np.array(avglon),'twt':twt,'thickness':total_dataframe['thickness'],'snow_dens':total_dataframe['snow_dens'],'snow_depth':total_dataframe['snow_depth']}
    # dF = pd.DataFrame(data=current_df)
    # dF.to_csv('output.csv',index=False)
    dF = dF.set_index([pd.Index(np.arange(len(dF)))])
    return dF


def find_min_between_lists(query,base):
    interval = base.shape[0]/(query.shape[0]) ## take the number of base points that go within a query
    minval = np.empty(query.shape[0])
    for count,val in enumerate(query):
        count
        if (int(count*interval)-math.ceil(interval))<0:
            start=0
            end = int(count*interval)+2*math.ceil(interval)
            var = np.absolute(base[start:end]-val)
            var1 = var.argmin()
            minval[count] = var1+start
        elif (int(count*interval)+math.ceil(interval))>base.shape[0]:
            print('special end case')
            start = int(minval[count-1])
            end = int(base.shape[0])
            var = np.absolute(base[start:end]-val)
            var1 = var.argmin()
            minval[count] = var1+start
        else:
            start = int(minval[count-1])
            end = int(start+2*math.ceil(interval))
            var = np.absolute(base[start:end]-val)
            var1 = var.argmin()
            minval[count] = var1+start
    return minval








def avg_of_lists(list_of_arrays): # simple function to get the average of the list of arrays produced in both the futures
    import numpy as np
    mean = []
    for array in list_of_arrays:
        mean.append(np.nanmean(array))
    mean = np.nanmean(mean)
    return mean

def list_of_lists(list_of_arrays):
    final_list = np.array([])
    for arr in list_of_arrays:
        final_list = np.append(final_list,np.array(arr))
    return final_list

def list_of_lists1(list_of_arrays):
    final_list = np.array([])
    for arr in list_of_arrays:
        final_list = np.append(final_list,arr[0])
    return final_list

def max_of_lists(list_of_arrays):
    maxval = []
    for array in list_of_arrays:
        try:
            maxval.append(np.max(array))
        except:
            i=1
    maxval = np.max(maxval)
    return maxval

def min_of_lists(list_of_arrays):
    minval = []
    for array in list_of_arrays:
        try:
            minval.append(np.min(array))
        except:
            i=1
    minval = np.min(minval)
    return minval

def min_max_of_lists(list_of_arrays):
    min_list = np.array([])
    max_list = np.array([])
    for arr in list_of_arrays:
        try:
            min_list = np.append(min_list,np.nanmin(arr))
            max_list = np.append(max_list,np.nanmax(arr))
        except:
            i=1
    final_min = np.min(min_list)
    final_max = np.max(max_list)
    return final_min,final_max


# def give_me_the_strong_beams(time):
#     if time < 4 #12/28/2018:

def get_ATL03_profiles(ATL03future,binsize):
    ##### will sort into horizontal bins, then 
    dF = pd.DataFrame()
    for index in np.arange(len(ATL03future)):
        ATL03data = ATL03future.iloc[index] # take one continuous transect of photon returns
        ATL03height = ATL03data['heights']
        ATL03dist = ATL03data['dist']
        ATL03time = ATL03data['time']
        bin_num = np.ptp(ATL03dist)/binsize # split that transect into your binsize
        if int(bin_num)==0: # if the transect is smaller than the binsize, take the whole transect
            split_heights = ATL03height
            split_dists = ATL03dist
            split_times = ATL03time
        else:
            split_heights = np.array_split(ATL03height,int(bin_num)) 
            split_dists = np.array_split(ATL03dist,int(bin_num))
            split_times = np.array_split(ATL03time,int(bin_num))
        current_df = [{'dists':split_dists,'heights':split_heights,'times':split_times}]
        dF = dF.append(current_df) # returns a data frame with each continuous transect broken up into your binsize
    dF = dF.set_index([pd.Index(np.arange(len(dF)))])
    return dF




def ATL03_deconvolve(Sm,vbins,tep_hist,tep_hist_time):
    from scipy import signal
    ## deconvolve signal
    # Fz = some response model
    # Sz = scipy.signal.deconvolve(Sm,F)
    # fzfile = h5py.File(fzfile) # load in impulse response function, currently using the wrong function or something
    # fzdepth = fzfile['pce1_s/fit_hist/bine'][:]
    # fzdata = fzfile['pce1_s/fit_hist/hist'][:]
    #
    # fzdepth = fzfile['pce1_s/fit_hist/bine'][:]-39
    # fzdepth = fzdepth*-1
    # fzdata = fzfile['pce1_s/fit_hist/hist'][:]
    # fzdata=fzdata[0:-1]
    #
    # vbins = fzdepth
    # hist,bin_edges = np.histogram(Sm,vbins*-1) # fit our signal to the same bins as the impulse function
    #
    # h_norm = (hist - np.amin(hist)) / np.ptp(hist) # normalize signal

    #############################
    # [h1,bin_edges] = np.histogram(Sm,vbins) # bin h_ph into hbins and vbins

    # h_norm1 = (h1-np.min(h1)) / np.ptp(h1)
    # h_norm = h1/np.sum(h1) # this is the norm as defined in the ATL03 ATBD
    tep_dist = -1*(299792458 * tep_hist_time)/2 # dist calculated from time, as in Lu (something)

    # griddedfz = griddata(tep_dist,tep_hist,bin_edges[1::]) # put tep on same vertical scale
    #
    griddedfz=tep_hist

    firstmax = np.argmax(griddedfz) # find first response

    tempvar = np.zeros(griddedfz.shape[0])

    interval_2m = 2/(np.absolute(np.diff(tep_dist[0:2])))

    tempvar[0:firstmax-int(interval_2m)] = griddedfz[0:firstmax-int(interval_2m)] # to find second local max, get rid of first
    tempvar[firstmax+int(interval_2m)::] = griddedfz[firstmax+int(interval_2m)::]

    tep_surface = tep_dist[np.argmax(tempvar)]
    fzdepth = tep_dist - tep_surface

    # changing depths of tep data to adjust for the surface return
    fzdata = tep_hist
    # fzdepth = bin_edges-bin_edges[tep_surface[0][0]]

    [h1,bin_edges] = np.histogram(Sm,np.flip(fzdepth)) # bin h_ph into hbins and vbins
    h_norm = h1/np.sum(h1)
    h_norm = np.flip(h_norm)

    # putting tep data and measured signal over same range
    # fzdata = fzdata[np.logical_and(fzdepth[1::]>=np.min(bin_edges[1::]),fzdepth[1::]<=np.max(bin_edges[1::]))]
    # h_norm = h_norm[np.logical_and(bin_edges[1::]>=np.min(fzdepth[1::]),bin_edges[1::]<=np.max(fzdepth[1::]))]
    # fzdepth = fzdepth[np.logical_and(fzdepth>=np.min(bin_edges),fzdepth<=np.max(bin_edges))]
    # bin_edges = bin_edges[np.logical_and(bin_edges>=np.min(fzdepth),bin_edges<=np.max(fzdepth)+.002)]
    # # get rid of zeros, so they don't mess up the logs
    # fzdata=fzdata[h_norm>0]
    # fzdepth = fzdepth[1::][h_norm>0]
    # bin_edges = bin_edges[1::][h_norm>0]
    # h_norm=h_norm[h_norm>0]

    # bin_edges = bin_edges[~np.isnan(fzdata_var[1::])]
    # fzdepth_var = fzdepth_var[~np.isnan(fzdata_var)]
    # Bm = Bm[~np.isnan(fzdata_var[1::])]
    # Bm = np.log(Bm)
    # fzdata_var = fzdata_var[~np.isnan(fzdata_var)]
    # fzdata_var = np.log(fzdata_var)
    # fzdata_var[np.isnan(fzdata_var)] = np.nanmin(fzdata_var)
    # #
    # h_norm = np.log(h_norm)

    starting_bin = np.argmax(h_norm)-100 # from Lu, we only want to include a smaller depth range, starting with our surface returtn
    ending_bin = starting_bin + 1100 # here i've picked 10 meters, so thats 200 .05 m bins
    length = np.arange(starting_bin,ending_bin+1,1) # i just want something of the appropriate length to iterate over
    # final versions of all our signals
    Bm = h_norm[starting_bin:ending_bin+1] 
    bin_edges = np.flip(bin_edges)
    bin_edges = bin_edges[starting_bin:ending_bin+1]
    fzdata_var = fzdata[starting_bin:ending_bin+2]
    fzdepth_var = fzdepth[starting_bin:ending_bin+2]
    # getting rid of all the nans
    # length = length[~np.isnan(fzdata_var[1::])]
    # length = length[1::]


    Fz_matrix = np.zeros([length.shape[0],length.shape[0]]) # minus one because we neglect the influence far from surface (lu)
    #
    # fzdata_var[fzdata_var<-12.5] = Bm[fzdata_var<-12.5]
    for count,val in enumerate(length): ## REMAINING ISSUE: LAST ROW OF FZ IS IDENTICAL TO PREVIOUS ROWS
        count
        if count+1 == length.shape[0]:
            Fz_matrix[count,0:count+1] = fzdata_var[int(count+2):0:-1]
        elif count+2 == length.shape[0]:
            Fz_matrix[count,0:count+1] = fzdata_var[int(count+1):0:-1]
        else:
            Fz_matrix[count,0:count+2] = fzdata_var[int(count+1)::-1]
    #
    #
    Bc = np.dot(np.linalg.inv(Fz_matrix).T,Bm)
    return Bc,fzdepth_var,bin_edges,fzdata,fzdepth,Bm




        
def bb(Sm, Bc,z):
    Kd,b = np.polyfit(z[1::],Bc,1) # find attenuation coefficient from linear fit 
    #
    #
    # sigma = 
    # theta =
    # Bs = (.0209/(4*3.15*(sigma**2)*(np.cos(theta)**4))) * np.exp(-(np.tan(theta)**2)/(2*sigma**2))
    #
    # C = Sm[0]/ Bs
    # eventually, C goes in the denominator below
    #
    #
    bb = (Bc * np.exp(2*Kd*z[1::]) ) / (.32 * (.98)**2 )
    #
    #
    # Rrs = np.sum(Sz/C)
    # t = .98
    # Bp = .32
    # bb = ((2*m**2) * Kd * Rrs) / (Bp*(t**2))
    # bbw = 1.05 * 10**(-3)
    # bbp = bb - bbw
    return bb