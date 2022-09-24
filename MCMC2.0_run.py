#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:07:17 2022
W
@author: giovanni
"""

################## DEFAULT SETTING ##################
import os,sys,glob,warnings,argparse
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('../')
from config import path2projects,path2data
sys.path.append(path2projects)
# import miscellaneus
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
sys.path.append(path2projects+'/MCMC_analysis')
from mcmc import MCMC,run
from miscellaneus import get_Av_list
# import show_priors
import numpy as np
import pandas as pd
import pickle

from PyAstronomy import pyasl
def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

sdj = pyasl.SpecTypeDeJager()
pd.set_option('display.max_columns', 500)



def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_number',type=int,default=0,help='Use to select IDs in conjunction with batch_size using the following formula: [batch_number*batch_size:(batch_number+1)*batch_size]. Default=0')
    parser.add_argument('-batch_size',type=int,default=200,help='Use to select IDs in conjunction with batch_number using the following formula: [batch_number*batch_size:(batch_number+1)*batch_size]. Default=200')
    parser.add_argument('-ID_list', type=int,nargs='+',default=[],help='ID list of selected targets for this run. Default=[]')
    parser.add_argument('-workers',default=None,type=int,help='Number of workers allowed. Default=None')
    parser.add_argument('-override', action='store_true',help='Override existing sampler. Default=False')
    parser.add_argument('-dfname',type=str,default='ONC_combined_df',help='name for the dataframe. Default=ONC_combined_df')
    parser.add_argument('-isoname',type=str,default='interpolated_isochrones',help='name for the pickle interpolated isochones file. Default=interpolated_isochrones')
    parser.add_argument('-savedir',type=str,default='',help='name for the save dir. Default=')
  
    parser.add_argument('-parallelize_runs', action='store_true',help='Parallelize runs. Default=False')
    parser.add_argument('-parallelize_sampler', action='store_true',help='Parallelize samplers. Default=False')
    parser.add_argument('-progress', action='store_true',help='Show single run progress. Default=False')
    parser.add_argument('-show_test', action='store_true',help='Show input photometry of target. Default=False')
    
    parser.add_argument('-use_Av_KDE', action='store_true',help='Use the Av KDE. Default=False')
    parser.add_argument('-use_Age_KDE', action='store_true',help='Use the Age KDE. Default=False')
    parser.add_argument('-use_mass_KDE', action='store_true',help='Use the mass KDE. Default=False')
 
    parser.add_argument('-Rv',type=float,default=3.1,help='Rv value for extinction. Default=3.1')
    parser.add_argument('-mag_label_list',type=str,nargs='+',default=['m336','m439','m656','m814','m435','m555','m658','m775','m850','m110','m160','m130','m139'],help='list of magnitudes for the isochrone fit. Default=[m336,m439,m656,m814,m435,m555,m658,m775,m850,m110,m160,m130,m139]')
    parser.add_argument('-mags2fit',type=str,nargs='+',default=['m336','m439','m656','m814','m435','m555','m658','m775','m850','m110','m160','m130','m139'],help='list of magnitudes for the MCMC run. Default=[m336,m439,m656,m814,m435,m555,m658,m775,m850,m110,m160,m130,m139]')
    parser.add_argument('-colors2fit',type=str,nargs='+',default=[],help='list of colors for the MCMC run. Default=[]')
    
    parser.add_argument('-logMass_range', type=float,nargs='+',default=[-2,1],help='Parameter range for the logMass variable. Default=[-2,1]')
    parser.add_argument('-logAv_range', type=float,nargs='+',default=[-1,2],help='Parameter range for the logAv variable. Default=[-1,2]')
    parser.add_argument('-logAge_range', type=float,nargs='+',default=[0,4],help='Parameter range for the logAge variable. Default=[0,4]')
    parser.add_argument('-logSPacc_range', type=float,nargs='+',default=[-5,1],help='Parameter range for the logSPacc variable. Default=[-5,1]')
    parser.add_argument('-Parallax_range', type=float,nargs='+',default=[0.02,6],help='Parameter range for the Parallax variable. Default=[0.02,6]')
        
    parser.add_argument('-nwalkers_ndim_niters', type=int,nargs='+',default=[100,5,20000],help='nwalkers, ndim and niters for the MCMC. Default=[100,5,15000]')

    parser.add_argument('-err',default=None,help='Fixed error on the photometry for the MCMC run. Default=None')
    parser.add_argument('-err_max',default=0.1,help='Maximum error to accept photometry in the MCMC run. Default=0.05')
    parser.add_argument('-err_min',default=0.001,help='Minimum error to accept photometry in the MCMC run. Default=0.001')

    parser.add_argument('-smooth',default=0.001,help='Smooth parameter in the ND isochronal fit. Default=0.001')
    parser.add_argument('-method',default='linear',help='method parameter in the ND isochronal fit. Default=linear')

    args = parser.parse_args()
    return(args)

if __name__ == '__main__':
    args=get_opt()
    dfname=args.dfname
    batch_number=args.batch_number
    batch_size=args.batch_size
    ID_list=args.ID_list
    Rv=args.Rv
    mag_label_list=args.mag_label_list
    mags2fit=args.mags2fit
    colors2fit=args.colors2fit
    workers=args.workers
    savedir=args.savedir
    isoname=args.isoname
    
    override=args.override
    parallelize_runs=args.parallelize_runs
    parallelize_sampler=args.parallelize_sampler
    progress=args.progress
    show_test=args.show_test

    use_Av_KDE=args.use_Av_KDE
    use_Age_KDE=args.use_Age_KDE
    use_mass_KDE=args.use_mass_KDE
        
    logMass_range=args.logMass_range
    logAv_range=args.logAv_range
    logAge_range=args.logAge_range
    Parallax_range=args.Parallax_range
    logSPacc_range=args.logSPacc_range
    nwalkers_ndim_niters=args.nwalkers_ndim_niters
    
    err=args.err
    err_max=args.err_max
    err_min=args.err_min
    
    smooth=args.smooth
    method=args.method
    
    ########################################## Importing catalogs ################################################
    print('> Importing catalogs')
    # Besançon_plus_GAIA_ONCFOV_df=pd.read_csv(path2data+'/Giovanni/MCMC_analysis/catalogs/Besançon_plus_GAIA_ONCFOV.csv')
    Strampelli_df=pd.read_hdf(path2data+'/Giovanni/MCMC_analysis/%s.hdf'%dfname,'df')
    # Av_df=pd.read_csv(path2data+'/Giovanni/MCMC_analysis/catalogs/Av.csv')
    # Age_df=pd.read_csv(path2data+'/Giovanni/MCMC_analysis/catalogs/Age.csv')
    # Mass_df=pd.read_csv(path2data+'/Giovanni/MCMC_analysis/catalogs/Mass.csv')
    if len(ID_list)==0:
        ID_list=np.sort(Strampelli_df.avg_ids.unique())[batch_number*batch_size:(batch_number+1)*batch_size]
    if savedir=='':savedir=path2data+"/Giovanni/MCMC_analysis/samplers"
    if not override:
        file_list=sorted(glob.glob(savedir+"/*ID_*"))
        fileID_list=[]
        for file in file_list:
            fileID_list.append(int(float(file.split('_')[-1])))
            # print(file.split('_')[-1])
        ID_list = [x for x in ID_list if x not in fileID_list]
    print('> ID_list:\n',ID_list)
    # iso_df=pd.read_hdf(path2data+"/Giovanni/MCMC_analysis/bt_settl_AGSS2009_isochrones_new.h5",'df')
    # iso_df=pd.read_hdf(path2data+"/Giovanni/Synthetic Photometry/%s.h5"%isoname,'df')

    ########################################## ISO interpolations ################################################
    print('> ISO interpolations')
    mag_label_list=np.array(['m336','m439','m656','m814','m435','m555','m658','m775','m850','m110','m160','m130','m139'])
    
    # node_label_list= list(mag_label_list)+['teff','logL','logg','logLacc','logMacc']
    # node_list=[iso_df[label].values.ravel() for label in node_label_list]
    # x=np.log10(iso_df['mass'].values).ravel()
    # y=np.log10(iso_df.index.get_level_values('Age').values).ravel()
    # z=iso_df.index.get_level_values('logAcc').values.ravel()
    
    # interp_btsettl=miscellaneus.interpND([x,y,z,node_list],method=method,showplot=False,smooth=smooth,z_label=node_label_list,workers=workers)
    with open(path2data+"/Giovanni/Synthetic Photometry/%s.pck"%isoname, 'rb') as file_handle:
        interp_btsettl = pickle.load(file_handle)

    ############################################ Extinction dictionary ####################################################
    print('> Extinction dictionary')
    filter_label_list=np.array(['F336W','F439W','F656N','F814W','F435W','F555W','F658N','F775W','F850LP','F110W','F160W','F130N','F139M'])
    Av_dict=get_Av_list(filter_label_list,verbose=False,Rv=Rv)
    
    ############################################ Saturation dictionary ####################################################
    print('> Saturation dictionary')
    sat_list=['N/A','N/A','N/A','N/A',16,15.75,12.25,15.25,14.25,0,0,10.9,9.5]
    sat_dict=dict(zip(mag_label_list,sat_list))
    
    ############################################ prior on distance 4 not Gaias ####################################################
    print('> Priors 4 not Gaias ')
    # bw_method=None
    # parallax_KDE=show_priors.kernel_prior(Besançon_plus_GAIA_ONCFOV_df.parallax,xlabel='parallax',nbins=50,bw_method=0.1,bandwidth2fit=np.linspace(0.01, 0.1, 1000),xlogscale=False,density=True,kernel='gaussian',return_prior=True,showplot=False)
    with open(path2data+"/Giovanni/MCMC_analysis/catalogs/parallax.pck", 'rb') as file_handle:
        parallax_KDE = pickle.load(file_handle)

    if use_Av_KDE: 
        # Av_KDE=show_priors.kernel_prior(Av_df.Av,xlabel='logAv',nbins=10,bw_method=0.07,bandwidth2fit=np.linspace(0.1, 1, 1000),xlogscale=False,density=True,kernel='gaussian',return_prior=True,showplot=False)
        with open(path2data+"/Giovanni/MCMC_analysis/catalogs/av.pck", 'rb') as file_handle:
                Av_KDE = pickle.load(file_handle)
    else: Av_KDE=None
    if use_Age_KDE: 
        # Age_KDE=show_priors.kernel_prior(Age_df.Age,xlabel='logAge [Myr]',nbins=30,bw_method=0.05,bandwidth2fit=np.linspace(0.1, 1, 1000),xlogscale=False,density=True,kernel='gaussian',return_prior=True,showplot=False)
        with open(path2data+"/Giovanni/MCMC_analysis/catalogs/age.pck", 'rb') as file_handle:
            Age_KDE = pickle.load(file_handle)

    else: Age_KDE=None
    if use_mass_KDE: 
        # mass_KDE=show_priors.kernel_prior(Mass_df.Mass,xlabel='logMass',nbins=30,bw_method=0.04,bandwidth2fit=np.linspace(0.01, 0.1, 1000),xlogscale=False,density=True,kernel='gaussian',return_prior=True,showplot=False)
        with open(path2data+"/Giovanni/MCMC_analysis/catalogs/mass.pck", 'rb') as file_handle:
            mass_KDE = pickle.load(file_handle)

    else: mass_KDE=None
    ############################################ MCMC Run ####################################################
    print('> MCMC Run')
    mcmc=MCMC(interp_btsettl,mag_label_list,sat_dict,Av_dict,parallax_KDE=parallax_KDE,Av_KDE=Av_KDE,Age_KDE=Age_KDE,mass_KDE=mass_KDE,logMass_range=logMass_range,logAge_range=logAge_range,logSPacc_range=logSPacc_range,logAv_range=logAv_range,Parallax_range=Parallax_range,nwalkers_ndim_niters=nwalkers_ndim_niters,err=err,err_max=err_max,err_min=err_min,parallelize_runs=parallelize_runs,parallelize_sampler=parallelize_sampler,show_test=show_test,progress=progress,simulation=False,thin=None,discard=None,blobs=True,mags2fit=mags2fit,colors2fit=colors2fit,workers=workers,savedir=savedir)
    run(mcmc,Strampelli_df,ID_list)