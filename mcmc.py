#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:34:01 2021

@author: giovanni
"""

import sys,emcee,os
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append('./')
from config import path2data
import mcmc_utils#,show_priors
# from priors import normal_dist
import numpy as np
from astropy.table import QTable
from multiprocessing import Pool
# import pandas as pd
import concurrent.futures
from tqdm import tqdm
from itertools import repeat
from astropy import units as u
from scipy.stats import multivariate_normal
# from scipy.stats.kde import gaussian_kde
from scipy.stats import skewnorm
# from twopiece.scale import tpnorm
import bz2
import pickle
import multiprocessing as mp

# global data_mag
# data_mag= []

class MCMC():
    ############
    #Main body #
    ############
    
    def __init__(self,interp,mag_label_list,sat_dict,AV_dict,savedir='samplers',parallax_KDE=None,Av_KDE=None,Age_KDE=None,mass_KDE=None,parallax=2.487562189054726,eparallax=0.030559732052269695,truths=[None,None,None,None,None],discard=None,thin=None,logMass_range=[-3,1],logAv_range=[2,1],logAge_range=[-2,2],logSPacc_range=[-6,10],Parallax_range=[0.01,6],nwalkers_ndim_niters=[50,3,10000],mu_sigma_t=[2,1],ID_label='avg_ids',Teff_label='Teff',eTeff_label='eTeff',logLacc_label='logLacc',elogLacc_label='elogLacc',parallax_label='parallax',eparallax_label='parallax_error',WCaII_label='WCaII',backend_sampler=False,sampler_dir_path='/Giovanni/MCMC_analysis/samplers/',workers=None,err=None,err_max=0.1,err_min=0.001,r2=4,gaussian_kde_bw_method=0.1,blobs=False,nmag2fit=1,magnitude_fit=False,color_fit=False,magnitude_color_fit=False,show_test=True,progress=True,parallelize_runs=False,parallelize_sampler=False,simulation=False,mags2fit=[],colors2fit=[],check_acor=100,Fconv=100,conv_thr=0.01,ndesired=2000):
        '''
        This is the initialization step of the MCMC class. The MCMC can be run to fit 3 varables at the time. The variables for the fit are:
        [logMass, logAv, Age]. 

        Parameters
        ----------
        interp : list
            dictionary of interpolation functions.
        mag_label_list : list
            list of magnitude labels. 
        sat_dict : dict
            dict of saturation values. 
        AV_dict : list
            dictionary of filter extinction values.
        parallax_KDE: func
            Prior for the parallax of the star when we don't have a mesurment from Gaia. Default None
        Av_KDE: func
            Prior for the Av of the star when we don't have a distance mesurment from Gaia. Default None
        Age_KDE: func
            Prior for the Age of the star when we don't have a distance mesurment from Gaia. Default None
        mass_KDE: func
            Prior for the mass of the star when we don't have a distance mesurment from Gaia. Default None
        parallax : float, optional
            parallax of the target. The default is 402 for the ONC.
        eparallax : float, optional
            error on the parallax of the target. The default is 5 for the ONC.
        truths : list, optional
            list of truth values for the star. It's used when simulation is enable or to check versus know values. The default is [None,None,None].
        vlabel_Rv : list, optional
            list of coded Var lables and reddening labes. 
            vlabel:
                0: No priors; 1: kroupa; 2: singles; 3: systems; 4: Teff fit;
            Rv:
                a: Rv=3.1, b: Rv=5.5 (Fang et al. 2021)
            The default is ['3','a'].
        discard:
            Discard the first discard steps in the chain as burn-in. If None, discard the first half.
            The default is None.
        thin:
            Use only every thin steps from the chain. The returned estimate is multiplied by thin so the estimated time is in units of steps, not thinned steps.
            If None, use the minimum of the autocrlation time to thin. The default is None.
        logMass_range : list, optional
            minimum, maximum logMass range for walkers to explore in logspace. The default is [0-3,1].
        logAv_range : list, optional
            minimum, maximum extinction range for walkers to explore in logspace. The default is [-2,1].
        logAge_range : list, optional
            minimum, maximum age range for walkers to explore in logspace. The default is [-2,2].
        logSPacc_range : list, optional
            minimum, maximum logSPacc range for walkers to explore in logspace. The default is [-6,0].
        Parallax_range : list, optional
            minimum, maximum parallax range for walkers to explore in logspace. The default is [1,5].
        nwalkers_ndim_niters : list, optional
            list of number of walker, dimension (number of variables) and number of steps for the MCMC run. The default is [50,3,10000].
        ID_label : str, optional
            label identifing the IDs in the avg dataframe. The default is 'avg_ids'.
        Teff_label : str, optional
            label identifing the Teff in the avg dataframe. The default is 'Teff'.
        eTeff_label : str, optional
            label identifing the error on Teff in the avg dataframe. The default is 'eTeff'.
        parallax_label : str, optional
            label identifing the parallax of the star in the avg dataframe. The default is 'parallax'.
        eparallax_label : str, optional
            label identifing the error on parallax of the star in the avg dataframe. The default is 'eparallax'.
        backend_sampler: bool, optional
            if True, save the entire sampler for each ID in the run. Default is False.
        path2backend : str, optional
            path to dir where to save the samplers. The default is '/Giovanni/MCMC_analysis/samplers/'.
        workers : int, optional
            number of warkers for the parallelization process. The default is 8.
        err : float, optional
            error to assign to the simulated magnitude. The default is None.
        err_max : floar, optional
            maximum error accepted before discarding magnitude (and in turn color) for fit. The default is 0.1.
        err_min : float, optional
            minimum error accepted before discarding magnitude (and in turn color) for fit. The default is 0.001.
        r2 : int, optional
            rounding all magnitude and color related lists to this number of decimals. The default is 4.
        gaussian_kde_bw_method: float, optional
             The method used to calculate the estimator bandwidth for the parallax distribution. This can be 'scott', 'silverman', 
             a scalar constant or a callable. If a scalar, this will be used directly as kde.factor. If a callable, it should take a 
             gaussian_kde instance as only parameter and return a scalar. If None, 'scott' is used. The default is 0.01.
        blobs : bool, optional
            enable/disable blobs in MCMC. The default is False (NOT WORKING AT THE MOMENT).
        nmag2fit:
            number of good magnitude to fit in the magnitude_color_fit. It will start from the first good one and counting. The default is 1.            
        magnitude_fit : bool, optional
            enable/disable magnitude fit. The default is False.
        color_fit : bool, optional
            enable/disable color fit. The default is False.
        magnitude_color_fit:
            enable/disable magnitude AND color fit. The default is False.            
        show_test : bool, optional
            enable/disable printing magnitude and colors for fit. The default is True.
        progress : bool, optional
            enable/disable printing step progression. The default is True.
        parallelize_runs : bool, optional
            enable/disable the parallelization of different targets. The default is False.
        parallelize_sampler : bool, optional
            enable/disable the parallelization of different targets walkers. The default is False (NOT WORKING AT THE MOMENT).
        simulation : bool, optional
            enable/disable simulations. If true, it will simulate magnitudes and color interpolating the input thruths with interp_mags and interp_colors
            The default is False.
        # display_priors : bool, optional
        #     show all priors in the likelihood. The default is False.

        Returns
        -------
        None.

        '''
        self.interp=interp
        
        self.mag_label_list=mag_label_list
        self.emag_label_list=['e%s'%i[1:] for i in mag_label_list]
        self.sat_dict=sat_dict
        self.AV_dict=AV_dict
        self.parallax_KDE=parallax_KDE
        self.Av_KDE=Av_KDE
        self.Age_KDE=Age_KDE
        self.mass_KDE=mass_KDE

        if savedir==None: self.path2savedir=path2data+'/Giovanni/MCMC_analysis/samplers'
        else:self.path2savedir=savedir

        self.parallax=parallax
        self.eparallax=eparallax
        # self.d=dist*u.pc.to(u.cm)*u.cm
        # self.ed=edist*u.pc.to(u.cm)*u.cm
        # self.DM=5*np.log10(dist/10)
        
        self.discard=discard
        self.thin=thin

        self.logMass_min,self.logMass_max=logMass_range
        self.logAv_min,self.logAv_max=logAv_range
        self.logAge_min,self.logAge_max=logAge_range
        self.logSPacc_min,self.logSPacc_max=logSPacc_range
        self.logParallax_min,self.logParallax_max=Parallax_range
        self.nwalkers,self.ndim,self.niters=nwalkers_ndim_niters
        self.Mass2simulate,self.Av2simulate,self.Age2simulate,self.logSPacc2simulate,self.parallax2simulate=truths        

        if workers==None: 
            ncpu=mp.cpu_count()
            if ncpu>=3: workers=ncpu-2
            else: workers=1    
        print('> Selected Workers:', workers)
        self.workers=workers
        self.Teff_label=Teff_label
        self.eTeff_label=eTeff_label
        self.logLacc_label=logLacc_label
        self.elogLacc_label=elogLacc_label
        self.parallax_label=parallax_label
        self.eparallax_label=eparallax_label
        self.WCaII_label=WCaII_label
        
        self.ID_label=ID_label
        self.backend_sampler=backend_sampler
        self.path2backend=path2data+sampler_dir_path
        self.show_test=show_test
        self.progress=progress
        self.parallelize_runs=parallelize_runs
        self.parallelize_sampler=parallelize_sampler
        self.simulation=simulation
        # self.display_priors=display_priors
        
        self.err=err
        self.err_max=err_max
        self.err_min=err_min
        self.r2=r2
        self.gaussian_kde_bw_method=gaussian_kde_bw_method
        self.blobs=blobs
        
        if len(mags2fit)==0:self.mags2fit=self.mag_label_list
        else:self.mags2fit=np.array(mags2fit)
        self.colors2fit=np.array(colors2fit)
        self.ecolor_label_list=['e(%s)'%i for i in colors2fit]
        self.check_acor=check_acor
        self.Fconv=Fconv
        self.conv_thr=conv_thr
        self.ndesired=ndesired

def run(MCMC,avg_df,ID_list):
    '''
    This is the actual run of the MCMC 

    Parameters
    ----------
    avg_df : pandas dataframe.
        this is the dataframe with the average magnitudes of each target.
    ID_list : list
        list of ids in the avg_df to pull for this run.

    Returns
    -------
    MCMC_sim_df: pandas dataframe.
        this dataframe stor all the output of the MCMC run.

    '''
    # global parallax_kde,Av_kde,Age_kde,mass_kde
    # MCMC_sim_df=pd.DataFrame(columns=['ID','variables','truths','samples','blobs','blob_avg','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','Teff','eTeff','T','eT_u','eT_d','logL','elogL_u','elogL_d','logSPacc','elogSPacc_u','elogSPacc_d','logLacc','elogLacc_u','elogLacc_d','logMacc','elogMacc_u','elogMacc_d','Parallax','eParallax_d','eParallax_u','d','ed_u','ed_d'])
    # MCMC_sim_df['ID']=ID_list
    
    parallax_kde=MCMC.parallax_KDE
    Av_kde=MCMC.Av_KDE
    Age_kde=MCMC.Age_KDE
    mass_kde=MCMC.mass_KDE
    
    if MCMC.parallelize_runs: 

        ntarget=len(ID_list)
        num_of_chunks = 3*MCMC.workers
        chunksize = ntarget // num_of_chunks
        if chunksize <=0:
            chunksize = 1
        print('> workers %i,chunksize %i,ntarget %i'%(MCMC.workers,chunksize,ntarget))
        with concurrent.futures.ProcessPoolExecutor(max_workers=MCMC.workers) as executor:#,mp_context=mp.get_context('fork')) as executor:
            for tau,autocorr,converged,burnin,thin,sampler,variable_label_list,variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,ID,mu_T,sig_T,mu_Parallax,sig_Parallax in tqdm(executor.map(aggregated_tasks,repeat(MCMC),ID_list,repeat(avg_df),repeat(parallax_kde),repeat(Av_kde),repeat(Age_kde),repeat(mass_kde),chunksize=chunksize)):
                # MCMC_sim_df=update_dataframe(MCMC,ID,MCMC_sim_df,sampler,tau,variable_label_list,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list,mu_T,sig_T,mu_Parallax,sig_Parallax)
                save_target(MCMC,ID,sampler,tau,mag_good_list,color_good_list,variable_label_list,variable_list,autocorr,converged,burnin,thin)
    else:
        for ID in ID_list:
            tau,autocorr,converged,burnin,thin,sampler,variable_label_list,variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,ID,mu_T,sig_T,mu_Parallax,sig_Parallax=aggregated_tasks(MCMC,ID,avg_df,parallax_kde,Av_kde,Age_kde,mass_kde)
            # MCMC_sim_df=update_dataframe(MCMC,ID,MCMC_sim_df,sampler,tau,variable_label_list,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list,mu_T,sig_T,mu_Parallax,sig_Parallax)
            save_target(MCMC,ID,sampler,tau,mag_good_list,color_good_list,variable_label_list,variable_list,autocorr,converged,burnin,thin)
    # return(MCMC_sim_df)


#############
# MCMC task #
#############
def aggregated_tasks(MCMC,ID,avg_df,parallax_kde_in,Av_kde_in,Age_kde_in,mass_kde_in):
    # try:
    global mu_T,sig_T,mu_logLacc,sig_logLacc,mu_Parallax,sig_Parallax,parallax_kde,Av_kde,Age_kde,mass_kde
    parallax_kde=parallax_kde_in
    Av_kde=Av_kde_in
    Age_kde=Age_kde_in
    mass_kde=mass_kde_in
    
    
    if not np.isnan(avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.WCaII_label].values[0]):
        if avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.WCaII_label].values[0]>0:
            if MCMC.show_test: print('> WCaII > 0. Dropping ...')
            if 'm656' in MCMC.mag_label_list: 
                if MCMC.show_test: print('> ... m656')
                
                index1 = np.argwhere(MCMC.mag_label_list=='m656')
                MCMC.mag_label_list=np.delete(MCMC.mag_label_list,index1)
                MCMC.emag_label_list=np.delete(MCMC.emag_label_list,index1)

            if 'm658' in MCMC.mag_label_list: 
                if MCMC.show_test: print('> ... m658')
                index1 = np.argwhere(MCMC.mag_label_list=='m658')                
                MCMC.mag_label_list=np.delete(MCMC.mag_label_list,index1)
                MCMC.emag_label_list=np.delete(MCMC.emag_label_list,index1)
    
    mag_list,emag_list,mu_T,sig_T,mu_logLacc,sig_logLacc,mu_Parallax,sig_Parallax=pre_task(MCMC,avg_df,ID)
    tau,autocorr,converged,burnin,thin,sampler,variable_label_list,variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list=task(MCMC,ID,mag_list,emag_list,avg_df)
    # except: raise ValueError('Problem with ID: %i. Aborting the run.'%ID)
    return(tau,autocorr,converged,burnin,thin,sampler,variable_label_list,variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,ID,mu_T,sig_T,mu_Parallax,sig_Parallax)       

def pre_task(MCMC,avg_df,ID):
    if MCMC.simulation: 
        mu_T=MCMC.interp['teff'](np.log10(MCMC.Mass2simulate),np.log10(MCMC.Age2simulate),MCMC.logSPacc2simulate)
        sig_T=150
            
        mu_Parallax=MCMC.parallax2simulate
        sig_Parallax=MCMC.eparallax

    else:
        if MCMC.Teff_label in avg_df.columns: mu_T=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.Teff_label].values[0]
        else:mu_T=np.nan
        if MCMC.eTeff_label in avg_df.columns: sig_T=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.eTeff_label].values[0]
        else:sig_T=np.nan

        if MCMC.logLacc_label in avg_df.columns: mu_logLacc=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.logLacc_label].values[0]
        else:mu_logLacc=np.nan
        if MCMC.elogLacc_label in avg_df.columns: sig_logLacc=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.elogLacc_label].values[0]
        else:sig_logLacc=np.nan
        
        if MCMC.parallax_label in avg_df.columns:  mu_Parallax=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.parallax_label].values[0]
        else:mu_Parallax=np.nan
        if MCMC.eparallax_label in avg_df.columns: sig_Parallax=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.eparallax_label].values[0]
        else:sig_Parallax=np.nan

    if MCMC.simulation: 
        mag_list=[]
        emag_list=[]
    else: 
        mag_list=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.mag_label_list].values[0]
        emag_list=avg_df.loc[avg_df[MCMC.ID_label]==ID,MCMC.emag_label_list].values[0]
    return(mag_list,emag_list,mu_T,sig_T,mu_logLacc,sig_logLacc,mu_Parallax,sig_Parallax)

def init_pool(var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21,var22,var23,var24,var25,var26,var27,var28):
    global data_mag,data_color,logMass_min,logMass_max,logAv_min,logAv_max,logAge_min,logAge_max,xParallax_min,xParallax_max,logSPacc_min,logSPacc_max,blobs,mu_Parallax,sig_Parallax,mu_T,sig_T,mu_logLacc,sig_logLacc,interp,mag_good_label_list,color_good_label_list,my_normal_mags,parallax_kde,Av_kde,Age_kde,mass_kde,AV_dict
    data_mag = var1
    data_color = var2
    logMass_min=var3
    logMass_max=var4
    logAv_min=var5
    logAv_max=var6
    logAge_min=var7
    logAge_max=var8
    xParallax_min=var9
    xParallax_max=var10
    logSPacc_min=var11
    logSPacc_max=var12
    blobs=var13
    mu_Parallax=var14
    sig_Parallax=var15
    mu_T=var16
    sig_T=var17
    mu_logLacc=var18
    sig_logLacc=var19
    interp=var20
    mag_good_label_list=var21
    color_good_label_list=var22
    my_normal_mags=var23
    parallax_kde=var24
    Av_kde=var25
    Age_kde=var26
    mass_kde=var27
    AV_dict=var28
    
def task(MCMC,ID,mag_list,emag_list,avg_df):
    global my_normal_mags,my_normal_colors,AV_dict,interp,data_mag,data_color,mag_good_label_list,color_good_label_list,logMass_min,logMass_max,logAv_min,logAv_max,logAge_min,logAge_max,xParallax_min,xParallax_max,logSPacc_min,logSPacc_max,blobs
    logAv_min,logAv_max=[MCMC.logAv_min,MCMC.logAv_max]
    logAge_min,logAge_max=[MCMC.logAge_min,MCMC.logAge_max]
    logSPacc_min,logSPacc_max=[MCMC.logSPacc_min,MCMC.logSPacc_max]
    logMass_min,logMass_max=[MCMC.logMass_min,MCMC.logMass_max]
    
    if np.isnan(mu_Parallax):
        xParallax_min,xParallax_max=[MCMC.logParallax_min,MCMC.logParallax_max]
    else:
        if mu_Parallax-sig_Parallax*3 >0.002: xParallax_min,xParallax_max=[mu_Parallax-sig_Parallax*3,mu_Parallax+sig_Parallax*3]
        else:xParallax_min,xParallax_max=[0.002,mu_Parallax+mu_Parallax+sig_Parallax*3]
    # show_priors.skewnormal_prior(mu_Parallax,sig_Parallax,xlabel='Dist')
    # print(xParallax_min,xParallax_max)
    # sys.exit()

    blobs=MCMC.blobs
    AV_dict=MCMC.AV_dict
    mag_list,emag_list,mag_temp_list,emag_temp_list,mag_good_list=mcmc_utils.simulate_mag_star(ID,[MCMC.sat_dict[i] for i in MCMC.mag_label_list],MCMC.interp,MCMC.mag_label_list,MCMC.AV_dict,ID_label=MCMC.ID_label,mag_list=mag_list,emag_list=emag_list,var=MCMC.Mass2simulate,Av1=MCMC.Av2simulate,age=MCMC.Age2simulate,logSPacc=MCMC.logSPacc2simulate,parallax=MCMC.parallax,err=MCMC.err,err_min=MCMC.err_min,err_max=MCMC.err_max,avg_df=avg_df)
    mag_good_list[np.where(np.in1d(MCMC.mag_label_list, np.setdiff1d(MCMC.mag_label_list,MCMC.mags2fit)))]=False
    mag_list=np.round(mag_list,MCMC.r2)
    mag_temp_list=np.round(mag_temp_list,MCMC.r2)
    emag_temp_list=np.round(emag_temp_list,MCMC.r2)
    emag_list=np.round(emag_list,MCMC.r2)
    mag_good_label_list=MCMC.mag_label_list[mag_good_list]
    data_mag=[mag_list[mag_good_list],emag_list[mag_good_list]]

    if len(MCMC.colors2fit)>0: 
        color_list,ecolor_list,Av1_color_list,color_good_list=mcmc_utils.simulate_color_star(mag_list,emag_list,AV_dict,MCMC.mag_label_list,MCMC.colors2fit)
        color_list=np.round(color_list,MCMC.r2)    
        ecolor_list=np.round(ecolor_list,MCMC.r2)
        Av1_color_list=np.round(Av1_color_list,MCMC.r2)
        color_good_label_list=MCMC.colors2fit[color_good_list]
        data_color=[color_list[color_good_list],ecolor_list[color_good_list]]
    else:
        color_good_label_list=[]
        data_color=[[],[]]
        color_good_list=[]
        color_list=[]
        ecolor_list=[]
    variable_label_list=np.append(mag_good_label_list,color_good_label_list)
    variable_list=np.append(data_mag[0],data_color[0])
    interp=MCMC.interp

    if MCMC.show_test:
        print('> Input data:')
        table=QTable([MCMC.mag_label_list,mag_temp_list, [MCMC.sat_dict[i] for i in MCMC.mag_label_list],emag_temp_list,['%s/%s'%(MCMC.err_min,MCMC.err_max)]*len(MCMC.mag_label_list),mag_list,emag_list,mag_good_list,[MCMC.AV_dict[i] for i in MCMC.mag_label_list]],
                names=('mag_label','mags_temp','sat_th','emags_temp','emag_th', 'mags', 'emags','selected','Av1' ))
        table.pprint()
        if len(MCMC.colors2fit)>0: 
            table=QTable([MCMC.colors2fit,color_list,ecolor_list, color_good_list,Av1_color_list],
                    names=('col_label',  'color', 'ecolor', 'selected','Av1'))
            table.pprint()
    
    if MCMC.blobs==True: MCMC.dtype = [("%s"%i,np.float) for i in variable_label_list]
    else: MCMC.dtype=None
    if MCMC.show_test:
        print('> Magnitude b fit')
        print(variable_label_list)

    moves=[(emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3),]
    if any(mag_good_list) or any(color_good_list):
        my_normal_mags=my_multivariate_normal(data_mag[0],data_mag[1])
        if any(color_good_list): my_normal_colors=my_multivariate_normal(data_color[0],data_color[1])      
        else: my_normal_colors=None
        try:
            if not np.isnan(mu_Parallax): pos = np.array([np.array([np.random.uniform(logMass_min,logMass_max),np.random.uniform(MCMC.logAv_min,MCMC.logAv_max),np.random.uniform(MCMC.logAge_min,MCMC.logAge_max),np.random.uniform(MCMC.logSPacc_min,MCMC.logSPacc_max),np.random.uniform(xParallax_min,xParallax_max)]) for i in range(MCMC.nwalkers)])        
            else: pos = np.array([np.array([np.random.uniform(logMass_min,logMass_max),np.random.uniform(MCMC.logAv_min,MCMC.logAv_max),np.random.uniform(MCMC.logAge_min,MCMC.logAge_max),np.random.uniform(MCMC.logSPacc_min,MCMC.logSPacc_max),np.random.uniform(xParallax_min,xParallax_max)]) for i in range(MCMC.nwalkers)])        
        except:
            print('problems with ID: ',ID)
            return(np.nan,[],variable_label_list,variable_list,mag_list,emag_list,[False],[False],[False],[False])
        if MCMC.backend_sampler:
            path2savename = MCMC.path2backend+'sampler_ID%s.h5'%(ID)
            if os.path.exists(path2savename):
                os.remove(path2savename)
            backend = emcee.backends.HDFBackend(path2savename)
            backend.reset(MCMC.nwalkers, MCMC.ndim)
        else: backend=None
            
        
        if MCMC.parallelize_sampler:
            with Pool(initializer=init_pool, initargs=(data_mag,data_color,logMass_min,logMass_max,logAv_min,logAv_max,logAge_min,logAge_max,xParallax_min,xParallax_max,logSPacc_min,logSPacc_max,blobs,mu_Parallax,sig_Parallax,mu_T,sig_T,mu_logLacc,sig_logLacc,interp,mag_good_label_list,color_good_label_list,my_normal_mags,parallax_kde,Av_kde,Age_kde,mass_kde,AV_dict)) as pool:
              if MCMC.blobs:sampler = emcee.EnsembleSampler(MCMC.nwalkers, MCMC.ndim, log_probability, blobs_dtype=MCMC.dtype, pool=pool, moves=moves, backend=backend)
              else: sampler = emcee.EnsembleSampler(MCMC.nwalkers, MCMC.ndim, log_probability,pool=pool,moves=moves, backend=backend)
              sampler,autocorr,converged,burnin,thin=sampler_convergence(MCMC,sampler,pos)

        else:
            if MCMC.blobs:sampler = emcee.EnsembleSampler(MCMC.nwalkers, MCMC.ndim, log_probability, blobs_dtype=MCMC.dtype,moves=moves, backend=backend)
            else: sampler = emcee.EnsembleSampler(MCMC.nwalkers, MCMC.ndim, log_probability,moves=moves, backend=backend)
            sampler,autocorr,converged,burnin,thin=sampler_convergence(MCMC,sampler,pos)
        tau = sampler.get_autocorr_time(tol=0)
        return(tau,autocorr,converged,burnin,thin,sampler,variable_label_list,variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list)

    else:
        return(np.nan,np.nan,np.nan,np.nan,np.nan,[],variable_label_list,variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list)

def sampler_convergence(MCMC,sampler,pos):
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(MCMC.niters*3)
    
    # This will be useful to testing convergence
    old_tau = np.inf
    converged = False
    for sample in sampler.sample(pos, iterations=MCMC.niters*3, progress=MCMC.progress):
    # Only check convergence every check_acor steps
        if converged == False:
            # print('VVVVVVVVVVVVVVVVVVVVVV')
            if sampler.iteration % MCMC.check_acor:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            if np.any(np.isnan(tau)):
                tau[:] = sampler.iteration/100
            
            autocorr[index] = np.mean(tau)
            thin = np.int_(2 * np.max(tau))
            burnin = np.int_(sampler.iteration/2)
            index += 1

            # print(index,int(MCMC.niters),index>int(MCMC.niters),converged)
        # Check convergence
            converged = np.all(tau * MCMC.Fconv < sampler.iteration)
            converged &= np.all( (np.abs(old_tau - tau) / tau) < MCMC.conv_thr)
            if converged:

                #Once converged, set the number of desired runs for further running the sampler
                #until we have the desired number of post-convergence, iid samples
                burnin = sampler.iteration
                n_post_convergence_runs = int(MCMC.ndesired//MCMC.nwalkers*thin)
                n_to_go = 0
                if MCMC.parallelize_sampler:
                    print('Converged at iteration {}'.format(burnin))
                    print('Autocorrelation times equal to: {}'.format(tau))
                    print('Thinning equal to: {}'.format(thin))
                    print('Running {} iterations post-convergence'.format(n_post_convergence_runs))
                sys.stdout.flush()
            
            elif index>=int(MCMC.niters/MCMC.check_acor): 
                break
            old_tau = tau
        
        else:
            #Post-convergence samples
            n_to_go +=1
            if n_to_go > n_post_convergence_runs:
                break
    return(sampler,autocorr,converged,burnin,thin)

                    
def save_target(MCMC,ID,sampler,tau,mag_good_list,color_good_list,variable_label_list,variable_list,autocorr,converged,burnin,thin):
    # n=10
    if MCMC.parallelize_sampler: print('> tau: ',tau)
    if (any(mag_good_list) or any(color_good_list)) and not (all(np.isnan(tau))):
        if converged and ( (sampler.iteration+1) >= MCMC.niters):
            converged = False
            burnin = np.int_(sampler.iteration/2)
            thin = np.int_(sampler.iteration/20)
            
        if converged and (burnin == MCMC.niters):
            converged = False
            burnin = np.int_(sampler.iteration/2)
            thin = np.int_(sampler.iteration/20)
        if MCMC.discard==None:MCMC.discard=burnin
        if MCMC.thin==None:MCMC.thin=thin
        
        
        blobs = sampler.get_blobs(discard=MCMC.discard, thin=MCMC.thin)
        samples = sampler.get_chain(discard=MCMC.discard, thin=MCMC.thin)
        dict_to_save = {'id':ID,
                         'samples':samples.tolist(),
                         # 'log_prob':sampler.get_log_prob(),
                         'blobs':blobs.tolist(),
                         'autocorr':autocorr,
                         'converged':converged,
                         'thin':MCMC.thin,
                         'burnin':MCMC.discard,
                         'tau':tau,
                         'variables_label':variable_label_list.tolist(),
                         'variables':variable_list.tolist(),
                         # 'sdbid':row['star_ID'],
                         # 'name':row['star_ID'],
                         # 'par_sigmas':par_sigmas,
                         # 'par_iqrs':par_iqrs,
                         # 'par_means':par_means,
                         # 'par_std':par_std,
                         # 'q2fit':q2fit
                         }
                 
        with bz2.BZ2File(MCMC.path2savedir+'/samplerID_%s'%ID, 'w') as f:
            pickle.dump(dict_to_save, f)
        
##################################
# Probability functions          #
# var can be either mass or Teff #
##################################

def log_mag_likelihood(params):#,y,yerr):
    ym=model_mag(params)
    ll=np.log(my_normal_mags.pdf(ym))
    return(ll,ym)

def log_col_likelihood(params):#,y,yerr):
    ym=model_col(params)
    ll=np.log(my_normal_colors.pdf(ym))
    return(ll,ym)

def log_prior(params):
    logMass_x,logAv_x,logAge_x,logSPacc_x,xParallax_x=params
    if (logMass_min <= logMass_x <= logMass_max) and (logAv_min <= logAv_x <= logAv_max) and (logAge_min <= logAge_x <= logAge_max) and (xParallax_min <= xParallax_x <= xParallax_max) and (logSPacc_min <= logSPacc_x <= logSPacc_max):
        lp_Av=0
        lp_parallax=0
        lp_mass=0
        lp_A=0
        lp_logLacc=0
        if not np.isnan(mu_Parallax): 
            lp_parallax = np.log(skewnorm.pdf(xParallax_x,a=0,loc=mu_Parallax, scale=sig_Parallax))
        else:
            if parallax_kde!=None: lp_parallax=np.log(parallax_kde.pdf(xParallax_x))
            if Av_kde!=None: lp_Av=np.log(Av_kde.pdf(logAv_x))
            if Age_kde!=None: lp_A=np.log(Age_kde.pdf(logAge_x))
            if mass_kde!=None: lp_mass=np.log(mass_kde.pdf(logMass_x))
        if not np.isnan(mu_T): lp_mass=np.log(skewnorm.pdf(interp['teff'](logMass_x,logAge_x,logSPacc_x),a=0,loc=mu_T,scale=sig_T))
        # if not np.isnan(mu_logLacc): lp_logLacc=np.log(skewnorm.pdf(interp['logLacc'](logMass_x,logAge_x,logSPacc_x),a=0,loc=mu_logLacc,scale=sig_logLacc))
        
        return(lp_mass+lp_parallax+lp_Av+lp_A+lp_logLacc)
    return(-np.inf)

def log_probability(params):
    mag,emag=data_mag
    col,ecol=data_color
    lp = log_prior(params)
    y=np.append(mag,col)
    if not np.isfinite(lp):
        if not blobs:return(-np.inf)
        else:
            if len(y)==1:return(-np.inf,-np.inf)
            elif len(y)==2:return(-np.inf,-np.inf,-np.inf)
            elif len(y)==3:return(-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==4:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==5:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==6:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==7:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==8:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==9:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==10:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==11:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==12:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            elif len(y)==13:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)

    llmag,ymmag=log_mag_likelihood(params)#,mag,emag)
    if  len(col)!=0: llcol,ymcol=log_col_likelihood(params)#,col,ecol)
    else: llcol,ymcol=[0,0]
    ll=llmag+llcol+lp
    ym=np.append(ymmag,ymcol)
    if not blobs:return(ll)
    else:
        if len(y)==1: return(ll,ym[0])
        elif len(y)==2: return(ll,ym[0],ym[1])
        elif len(y)==3: return(ll,ym[0],ym[1],ym[2])
        elif len(y)==4: return(ll,ym[0],ym[1],ym[2],ym[3])
        elif len(y)==5: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4])
        elif len(y)==6: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5])
        elif len(y)==7: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6])
        elif len(y)==8: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7])
        elif len(y)==9: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7],ym[8])
        elif len(y)==10: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7],ym[8],ym[9])
        elif len(y)==11: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7],ym[8],ym[9],ym[10])
        elif len(y)==12: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7],ym[8],ym[9],ym[10],ym[11])
        elif len(y)==13: return(ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7],ym[8],ym[9],ym[10],ym[11],ym[12])
    return(lp+llmag+llcol)

######################
# Ancillary routines #
######################
def my_multivariate_normal(measurements,errs):
    cov_matrix   = np.diag(errs)
    mynormal = multivariate_normal(mean = measurements, cov = cov_matrix, allow_singular=True)
    return(mynormal)

def model_mag(params):
    logMass_x,logAv_x,logAge_x,logSPacc_x,xParallax_x=params
    variables_list=[]
    distance=(xParallax_x* u.pc).to(u.mas, equivalencies=u.parallax()).value
    DM_x=5*np.log10(distance/10)

    # DM_x=5*np.log10(10**(xParallax_x* u.pc).to(u.mas, equivalencies=u.parallax()).value)

    for label in mag_good_label_list:
        x=interp[label](logMass_x,logAge_x,logSPacc_x)+10**logAv_x*AV_dict[label]+DM_x
        variables_list.append(x)
    return(np.array(variables_list))

def model_col(params):
    logMass_x,logAv_x,logAge_x,logSPacc_x,_=params
    variables_list=[]

    for label in color_good_label_list:
        label1=label.split('-')[0]
        label2=label.split('-')[1]
        x1=interp[label1](logMass_x,logAge_x,logSPacc_x)+10**logAv_x*AV_dict[label1]
        x2=interp[label2](logMass_x,logAge_x,logSPacc_x)+10**logAv_x*AV_dict[label2]
        variables_list.append(x1-x2)
    return(np.array(variables_list))
   
