#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 08:48:09 2021

@author: giovanni
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:33:35 2021

@author: giovanni
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:34:01 2021

@author: giovanni
"""

import sys,emcee,os,random
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append('./')
from ancillary import simulate_mag_star,simulate_color_star,get_Av_list
from priors import DaRio_dist,lognormal_dist,normal_dist,kroupa_dist,chabrier_dist
import numpy as np
from astropy.table import QTable
from multiprocessing import Pool
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from itertools import repeat
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt
from IPython.display import display

class MCMC():
    
    def __init__(self,interp_mags,interp_colors,interp_star_properties,filter_label_list,mag_label_list,emag_label_list,color_label_list,ecolor_label_list,sat_list,photflam,RW=74.96,dist=402,edist=5,DM_mass_age_4Av_test=[8.02,1,1],truths=[None,None,None],mlabel_Rv=['3','a'],var_range=[0.01,1.6],Av_range=[0,10],t_range=[0.5,100],nwalkers_ndim_niters=[50,3,10000],mu_sigma_m=[0.3,0.1],mu_sigma_t=[2,1],sigma_T=None,Teff_label='Teff',DaRio_path='/media/giovanni/DATA_backup/Lavoro/Giovanni/NGC1976/ACS/DaRio_ACS_matched.csv',Av_list=[],workers=8,distance=None,err=None,err_max=0.05,err_min=0.001,mag_list=[],emag_list=[],blobs=False,show_test=True,progress=True,parallelize_runs=False,parallelize_sampler=False,simulation=False,physical_prior=True,Rv_physical_prior=True,t_physical_prior=True,var_physical_prior=True):
        self.interp_mags=interp_mags
        self.interp_colors=interp_colors
        self.interp_star_properties=interp_star_properties
        
        self.filter_label_list=filter_label_list
        self.mag_label_list=mag_label_list
        self.emag_label_list=emag_label_list
        self.color_label_list=color_label_list
        self.ecolor_label_list=ecolor_label_list
        self.sat_list=sat_list
        self.photflam=photflam
        
        self.G=c.G.to(u.cm**3/(u.g*u.s**2))
        self.eG=c.G.uncertainty*(u.m**3/(u.kg*u.s**2)).to(u.cm**3/(u.g*u.s**2))*u.cm**3/(u.g*u.s**2)
        
        self.sigma=c.sigma_sb.to(u.erg/(u.s*u.cm**2*u.K**4))
        
        self.Lsun=c.L_sun.to(u.erg/u.s)*(u.erg/u.s) #https://link.springer.com/referenceworkentry/10.1007%2F1-4020-4520-4_374#:~:text=This%20translates%20into%20a%20solar,1988%3B%20see%20Solar%20constant).
        self.eLsun=0.003*self.Lsun
        
        self.Msun=c.M_sun.to(u.g)
        self.eMsun=c.M_sun.uncertainty*u.kg.to(u.g)*u.g
        
        self.Rsun=c.R_sun.to(u.cm)*u.cm
        self.eRsun=0.026*u.Mm.to(u.cm)*u.cm #https://iopscience.iop.org/article/10.1086/311416/fulltext/985175.text.html
        
        self.d=dist*u.pc.to(u.cm)*u.cm
        self.ed=edist*u.pc.to(u.cm)*u.cm
        
        self.RW=RW
        # self.zpt658=22.378
        # self.photlam658=1.977e-18
        
        self.mlabel,self.Rv=mlabel_Rv
        self.var_min,self.var_max=var_range
        self.Av_min,self.Av_max=Av_range
        self.t_min,self.t_max=t_range
        self.nwalkers,self.ndim,niters=nwalkers_ndim_niters
        self.mass2simulate,self.Av2simulate,self.age2simulate=truths        
        
        self.sig_T=sigma_T
        self.mu_m,self.sig_m=[np.log(mu_sigma_m[0]**2/np.sqrt(mu_sigma_m[0]**2+mu_sigma_m[1]**2)),np.sqrt(np.log(1+mu_sigma_m[1]**2/mu_sigma_m[0]**2))]
        self.mu_t,self.sig_t=[np.log(mu_sigma_t[0]**2/np.sqrt(mu_sigma_t[0]**2+mu_sigma_t[1]**2)),np.sqrt(np.log(1+mu_sigma_t[1]**2/mu_sigma_t[0]**2))]
        
        self.use_physical_prior=physical_prior
        self.use_Rv_physical_prior=Rv_physical_prior
        self.use_t_physical_prior=t_physical_prior
        self.use_var_physical_prior=var_physical_prior
        
    def run(self,mvs_df,avg_df,ID_list):
      
        if self.Rv=='a': Av_list,_=get_Av_list(self.interp_mags,interp_Tlogg,filter_label_list,mag_label_list,photflam,3,DM=DM_mass_age_4Av_test[0],mass=DM_mass_age_4Av_test[1],age=DM_mass_age_4Av_test[2],showplot=False,photlam658=1.977e-18)
        elif self.Rv=='b': Av_list,_=get_Av_list(interp_mags,interp_Tlogg,filter_label_list,mag_label_list,photflam,5,DM=DM_mass_age_4Av_test[0],mass=DM_mass_age_4Av_test[1],age=DM_mass_age_4Av_test[2],showplot=False,photlam658=1.977e-18)
    
        MCMC_sim_df=pd.DataFrame(columns=['ID','truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','L','eL_u','eL_d','T','eT_u','eT_d','L_corr','N'])
    
        MCMC_sim_df['ID']=ID_list
    
        if parallelize_runs: 
            ntarget=len(ID_list)
            num_of_chunks = 3*workers
            chunksize = ntarget // num_of_chunks
            if chunksize <=0:
                chunksize = 1
            print('> workers %i,chunksize %i,ntarget %i'%(workers,chunksize,ntarget))
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype,ID in tqdm(executor.map(task_parallelized,repeat(df),ID_list,repeat(simulation),repeat(Teff_label),repeat(interp_mags),repeat(interp_colors),repeat(filter_label_list),repeat(mag_label_list),repeat(emag_label_list),repeat(color_label_list),repeat(sat_list),repeat(mlabel_Rv),repeat(var_range),repeat(Av_range),repeat(t_range),repeat(nwalkers_ndim_niters),repeat(mu_sigma_m),repeat(mu_sigma_t),repeat(sigma_T),repeat(DaRio_path),repeat(Av_list),repeat(workers),repeat(mass2simulate),repeat(Av2simulate),repeat(age2simulate),repeat(distance),repeat(err),repeat(err_max),repeat(err_min),repeat(mag_list),repeat(emag_list),repeat(blobs),repeat(show_test),repeat(progress),repeat(parallelize_sampler),repeat(physical_prior),repeat(Rv_physical_prior),repeat(t_physical_prior),repeat(var_physical_prior),chunksize=chunksize)):
                    if len(data):
                        discard=int(niters/2)
                        thin=int(0.5*np.min(tau))
    
                        samples = sampler.get_chain(discard=discard, thin=thin)
                        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
                        mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=star_properties(flat_samples,ndim,interp_star_properties,mlabel)
                        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','T','eT_u','eT_d','L','eL_u','eL_d']]=[[truths,samples.tolist(),flat_samples.tolist(),mag_good_list.tolist(),color_good_list,tau.tolist(),mag_list,emag_list,color_list,ecolor_list,mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d]]
    
        else:
            for ID in ID_list:
                mu_sigma_T,mag_list,emag_list=pre_task(df,ID,simulation,mag_label_list,emag_label_list,Teff_label,sigma_T)
                tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype=task(interp_mags,interp_colors,filter_label_list,mag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,mu_sigma_T,DaRio_path,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,mag_list,emag_list,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior)
                if len(data):
                    discard=int(niters/2)
                    thin=int(0.5*np.min(tau))
    
                    samples = sampler.get_chain(discard=discard, thin=thin)
                    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
                    mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=star_properties(flat_samples,ndim,interp_star_properties,mlabel)
                    MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','T','eT_u','eT_d','L','eL_u','eL_d']]=[[truths,samples.tolist(),flat_samples.tolist(),mag_good_list.tolist(),color_good_list,tau.tolist(),mag_list,emag_list,color_list,ecolor_list,mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d]]
    
        # MCMC_sim_df.loc[~MCMC_sim_df.good_cols.isna(),'N']=np.array(MCMC_sim_df.loc[~MCMC_sim_df.good_cols.isna(),'good_cols'].tolist()).astype(int).sum(axis=1)
        # MCMC_sim_df=star_accrention_properties(MCMC_sim_df,mean_df,interp_mags,mlabel,Rv,interp_658,DM_mass_age_4Av_test[0])
        # df.loc[df.UniqueID.isin(MCMC_sim_df.ID.unique()),['N_%s%s'%(mlabel,Rv),'mass_%s%s'%(mlabel,Rv), 'emass_%s%s'%(mlabel,Rv), 'emass_d_%s%s'%(mlabel,Rv), 'emass_u_%s%s'%(mlabel,Rv), 
        #                                                 'Av_%s%s'%(mlabel,Rv), 'eAv_%s%s'%(mlabel,Rv), 'eAv_d_%s%s'%(mlabel,Rv), 'eAv_u_%s%s'%(mlabel,Rv),
        #                                                 'A_%s%s'%(mlabel,Rv), 'eA_%s%s'%(mlabel,Rv), 'eA_u_%s%s'%(mlabel,Rv), 'eA_d_%s%s'%(mlabel,Rv), 
        #                                                 'T_%s%s'%(mlabel,Rv), 'eT_%s%s'%(mlabel,Rv), 'eT_u_%s%s'%(mlabel,Rv), 'eT_d_%s%s'%(mlabel,Rv), 
        #                                                 'L_%s%s'%(mlabel,Rv), 'eL_%s%s'%(mlabel,Rv), 'eL_d_%s%s'%(mlabel,Rv), 'eL_u_%s%s'%(mlabel,Rv), 
        #                                                 'L_corr_%s%s'%(mlabel,Rv), 'eL_corr_%s%s'%(mlabel,Rv),
        #                                                 'DHa_%s%s'%(mlabel,Rv), 'eDHa_%s%s'%(mlabel,Rv),
        #                                                 'EQW_%s%s'%(mlabel,Rv), 'eEQW_%s%s'%(mlabel,Rv), 
        #                                                 'logLHa_%s%s'%(mlabel,Rv), 'elogLHa_%s%s'%(mlabel,Rv), 'logL_acc_%s%s'%(mlabel,Rv), 'elogL_acc_%s%s'%(mlabel,Rv),
        #                                                 'logdM_acc_%s%s'%(mlabel,Rv), 'elogdM_acc_%s%s'%(mlabel,Rv)]]=MCMC_sim_df[['N', 'mass', 'emass', 'emass_d', 'emass_u', 
        #                                                 'Av', 'eAv', 'eAv_d', 'eAv_u',
        #                                                 'A', 'eA', 'eA_u', 'eA_d', 
        #                                                 'T', 'eT', 'eT_u', 'eT_d', 
        #                                                 'L', 'eL', 'eL_d', 'eL_u', 
        #                                                 'L_corr', 'eL_corr',
        #                                                 'DHa', 'eDHa',
        #                                                 'EQW', 'eEQW', 'logLHa', 'elogLHa', 'logL_acc', 'elogL_acc',
        #                                                 'logdM_acc', 'elogdM_acc']].values
        return(MCMC_sim_df,df)
    
#############
# MCMC task #
#############

def pre_task(df,ID,simulation,mag_label_list,emag_label_list,Teff_label,sigma_T):
    if mlabel=='0':mu_T=df.loc[df.UniqueID==ID,Teff_label].values[0]
    else: mu_T=None
    mu_sigma_T=[mu_T,sigma_T]
    if simulation: 
        mag_list=[]
        emag_list=[]
    else: 
        mag_list=df.loc[df.UniqueID==ID,mag_label_list].values[0]
        emag_list=df.loc[df.UniqueID==ID,emag_label_list].values[0]
    return(mu_sigma_T,mag_list,emag_list)

def task(interp_mags,interp_color,filter_label_list,mag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,mu_sigma_T,DaRio_finename,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,mag_list,emag_list,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior):

    DaRio_pdf=DaRio_dist(Av_min,Av_max,DaRio_finename=DaRio_finename)

    if mlabel=='2':
        cc=0.158
        cmu=0.079
        csig=0.69
    elif mlabel=='3':
        cc=0.086
        cmu=0.22
        csig=0.57
    else:
        cc=None
        cmu=None
        csig=None
    
    mag_list,emag_list,mag_temp_list,emag_temp_list,mag_good_list=simulate_mag_star(sat_list,interp_mags,mag_label_list,Av_list,mag_list=mag_list,emag_list=emag_list,mass=mass2simulate,Av1=Av2simulate,age=age2simulate,distance=distance,err=err,err_min=err_min,err_max=err_max)
    color_list,ecolor_list,Av1_color_list,color_good_list=simulate_color_star(mag_list,emag_list,Av_list,mag_label_list,color_label_list)

    data_variable_list=[color_list[color_good_list],ecolor_list[color_good_list]]
    Av1_variable_list=Av1_color_list[color_good_list]
    variables_interp=interp_color[color_good_list]
    if blobs==True: dtype = [("%s"%i,np.float) for i in color_label_list[color_good_list]]
    else: dtype=None
    if show_test:
        table=QTable([mag_label_list,mag_good_list, sat_list,mag_temp_list,emag_temp_list,mag_list,emag_list,Av_list],
                names=('mag_label','good' ,'sat_list','mags_temp','emags_temp', 'mags', 'emags','Av1'))
        table.pprint()

        table=QTable([color_label_list, color_good_list,color_list,ecolor_list,Av1_color_list],
                names=('col_label', 'good',  'color', 'ecolor','Av1'))
        table.pprint()

    if any(color_good_list):
        pos = np.array([np.array([np.random.uniform(var_min,var_max),np.random.uniform(Av_min,Av_max),np.random.uniform(t_min,t_max)]) for i in range(nwalkers)])
        if parallelize_sampler:
             with Pool(processes=workers) as pool:
                if blobs: sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,blobs_dtype=dtype, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                else: sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])    
                sampler.run_mcmc(pos, niters, progress=progress)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,blobs_dtype=dtype,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
            sampler.run_mcmc(pos, niters, progress=progress)
        tau = sampler.get_autocorr_time(tol=0)
        return(tau,sampler,data_variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype)

    else:
        return(np.nan,[],[],mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,[])
  
def task_parallelized(df,ID,simulation,Teff_label,interp_mags,interp_colors,filter_label_list,mag_label_list,emag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,sigma_T,DaRio_path,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,mag_list,emag_list,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior):
    # if mlabel=='0':mu_T=df.loc[df.UniqueID==ID,Teff_label].values[0]
    # else: mu_T=None
    # mu_sigma_T=[mu_T,sigma_T]
    # if simulation: 
    #     mag_list=[]
    #     emag_list=[]
    # else: 
    #     mag_list=df.loc[df.UniqueID==ID,mag_label_list].values[0]
    #     emag_list=df.loc[df.UniqueID==ID,emag_label_list].values[0]
    mu_sigma_T,mag_list,emag_list=pre_task(df,ID,simulation,mag_label_list,emag_label_list,Teff_label,sigma_T)
    tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype=task(interp_mags,interp_colors,filter_label_list,mag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,mu_sigma_T,DaRio_path,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,mag_list,emag_list,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior)
    return(tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype,ID)       
     

##################################
# Probability functions          #
# var can be either mass or Teff #
##################################

def truth_list(mass,Av,age,mass_lim=[0.1,0.9],Av_lim=[0,10],age_lim=[0,100]):
    if mass==None:mass=np.round(random.uniform(mass_lim[0],mass_lim[1]),4)
    if Av==None: Av=np.round(random.uniform(Av_lim[0],Av_lim[1]),4)
    if age==None:age=np.round(random.uniform(age_lim[0],age_lim[1]),4)
    return(mass,Av,age)

def models(params):
    # mass_x,Av_x,t_x=params
    var_x,Av_x,t_x=params
    variables_list=[]
    for elno in range(np.array(variables_interp).shape[0]):
        x=variables_interp[elno](np.log10(var_x),np.log10(t_x))
        variables_list.append(x+Av_x*Av1_variable_list[elno])
    variables_list=np.array(variables_list)
    return(variables_list)

def log_likelihood(params,y,yerr):
    ym=models(params)
    return(-0.5*np.sum((np.round(y,3)-np.round(ym,3))**2/yerr**2),np.round(ym,3))

def log_prior(params):
    var_x,Av_x,t_x=params
    if (var_min <= var_x <= var_max) and (Av_min <= Av_x <= Av_max) and (t_min <= t_x <= t_max):
        if use_physical_prior==True:
            if use_Rv_physical_prior:
                if Rv=='a': 
                    lp_Av=np.log(DaRio_pdf(Av_x)[0])
                else: lp_Av=0
            else: lp_Av=0
            
            if use_t_physical_prior: lp_t=np.log(lognormal_dist(x=t_x,mu=mu_t,sig=sig_t))
            else: lp_t=0
            
            if use_var_physical_prior: 
                # if mlabel=='0': lp_var=np.log(normal_dist(x=np.log10(var_x),mu=np.log10(mu_T),sig=np.log10(sig_T)))
                if mlabel=='0': lp_var=np.log(normal_dist(x=var_x,mu=mu_T,sig=sig_T))
                elif mlabel=='1': lp_var=np.log(kroupa_dist(mass=var_x)) 
                elif mlabel=='2' or mlabel == '3': lp_var=np.log(chabrier_dist(mass=var_x,cc=cc,cmu=cmu,csig=csig))
                elif mlabel=='4': lp_var=np.log(lognormal_dist(var_x,mu_m,sig_m))
                # print('var_x: %.3f, Av_x: %.3f, t_x: %.3f'%(var_x,Av_x,t_x))
                # print('lp_var: %.3f, lp_Av: %.3f, lp_t: %.3f'%(lp_var,lp_Av,lp_t))
            else: lp_var=0
            return(lp_var+lp_Av+lp_t)
                
        else:         
            return(0.0)
    return(-np.inf)

def log_probability(params):
    y,yerr=data_variable_list
    lp = log_prior(params)
    if not np.isfinite(lp):return(-np.inf)
#         if len(y)==1:return(-np.inf,-np.inf)
#         if len(y)==2:return(-np.inf,-np.inf,-np.inf)
#         if len(y)==3:return(-np.inf,-np.inf,-np.inf,-np.inf)
#         if len(y)==4:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
#         if len(y)==5:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
#         if len(y)==6:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
#         return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
    ll,ym=log_likelihood(params,y,yerr)
#     if len(y)==1: return(lp+ll,ym[0])
#     if len(y)==2: return(lp+ll,ym[0],ym[1])
#     if len(y)==3: return(lp+ll,ym[0],ym[1],ym[2])
#     if len(y)==4: return(lp+ll,ym[0],ym[1],ym[2],ym[3])
#     if len(y)==5: return(lp+ll,ym[0],ym[1],ym[2],ym[3],ym[4])
#     if len(y)==6: return(lp+ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5])
#     return(lp+ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5],ym[6],ym[7],ym[8],ym[9],ym[10])
    if np.isnan(ll):print(params)
    # print('lp: %.3f, ll: %.3f\n'%(lp,ll))
    return(lp+ll)


################################
# Star and accretion proprties #
################################

def star_properties(flat_samples,ndim,interp_star_properties,mlabel):
    if mlabel == '0':
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            if i == 0: 
                T= round(mcmc[1],4)
                eT_u= round(q[1],4)
                eT_d= round(q[0],4)
            elif i == 1: 
                Av= round(mcmc[1],4)
                eAv_u= round(q[1],4)
                eAv_d= round(q[0],4)
            elif i == 2: 
                age= round(mcmc[1],4)
                eage_u= round(q[1],4)
                eage_d= round(q[0],4)
                
        mass=round(float(interp_star_properties[0](np.log10(T),np.log10(age))),4)
        emass_u=round(float(interp_star_properties[0](np.log10(T+eT_u),np.log10(age+eage_u)))-mass,4)
        emass_d=round(mass-float(interp_star_properties[0](np.log10(T-eT_d),np.log10(age-eage_d))),4)
        
        if emass_u <=0: emass_u = 0.1*mass
        if emass_d <=0: emass_d=emass_u
    
        L=round(float(10**interp_star_properties[1](np.log10(T),np.log10(age))),4)
        eL_u=round(float(10**interp_star_properties[1](np.log10(T+emass_u),np.log10(age+eage_u)))-L,4)
        eL_d=round(L-float(10**interp_star_properties[1](np.log10(T-emass_d),np.log10(age-eage_d))),4)
        
        if eL_u <=0: eL_u = 0.1*L
        if eL_d <=0: eL_d=eL_u
    else: 
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            if i == 0: 
                mass= round(mcmc[1],4)
                emass_u= round(q[1],4)
                emass_d= round(q[0],4)
            elif i == 1: 
                Av= round(mcmc[1],4)
                eAv_u= round(q[1],4)
                eAv_d= round(q[0],4)
            elif i == 2: 
                age= round(mcmc[1],4)
                eage_u= round(q[1],4)
                eage_d= round(q[0],4)
                
        T=round(float(interp_star_properties[0](np.log10(mass),np.log10(age))),4)
        eT_u=round(float(interp_star_properties[0](np.log10(mass+emass_u),np.log10(age+eage_u)))-T,4)
        eT_d=round(T-float(interp_star_properties[0](np.log10(mass-emass_d),np.log10(age-eage_d))),4)

        if eT_u <=0: eT_u = 0.1*T
        if eT_d <=0: eT_d=eT_u
    
        L=round(float(10**interp_star_properties[1](np.log10(mass),np.log10(age))),4)
        eL_u=round(float(10**interp_star_properties[1](np.log10(mass+emass_u),np.log10(age+eage_u)))-L,4)
        eL_d=round(L-float(10**interp_star_properties[1](np.log10(mass-emass_d),np.log10(age-eage_d))),4)
        
        if eL_u <=0: eL_u = 0.1*L
        if eL_d <=0: eL_d=eL_u
            
    return(mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d)

def lum_corr(MCMC_sim_df,ID,interp_mags,Av_list,DM):
    mag_good_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'good_mags'].values[0]
    mag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mags'].values[0]
    emag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'emags'].values[0]
    mass=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]
    Av=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]
    age=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]
    L=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'L'].values[0]

    dmag_corr_list=[]
    emag_corr_list=[]
    q=np.where(mag_good_list)[0]

    for elno in q:
        iso_mag=float(interp_mags[elno](np.log10(mass),np.log10(age)))
        mag=mag_list[elno]-DM-Av*Av_list[elno]
        dmag=float((mag-iso_mag)/iso_mag)
        dmag_corr_list.append(dmag)
        emag=emag_list[elno]
        emag_corr_list.append(emag**(-2))

    dmag_corr_list=np.array(dmag_corr_list)
    emag_corr_list=np.array(emag_corr_list)
    dmag_mag=np.average(dmag_corr_list,weights=emag_corr_list)
    df_f=dmag_mag/1.087 
    return(L*(1+df_f))

def accr_stats(MCMC_sim_df,ID,m658_c,m658_d,e658,e658_c,label=''):
    M=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]*Msun
    eM=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'emass'].values[0]*Msun

    T=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'T'].values[0]*u.K
    eT=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eT'].values[0]*u.K

    L=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'L_corr'].values[0]*Lsun
    eL=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eL_corr'].values[0]*Lsun

    # A=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]*u.Myr
    # eA=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eA'].values[0]*u.Myr
                    
    R=np.sqrt(L/(4*np.pi*sigma*T**4))
    eR=np.sqrt(R**2*np.array([(eL/L).value**2+(eT/T).value**2]))[0]

    EQW=RW*(1-10**(-0.4*(m658_c-m658_d)))
    eEQW=np.sqrt(RW**2*(-0.4*np.log(10)*np.sqrt(e658**2+e658_c**2))**2)

    if (m658_c-m658_d>= 3*e658) and (EQW>10):

        electrons_d=10**(-(m658_d-zpt658)/2.5)
        e_electrons_d=electrons_d*e658/1.086

        electrons_c=10**(-(m658_c-zpt658)/2.5)
        e_electrons_c=electrons_d*e658_c/1.086

        dflux_density_Ha=(electrons_d-electrons_c)*photlam658*87.487*u.angstrom
        edflux_density_Ha=np.sqrt((e_electrons_d**2+e_electrons_c**2)*(photlam658*87.487*u.angstrom).value**2)

        LHa=((dflux_density_Ha*4*np.pi*d**2)/Lsun).value
        if not np.isnan(np.log10(LHa)): #
            eLHa=np.sqrt(LHa**2*np.array([(edflux_density_Ha/dflux_density_Ha.value)**2+(ed/d)**2+(eLsun/Lsun)**2]))[0]

            logL_acc=1.13*np.log10(LHa)+1.74 # Alcala 2017
            elogL_acc=np.sqrt((1.13*np.log10(LHa))**2*((LHa*0.05)**2+(1.13*eLHa/(LHa*np.log(10)))**2)+0.19**2)

            logR=np.log((R/Rsun).value)
            elogR=(eR/(R*np.log(10))).value

            logM=np.log((M/Msun).value)
            elogM=(eM/(M*np.log(10))).value

            logdM_acc=-7.39+logL_acc+logR-logM #DeMarchi 2011
            elogdM_acc=np.sqrt(elogL_acc**2+elogR**2+elogM**2)

            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'DHa%s'%label]=m658_c-m658_d
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eDHa%s'%label]=np.sqrt(e658**2+e658_c**2)

            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'EQW%s'%label]=EQW
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eEQW%s'%label]=eEQW

            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logLHa%s'%label]=np.log10(LHa)
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogLHa%s'%label]=np.log10((LHa+eLHa)/(LHa-eLHa))/2
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logL_acc%s'%label]=logL_acc
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogL_acc%s'%label]=elogL_acc
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logdM_acc%s'%label]=logdM_acc
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogdM_acc%s'%label]=elogdM_acc
        else:
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'DHa%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eDHa%s'%label]=np.nan

            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'EQW%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eEQW%s'%label]=np.nan

            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logLHa%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogLHa%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logL_acc%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogL_acc%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logdM_acc%s'%label]=np.nan
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogdM_acc%s'%label]=np.nan
            
    else:
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'DHa%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eDHa%s'%label]=np.nan

        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'EQW%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eEQW%s'%label]=np.nan

        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logLHa%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogLHa%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logL_acc%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogL_acc%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'logdM_acc%s'%label]=np.nan
        MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'elogdM_acc%s'%label]=np.nan
            
    return(MCMC_sim_df)

def star_accrention_properties(MCMC_sim_df,mean_df,interp_mags,mlabel,Rv,interp_658,DM,showplot=False,ID_list=[]): 
    # if Rv=='a': Av_list,Av_658=get_Av_list(3,mass=1,age=1,showplot=False)
    # elif Rv=='b': Av_list,Av_658=get_Av_list(5,mass=1,age=1,showplot=False)
    if Rv=='a': Av_list,Av_658=get_Av_list(interp_mags,interp_Tlogg,filter_label_list,mag_label_list,photflam,3,DM=DM_mass_age_4Av_test[0],mass=DM_mass_age_4Av_test[1],age=DM_mass_age_4Av_test[2],showplot=False,photlam658=1.977e-18)
    elif Rv=='b': Av_list,Av_658=get_Av_list(interp_mags,interp_Tlogg,filter_label_list,mag_label_list,photflam,5,DM=DM_mass_age_4Av_test[0],mass=DM_mass_age_4Av_test[1],age=DM_mass_age_4Av_test[2],showplot=False,photlam658=1.977e-18)

    
    print('> Working on accretion properties for stars in mlabel: %s Rv: %s'%(mlabel,Rv))
    MCMC_sim_df[['emass']]=MCMC_sim_df[['emass_d','emass_u']].mean(axis=1)
    MCMC_sim_df[['eAv']]=MCMC_sim_df[['eAv_d','eAv_u']].mean(axis=1)

    eA=MCMC_sim_df[['eA_d','eA_u']].mean(axis=1)
    MCMC_sim_df[['eA']]=eA

    eT=MCMC_sim_df[['eT_d','eT_u']].mean(axis=1)
    MCMC_sim_df[['eT']]=eT

    eL=MCMC_sim_df[['eL_d','eL_u']].mean(axis=1)
    MCMC_sim_df[['eL']]=eL

    MCMC_sim_df[['eL_corr']]=eL

    if len(ID_list) == 0: ID_list=MCMC_sim_df['ID'].unique()
    for ID in tqdm(ID_list): 
        color_good_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'good_cols'].values[0]
        if not np.all(np.isnan(color_good_list)) and np.any(color_good_list):
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'L_corr']=lum_corr(MCMC_sim_df,ID,interp_mags,Av_list,DM)

            mag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mags'].tolist()[0][:-2]
            emag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'emags'].tolist()[0][:-2]
            m435,m555,m775,m850=mag_list
            e435,e555,e775,e850=emag_list

            m658=mean_df.loc[mean_df.UniqueID==ID,'m658_p'].values[0]
            e658=mean_df.loc[mean_df.UniqueID==ID,'e658_p'].values[0]

            m435-=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]*Av_list[0]
            m555-=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]*Av_list[1]
            m658-=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]*Av_658
            m775-=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]*Av_list[2]
            m850-=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]*Av_list[3]


            m435_c=interp_mags[0](np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]),np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]))+DM
            m555_c=interp_mags[1](np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]),np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]))+DM
            m658_c=interp_658[0](np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]),np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]))+DM
            m775_c=interp_mags[2](np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]),np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]))+DM
            m850_c=interp_mags[3](np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]),np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]))+DM

            # dage=np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]+MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'eA'].values[0])
            # dmass=np.log10(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]+MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'emass'].values[0])
            e658_c=0#abs(m658_c-(interp_658_btsettl[0](dmass,dage)+DM))

            delta_list=np.array([m435-m435_c,m555-m555_c,m775-m775_c,m850-m850_c])

            if len(delta_list[~np.isnan(delta_list)])!=0:
                delta=np.average(delta_list[~np.isnan(delta_list)],weights=emag_list[~np.isnan(delta_list)]**(-2))

                m435_d=m435-delta
                m555_d=m555-delta
                m658_d=m658-delta
                m775_d=m775-delta
                m850_d=m850-delta

                if showplot:
                    display(MCMC_sim_df.loc[MCMC_sim_df.ID==ID])
                    plt.figure()
                    plt.plot([m435,m555,m775,m850],'or',ms=5)
                    plt.plot([m435_c,m555_c,m775_c,m850_c],'ok',ms=5)
                    plt.plot([m435_d,m555_d,m775_d,m850_d],'og',ms=4)
                    plt.show()
                if not np.isnan([m658_c-m658_d]):
                    MCMC_sim_df=accr_stats(MCMC_sim_df,ID,m658_c,m658_d,e658,e658_c,label='')
        else:
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['N', 'mass', 'emass', 'emass_d', 'emass_u', 
                        'Av', 'eAv', 'eAv_d', 'eAv_u',
                        'A', 'eA', 'eA_u', 'eA_d', 
                        'T', 'eT', 'eT_u', 'eT_d', 
                        'L', 'eL', 'eL_d', 'eL_u', 
                        'L_corr', 'eL_corr',
                        'DHa', 'eDHa',
                        'EQW', 'eEQW', 'logLHa', 'elogLHa', 'logL_acc', 'elogL_acc',
                        'logdM_acc', 'elogdM_acc']]=np.nan
    return(MCMC_sim_df)

