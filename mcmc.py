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

import sys,emcee,os
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append('./')
import mcmc_utils 
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
# import matplotlib.pyplot as plt
# from IPython.display import display

class MCMC():
    ############
    #Main body #
    ############
    
    def __init__(self,interp_mags,interp_colors,interp_star_properties,filter_label_list,mag_label_list,emag_label_list,color_label_list,ecolor_label_list,sat_list,Av_list,photflam,RW=74.96,dist=402,edist=5,DM_mass_age_4Av_test=[8.02,1,1],truths=[None,None,None],mlabel_Rv=['3','a'],var_range=[0.01,1.6],Av_range=[0,10],t_range=[0.5,100],nwalkers_ndim_niters=[50,3,10000],mu_sigma_m=[0.3,0.1],mu_sigma_t=[2,1],sigma_T=200,ID_label='UniqueID',Teff_label='Teff',DaRio_path='/media/giovanni/DATA_backup/Lavoro/Giovanni/NGC1976/ACS/DaRio_ACS_matched.csv',workers=8,err=None,err_max=0.05,err_min=0.001,blobs=False,show_test=True,progress=True,parallelize_runs=False,parallelize_sampler=False,simulation=False,physical_prior=True,Rv_physical_prior=True,t_physical_prior=True,var_physical_prior=True):
        self.interp_mags=interp_mags
        self.interp_colors=interp_colors
        self.interp_star_properties=interp_star_properties
        
        self.filter_label_list=filter_label_list
        self.mag_label_list=mag_label_list
        self.emag_label_list=emag_label_list
        self.color_label_list=color_label_list
        self.ecolor_label_list=ecolor_label_list
        self.sat_list=sat_list
        self.Av_list=Av_list
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
        
        self.mlabel,self.Rv=mlabel_Rv
        
        if self.mlabel=='2':
            self.cc=0.158
            self.cmu=0.079
            self.csig=0.69
        elif self.mlabel=='3':
            self.cc=0.086
            self.cmu=0.22
            self.csig=0.57
        else:
            self.cc=None
            self.cmu=None
            self.csig=None

        
        self.var_min,self.var_max=var_range
        self.Av_min,self.Av_max=Av_range
        self.t_min,self.t_max=t_range
        self.nwalkers,self.ndim,self.niters=nwalkers_ndim_niters
        self.mass2simulate,self.Av2simulate,self.age2simulate=truths        
        self.dist=dist
        
        self.sig_T=sigma_T
        self.mu_m,self.sig_m=[np.log(mu_sigma_m[0]**2/np.sqrt(mu_sigma_m[0]**2+mu_sigma_m[1]**2)),np.sqrt(np.log(1+mu_sigma_m[1]**2/mu_sigma_m[0]**2))]
        self.mu_t,self.sig_t=[np.log(mu_sigma_t[0]**2/np.sqrt(mu_sigma_t[0]**2+mu_sigma_t[1]**2)),np.sqrt(np.log(1+mu_sigma_t[1]**2/mu_sigma_t[0]**2))]
        
        self.workers=workers
        self.Teff_label=Teff_label
        self.ID_label=ID_label
        self.DaRio_path=DaRio_path
        self.DaRio_pdf=DaRio_dist(self.Av_min,self.Av_max,DaRio_finename=self.DaRio_path)

        self.use_physical_prior=physical_prior
        self.use_Rv_physical_prior=Rv_physical_prior
        self.use_t_physical_prior=t_physical_prior
        self.use_var_physical_prior=var_physical_prior
        
        self.show_test=show_test
        self.progress=progress
        self.parallelize_runs=parallelize_runs
        self.parallelize_sampler=parallelize_sampler
        self.simulation=simulation
        
        self.err=err
        self.err_max=err_max
        self.err_min=err_min
        self.blobs=blobs
        
        
    def run(self,mvs_df,avg_df,ID_list):    
        MCMC_sim_df=pd.DataFrame(columns=['ID','truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','L','eL_u','eL_d','Teff','T','eT_u','eT_d','L_corr','N'])
    
        MCMC_sim_df['ID']=ID_list
   
        if self.parallelize_runs: 
            ntarget=len(ID_list)
            num_of_chunks = 3*self.workers
            chunksize = ntarget // num_of_chunks
            if chunksize <=0:
                chunksize = 1
            print('> workers %i,chunksize %i,ntarget %i'%(self.workers,chunksize,ntarget))
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                for tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype,ID in tqdm(executor.map(self.aggregated_tasks,ID_list,repeat(mvs_df),repeat(avg_df),chunksize=chunksize)):
                    MCMC_sim_df=self.update_dataframe(ID,MCMC_sim_df,sampler,data,tau,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list)
                    # if len(data):
                    #     discard=int(self.niters/2)
                    #     thin=int(0.5*np.min(tau))
    
                    #     samples = sampler.get_chain(discard=discard, thin=thin)
                    #     flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
                    #     mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=mcmc_utils.star_properties(flat_samples)
                    #     MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','T','eT_u','eT_d','L','eL_u','eL_d']]=[[[self.mass2simulate,self.Av2simulate,self.age2simulate],samples.tolist(),flat_samples.tolist(),mag_good_list.tolist(),color_good_list,tau.tolist(),mag_list,emag_list,color_list,ecolor_list,mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d]]
    
        else:
            for ID in ID_list:
                tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype,ID=self.aggregated_tasks(ID,mvs_df,avg_df)
                MCMC_sim_df=self.update_dataframe(ID,MCMC_sim_df,sampler,data,tau,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list)
                # if len(data):
                #     discard=int(self.niters/2)
                #     thin=int(0.5*np.min(tau))
    
                #     samples = sampler.get_chain(discard=discard, thin=thin)
                #     flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
                #     mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=mcmc_utils.star_properties(flat_samples,self.ndim,self.interp_star_properties,self.mlabel)
                #     MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','T','eT_u','eT_d','L','eL_u','eL_d']]=[[[self.mass2simulate,self.Av2simulate,self.age2simulate],samples.tolist(),flat_samples.tolist(),mag_good_list.tolist(),color_good_list,tau.tolist(),mag_list,emag_list,color_list,ecolor_list,mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d]]
    
        return(MCMC_sim_df,mvs_df)

#     def accrention_properties(self,MCMC_sim_df):
#         MCMC_sim_df.loc[~MCMC_sim_df.good_cols.isna(),'N']=np.array(MCMC_sim_df.loc[~MCMC_sim_df.good_cols.isna(),'good_cols'].tolist()).astype(int).sum(axis=1)
#         MCMC_sim_df=mcmc_utils.star_accrention_properties(MCMC_sim_df,mean_df,interp_mags,mlabel,Rv,interp_658,DM_mass_age_4Av_test[0])
#         df.loc[df.UniqueID.isin(MCMC_sim_df.ID.unique()),['N_%s%s'%(mlabel,Rv),'mass_%s%s'%(mlabel,Rv), 'emass_%s%s'%(mlabel,Rv), 'emass_d_%s%s'%(mlabel,Rv), 'emass_u_%s%s'%(mlabel,Rv), 
#                                                         'Av_%s%s'%(mlabel,Rv), 'eAv_%s%s'%(mlabel,Rv), 'eAv_d_%s%s'%(mlabel,Rv), 'eAv_u_%s%s'%(mlabel,Rv),
#                                                         'A_%s%s'%(mlabel,Rv), 'eA_%s%s'%(mlabel,Rv), 'eA_u_%s%s'%(mlabel,Rv), 'eA_d_%s%s'%(mlabel,Rv), 
#                                                         'T_%s%s'%(mlabel,Rv), 'eT_%s%s'%(mlabel,Rv), 'eT_u_%s%s'%(mlabel,Rv), 'eT_d_%s%s'%(mlabel,Rv), 
#                                                         'L_%s%s'%(mlabel,Rv), 'eL_%s%s'%(mlabel,Rv), 'eL_d_%s%s'%(mlabel,Rv), 'eL_u_%s%s'%(mlabel,Rv), 
#                                                         'L_corr_%s%s'%(mlabel,Rv), 'eL_corr_%s%s'%(mlabel,Rv),
#                                                         'DHa_%s%s'%(mlabel,Rv), 'eDHa_%s%s'%(mlabel,Rv),
#                                                         'EQW_%s%s'%(mlabel,Rv), 'eEQW_%s%s'%(mlabel,Rv), 
#                                                         'logLHa_%s%s'%(mlabel,Rv), 'elogLHa_%s%s'%(mlabel,Rv), 'logL_acc_%s%s'%(mlabel,Rv), 'elogL_acc_%s%s'%(mlabel,Rv),
#                                                         'logdM_acc_%s%s'%(mlabel,Rv), 'elogdM_acc_%s%s'%(mlabel,Rv)]]=MCMC_sim_df[['N', 'mass', 'emass', 'emass_d', 'emass_u', 
#                                                         'Av', 'eAv', 'eAv_d', 'eAv_u',
#                                                         'A', 'eA', 'eA_u', 'eA_d', 
#                                                         'T', 'eT', 'eT_u', 'eT_d', 
#                                                         'L', 'eL', 'eL_d', 'eL_u', 
#                                                         'L_corr', 'eL_corr',
#                                                         'DHa', 'eDHa',
#                                                         'EQW', 'eEQW', 'logLHa', 'elogLHa', 'logL_acc', 'elogL_acc',
#                                                         'logdM_acc', 'elogdM_acc']].values
                                                                                                                                   
# return(MCMC_sim_df)
    
    #############
    # MCMC task #
    #############

    def pre_task(self,mvs_df,ID):
        if self.mlabel=='0':self.mu_T=mvs_df.loc[mvs_df[self.ID_label]==ID,self.Teff_label].values[0]
        else: self.mu_T=None

        if self.simulation: 
            mag_list=[]
            emag_list=[]
        else: 
            mag_list=mvs_df.loc[mvs_df[self.ID_label]==ID,self.mag_label_list].values[0]
            emag_list=mvs_df.loc[mvs_df[self.ID_label]==ID,self.emag_label_list].values[0]
        return(mag_list,emag_list)
    
    def task(self,mag_list,emag_list,mvs_df,avg_df):#,interp_mags,interp_color,filter_label_list,mag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,mu_sigma_T,DaRio_finename,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,mag_list,emag_list,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior):
        mag_list,emag_list,mag_temp_list,emag_temp_list,mag_good_list=mcmc_utils.simulate_mag_star(self.sat_list,self.interp_mags,self.mag_label_list,self.Av_list,mag_list=mag_list,emag_list=emag_list,mass=self.mass2simulate,Av1=self.Av2simulate,age=self.age2simulate,distance=self.dist,err=self.err,err_min=self.err_min,err_max=self.err_max,mvs_df=mvs_df,avg_df=avg_df)
        color_list,ecolor_list,Av1_color_list,color_good_list=mcmc_utils.simulate_color_star(mag_list,emag_list,self.Av_list,self.mag_label_list,self.color_label_list)
    
        self.data_variable_list=[color_list[color_good_list],ecolor_list[color_good_list]]
        self.Av1_variable_list=Av1_color_list[color_good_list]
        self.variables_interp=self.interp_colors[color_good_list]
        if self.blobs==True: dtype = [("%s"%i,np.float) for i in self.color_label_list[color_good_list]]
        else: dtype=None
        if self.show_test:
            table=QTable([self.mag_label_list,mag_good_list, self.sat_list,mag_temp_list,emag_temp_list,mag_list,emag_list,self.Av_list],
                    names=('mag_label','good' ,'sat_list','mags_temp','emags_temp', 'mags', 'emags','Av1'))
            table.pprint()
    
            table=QTable([self.color_label_list, color_good_list,color_list,ecolor_list,Av1_color_list],
                    names=('col_label', 'good',  'color', 'ecolor','Av1'))
            table.pprint()
    
        if any(color_good_list):
            pos = np.array([np.array([np.random.uniform(self.var_min,self.var_max),np.random.uniform(self.Av_min,self.Av_max),np.random.uniform(self.t_min,self.t_max)]) for i in range(self.nwalkers)])
            if self.parallelize_sampler:
                 with Pool(processes=self.workers) as pool:
                    if self.blobs: sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability,blobs_dtype=dtype, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                    else: sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])    
                    sampler.run_mcmc(pos, self.niters, progress=self.progress)
            else:
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability,blobs_dtype=dtype,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                sampler.run_mcmc(pos, self.niters, progress=self.progress)
            tau = sampler.get_autocorr_time(tol=0)
            return(tau,sampler,self.data_variable_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype)
    
        else:
            return(np.nan,[],[],mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,[])
      
    def aggregated_tasks(self,ID,mvs_df,avg_df):#,simulation,Teff_label,interp_mags,interp_colors,filter_label_list,mag_label_list,emag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,sigma_T,DaRio_path,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior):
        mag_list,emag_list=self.pre_task(mvs_df,ID)
        tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype=self.task(mag_list,emag_list,mvs_df,avg_df)#(interp_mags,interp_colors,filter_label_list,mag_label_list,color_label_list,sat_list,mlabel_Rv,var_range,Av_range,t_range,nwalkers_ndim_niters,mu_sigma_m,mu_sigma_t,mu_sigma_T,DaRio_path,Av_list,workers,mass2simulate,Av2simulate,age2simulate,distance,err,err_max,err_min,mag_list,emag_list,blobs,show_test,progress,parallelize_sampler,physical_prior,Rv_physical_prior,t_physical_prior,var_physical_prior)
        return(tau,sampler,data,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,dtype,ID)       

    def update_dataframe(self,ID,MCMC_sim_df,sampler,data,tau,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list):
        if len(data):
            discard=int(self.niters/2)
            thin=int(0.5*np.min(tau))
    
            samples = sampler.get_chain(discard=discard, thin=thin)
            flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
            mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=mcmc_utils.star_properties(flat_samples,self.ndim,self.interp_star_properties,self.mlabel)
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['truths','samples','flat_samples','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','Teff','T','eT_u','eT_d','L','eL_u','eL_d']]=[[[self.mass2simulate,self.Av2simulate,self.age2simulate],samples.tolist(),flat_samples.tolist(),mag_good_list.tolist(),color_good_list,tau.tolist(),mag_list,emag_list,color_list,ecolor_list,mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,self.mu_T,T,eT_u,eT_d,L,eL_u,eL_d]]
        return(MCMC_sim_df)

    ##################################
    # Probability functions          #
    # var can be either mass or Teff #
    ##################################
    
    def models(self,params):
        var_x,Av_x,t_x=params
        variables_list=[]
        for elno in range(np.array(self.variables_interp).shape[0]):
            x=self.variables_interp[elno](np.log10(var_x),np.log10(t_x))
            variables_list.append(x+Av_x*self.Av1_variable_list[elno])
        variables_list=np.array(variables_list)
        return(variables_list)
    
    def log_likelihood(self,params,y,yerr):
        ym=self.models(params)
        return(-0.5*np.sum((np.round(y,3)-np.round(ym,3))**2/yerr**2),np.round(ym,3))
    
    def log_prior(self,params):
        var_x,Av_x,t_x=params
        if (self.var_min <= var_x <= self.var_max) and (self.Av_min <= Av_x <= self.Av_max) and (self.t_min <= t_x <= self.t_max):
            if self.use_physical_prior==True:
                if self.use_Rv_physical_prior:
                    if self.Rv=='a': 
                        lp_Av=np.log(self.DaRio_pdf(Av_x)[0])
                    else: lp_Av=0
                else: lp_Av=0
                
                if self.use_t_physical_prior: lp_t=np.log(lognormal_dist(x=t_x,mu=self.mu_t,sig=self.sig_t))
                else: lp_t=0
                
                if self.use_var_physical_prior: 
                    if self.mlabel=='0': lp_var=np.log(normal_dist(x=var_x,mu=self.mu_T,sig=self.sig_T))
                    elif self.mlabel=='1': lp_var=np.log(kroupa_dist(mass=var_x)) 
                    elif self.mlabel=='2' or self.mlabel == '3': lp_var=np.log(chabrier_dist(mass=var_x,cc=self.cc,cmu=self.cmu,csig=self.csig))
                    elif self.mlabel=='4': lp_var=np.log(lognormal_dist(var_x,self.mu_m,self.sig_m))
                else: lp_var=0
                return(lp_var+lp_Av+lp_t)
                    
            else:         
                return(0.0)
        return(-np.inf)
    
    def log_probability(self,params):
        y,yerr=self.data_variable_list
        lp = self.log_prior(params)
        if not np.isfinite(lp):return(-np.inf)
    #         if len(y)==1:return(-np.inf,-np.inf)
    #         if len(y)==2:return(-np.inf,-np.inf,-np.inf)
    #         if len(y)==3:return(-np.inf,-np.inf,-np.inf,-np.inf)
    #         if len(y)==4:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
    #         if len(y)==5:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
    #         if len(y)==6:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
    #         return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
        ll,ym=self.log_likelihood(params,y,yerr)
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


