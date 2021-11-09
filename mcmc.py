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
from scipy.optimize import minimize

class MCMC():
    ############
    #Main body #
    ############
    
    def __init__(self,interp_mags,interp_colors,interp_star_properties,filter_label_list,mag_label_list,emag_label_list,color_label_list,ecolor_label_list,sat_list,Av_list,photflam,RW=74.96,dist=402,edist=5,truths=[None,None,None],vlabel_Rv=['3','a'],discard=None,thin=None,var_range=[0.01,1.6],Av_range=[0,10],t_range=[0.5,100],nwalkers_ndim_niters=[50,3,10000],mu_sigma_m=[0.3,0.1],mu_sigma_t=[2,1],ID_label='UniqueID',Teff_label='Teff',eTeff_label='eTeff',DaRio_path='/media/giovanni/DATA_backup/Lavoro/Giovanni/NGC1976/ACS/DaRio_ACS_matched.csv',workers=8,err=None,err_max=0.1,err_min=0.001,r2=4,blobs=False,nmag2fit=1,magnitude_fit=False,color_fit=False,magnitude_color_fit=False,show_test=True,progress=True,parallelize_runs=False,parallelize_sampler=False,simulation=False,physical_prior=True,Av_prior=True,t_prior=True,var_prior=True):
        '''
        This is the initialization step of the MCMC class. The MCMC can be run to fit 3 varables at the time. The variables for the fit are:
        [Var, Av, Age]. 'Var' can be either mass or Teff. 'Var*' refer to the other varibale not choosen, e.g. if Var = mass, then Var* = Teff and age, and vice versa.

        Parameters
        ----------
        interp_mags : list
            list of interpolation functions for magnitudes on log10(Var) and log10(age). They MUST follow the same order as filter_label_list.
        interp_colors : list
            list of interpolation functions for colors on log10(Var) and log10(age). They MUST follow the same order as filter_label_list.
        interp_star_properties : list
            list of interpolation functions for [Var*, L, logg] on log10(Var) and log10(age). They MUST follow the same order as filter_label_list.
        filter_label_list : list
            list of filter labels.
        mag_label_list : list
            list of magnitude labels. They MUST follow the same order as filter_label_list.
        emag_label_list : list
            list of error magnitude labels. They MUST follow the same order as filter_label_list.
        color_label_list : list
            list of color labels. They MUST follow the same order as filter_label_list.
        ecolor_label_list : TYPE
            list of error color labels. They MUST follow the same order as filter_label_list.
        sat_list : list
            list saturation values. They MUST follow the same order as filter_label_list.
        Av_list : list
            list of extinction values. They MUST follow the same order as filter_label_list.
        photflam : list
            list of phoyflam values. They MUST follow the same order as filter_label_list.
        RW : float, optional
            RW value for the estimate of accretion. The default is 74.96 for ACS.
        dist : float, optional
            distance of the target. The default is 402 for the ONC.
        edist : float, optional
            error on the distance of the target. The default is 5 for the ONC.
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
        var_range : list, optional
            minimum, maximum variable range for walkers to explore. Var can be either mass or Teff. The default is mass with [0.01,1.6].
        Av_range : list, optional
            minimum, maximum extinction range for walkers to explore. The default is [0,10].
        t_range : list, optional
            minimum, maximum age range for walkers to explore. The default is [0.5,100].
        nwalkers_ndim_niters : list, optional
            list of number of walker, dimension (number of variables) and number of steps for the MCMC run. The default is [50,3,10000].
        mu_sigma_m : list, optional
            list of mean and sigma for the gaussian prior on mass. The default is [0.3,0.1].
        mu_sigma_t : list, optional
            list of mean and sigma for the log-normal prior on age. The default is [2,1].
        ID_label : str, optional
            label identifing the IDs in the avg dataframe. The default is 'UniqueID'.
        Teff_label : str, optional
            label identifing the Teff in the avg dataframe. The default is 'Teff'.
        eTeff_label : str, optional
            label identifing the error on Teff in the avg dataframe. The default is 'eTeff'.
        DaRio_path : str, optional
            path to DaRio catalog to estimate the distribution of Av for the prior in the ONC. The default is '/media/giovanni/DATA_backup/Lavoro/Giovanni/NGC1976/ACS/DaRio_ACS_matched.csv'.
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
        physical_prior : bool, optional
            enable/disable the use of all priors in the likelihood. The default is True.
        Av_prior : bool, optional
            enable/disable the use of the Av prior in the likelihood. The default is True.
        t_prior : TYPE, optional
            enable/disable the use of the age prior in the likelihood. The default is True.
        var_prior : TYPE, optional
            enable/disable the use of the Var prior in the likelihood. The default is True.

        Returns
        -------
        None.

        '''
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
        
        self.dist=dist
        self.d=dist*u.pc.to(u.cm)*u.cm
        self.ed=edist*u.pc.to(u.cm)*u.cm
        self.DM=5*np.log10(dist/10)
        
        self.RW=RW
        
        self.vlabel,self.Rv=vlabel_Rv
        
        if self.vlabel=='0': physical_prior=False
        elif self.vlabel=='2':
            self.cc=0.158
            self.cmu=0.079
            self.csig=0.69
        elif self.vlabel=='3':
            self.cc=0.086
            self.cmu=0.22
            self.csig=0.57
        else:
            self.cc=None
            self.cmu=None
            self.csig=None

        self.discard=discard
        self.thin=thin
        self.var_min,self.var_max=var_range
        self.Av_min,self.Av_max=Av_range
        self.t_min,self.t_max=t_range
        self.nwalkers,self.ndim,self.niters=nwalkers_ndim_niters
        self.var2simulate,self.Av2simulate,self.age2simulate=truths        
        
        self.mu_m,self.sig_m=[np.log(mu_sigma_m[0]**2/np.sqrt(mu_sigma_m[0]**2+mu_sigma_m[1]**2)),np.sqrt(np.log(1+mu_sigma_m[1]**2/mu_sigma_m[0]**2))]
        self.mu_t,self.sig_t=[np.log(mu_sigma_t[0]**2/np.sqrt(mu_sigma_t[0]**2+mu_sigma_t[1]**2)),np.sqrt(np.log(1+mu_sigma_t[1]**2/mu_sigma_t[0]**2))]
        
        self.workers=workers
        self.Teff_label=Teff_label
        self.eTeff_label=eTeff_label
        self.ID_label=ID_label
        self.DaRio_path=DaRio_path
        self.DaRio_pdf=DaRio_dist(self.Av_min,self.Av_max,DaRio_finename=self.DaRio_path)

        self.physical_prior=physical_prior
        self.Av_prior=Av_prior
        self.t_prior=t_prior
        self.var_prior=var_prior
        
        if np.array([magnitude_fit,color_fit,magnitude_color_fit]).astype(int).sum()!=1:
            raise ValueError('Only one of the following: magnitude_fit, color_fit, magnitude_color_fit has to be true. Please choose which fit you intend to perform.')

        self.nmag2fit=nmag2fit
        self.magnitude_fit=magnitude_fit
        self.color_fit=color_fit
        self.magnitude_color_fit=magnitude_color_fit
        self.show_test=show_test
        self.progress=progress
        self.parallelize_runs=parallelize_runs
        self.parallelize_sampler=parallelize_sampler
        self.simulation=simulation
        
        self.err=err
        self.err_max=err_max
        self.err_min=err_min
        self.r2=r2
        self.blobs=blobs
        
        
    def run(self,avg_df,ID_list):
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
        MCMC_sim_df=pd.DataFrame(columns=['ID','variables','truths','samples','flat_samples','blobs','flat_blobs','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','L','eL_u','eL_d','Teff','eTeff','T','eT_u','eT_d','L_corr','N'])
    
        MCMC_sim_df['ID']=ID_list
   
        if self.parallelize_runs: 
            ntarget=len(ID_list)
            num_of_chunks = 3*self.workers
            chunksize = ntarget // num_of_chunks
            if chunksize <=0:
                chunksize = 1
            print('> workers %i,chunksize %i,ntarget %i'%(self.workers,chunksize,ntarget))
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                for tau,sampler,data,variable_label_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,ID,mu_T,sig_T in tqdm(executor.map(self.aggregated_tasks,ID_list,repeat(avg_df),chunksize=chunksize)):
                    MCMC_sim_df=self.update_dataframe(ID,MCMC_sim_df,sampler,data,tau,variable_label_list,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list,mu_T,sig_T)
    
        else:
            for ID in ID_list:
                tau,sampler,data,variable_label_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,ID,mu_T,sig_T=self.aggregated_tasks(ID,avg_df)
                MCMC_sim_df=self.update_dataframe(ID,MCMC_sim_df,sampler,data,tau,variable_label_list,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list,mu_T,sig_T)
    
        MCMC_sim_df.loc[~MCMC_sim_df.good_cols.isna(),'N']=np.array(MCMC_sim_df.loc[~MCMC_sim_df.good_cols.isna(),'good_cols'].tolist()).astype(int).sum(axis=1)
        return(MCMC_sim_df)

    def accrention_properties(self,MCMC_sim_df,avg_df,Av_list,interp_mags,interp_cols,interp_658,Av_658,zpt658,photlam658,showplot=False,ID_list=[],p='',s685=3,EQ_th=10,verbose=False):
        MCMC_sim_df=mcmc_utils.star_accrention_properties(self,MCMC_sim_df,avg_df,interp_mags,interp_cols,interp_658,self.DM,Av_list,Av_658,zpt658,photlam658,self.Msun,self.Lsun,self.eLsun,self.Rsun,self.d,self.ed,self.sigma,self.RW,showplot=showplot,ID_list=ID_list,p=p,s685=s685,EQ_th=EQ_th,verbose=verbose)
        avg_df.loc[avg_df.UniqueID.isin(MCMC_sim_df.ID.unique()),['N_%s%s'%(self.vlabel,self.Rv),'mass_%s%s'%(self.vlabel,self.Rv), 'emass_%s%s'%(self.vlabel,self.Rv), 'emass_d_%s%s'%(self.vlabel,self.Rv), 'emass_u_%s%s'%(self.vlabel,self.Rv), 
                                                        'Av_%s%s'%(self.vlabel,self.Rv), 'eAv_%s%s'%(self.vlabel,self.Rv), 'eAv_d_%s%s'%(self.vlabel,self.Rv), 'eAv_u_%s%s'%(self.vlabel,self.Rv),
                                                        'A_%s%s'%(self.vlabel,self.Rv), 'eA_%s%s'%(self.vlabel,self.Rv), 'eA_u_%s%s'%(self.vlabel,self.Rv), 'eA_d_%s%s'%(self.vlabel,self.Rv), 
                                                        'T_%s%s'%(self.vlabel,self.Rv), 'eT_%s%s'%(self.vlabel,self.Rv), 'eT_u_%s%s'%(self.vlabel,self.Rv), 'eT_d_%s%s'%(self.vlabel,self.Rv), 
                                                        'L_%s%s'%(self.vlabel,self.Rv), 'eL_%s%s'%(self.vlabel,self.Rv), 'eL_d_%s%s'%(self.vlabel,self.Rv), 'eL_u_%s%s'%(self.vlabel,self.Rv), 
                                                        'L_corr_%s%s'%(self.vlabel,self.Rv), 'eL_corr_%s%s'%(self.vlabel,self.Rv),
                                                        'DHa_%s%s'%(self.vlabel,self.Rv), 'eDHa_%s%s'%(self.vlabel,self.Rv),
                                                        'EQW_%s%s'%(self.vlabel,self.Rv), 'eEQW_%s%s'%(self.vlabel,self.Rv), 
                                                        'logLHa_%s%s'%(self.vlabel,self.Rv), 'elogLHa_%s%s'%(self.vlabel,self.Rv), 'logL_acc_%s%s'%(self.vlabel,self.Rv), 'elogL_acc_%s%s'%(self.vlabel,self.Rv),
                                                        'logdM_acc_%s%s'%(self.vlabel,self.Rv), 'elogdM_acc_%s%s'%(self.vlabel,self.Rv)]]=MCMC_sim_df[['N', 'mass', 'emass', 'emass_d', 'emass_u', 
                                                        'Av', 'eAv', 'eAv_d', 'eAv_u',
                                                        'A', 'eA', 'eA_u', 'eA_d', 
                                                        'T', 'eT', 'eT_u', 'eT_d', 
                                                        'L', 'eL', 'eL_d', 'eL_u', 
                                                        'L_corr', 'eL_corr',
                                                        'DHa', 'eDHa',
                                                        'EQW', 'eEQW', 'logLHa', 'elogLHa', 'logL_acc', 'elogL_acc',
                                                        'logdM_acc', 'elogdM_acc']].values
                                                                                                                                   
        return(MCMC_sim_df,avg_df)
    
    #############
    # MCMC task #
    #############

    def pre_task(self,avg_df,ID):
        if self.vlabel=='4':
            if self.simulation: 
                mu_T=self.var2simulate
                sig_T=150
            else:
                mu_T=avg_df.loc[avg_df[self.ID_label]==ID,self.Teff_label].values[0]
                sig_T=avg_df.loc[avg_df[self.ID_label]==ID,self.eTeff_label].values[0]
        else: 
            mu_T=None
            sig_T=None
            
        if self.simulation: 
            mag_list=[]
            emag_list=[]
        else: 
            mag_list=avg_df.loc[avg_df[self.ID_label]==ID,self.mag_label_list].values[0]
            emag_list=avg_df.loc[avg_df[self.ID_label]==ID,self.emag_label_list].values[0]
        return(mag_list,emag_list,mu_T,sig_T)
    
    def task(self,mag_list,emag_list,avg_df):
        global data_variable_list,Av1_variable_list,variables_interp
        mag_list,emag_list,mag_temp_list,emag_temp_list,mag_good_list=mcmc_utils.simulate_mag_star(self.sat_list,self.interp_mags,self.mag_label_list,self.Av_list,mag_list=mag_list,emag_list=emag_list,var=self.var2simulate,Av1=self.Av2simulate,age=self.age2simulate,distance=self.dist,err=self.err,err_min=self.err_min,err_max=self.err_max,avg_df=avg_df)
        # !!!!!!!!!!!!! Here there is a hack. You need to remove it!!!!!!#
        # emag_temp_list=emag_temp_list+0.05
        # emag_list=emag_list+0.05
        ##################################################################
        color_list,ecolor_list,Av1_color_list,color_good_list=mcmc_utils.simulate_color_star(mag_list,emag_list,self.Av_list,self.mag_label_list,self.color_label_list)
        mag_list=np.round(mag_list,self.r2)
        mag_temp_list=np.round(mag_temp_list,self.r2)
        color_list=np.round(color_list,self.r2)
        
        emag_temp_list=np.round(emag_temp_list,self.r2)
        emag_list=np.round(emag_list,self.r2)
        ecolor_list=np.round(ecolor_list,self.r2)
        self.Av_list=np.round(self.Av_list,self.r2)
        Av1_color_list=np.round(Av1_color_list,self.r2)
        if self.show_test:
            print('> Input data:')
            table=QTable([self.mag_label_list,mag_good_list, self.sat_list,mag_temp_list,['%s/%s'%(self.err_min,self.err_max)]*len(self.mag_label_list),emag_temp_list,mag_list,emag_list,self.Av_list],
                    names=('mag_label','good' ,'sat_list','mags_temp','emag_th','emags_temp', 'mags', 'emags','Av1'))
            table.pprint()
            if self.color_fit or self.magnitude_color_fit:
                table=QTable([self.color_label_list, color_good_list,color_list,ecolor_list,Av1_color_list],
                        names=('col_label', 'good',  'color', 'ecolor','Av1'))
                table.pprint()
        
        
        
        if self.magnitude_color_fit:
            if self.nmag2fit>len(mag_good_list):self.nmag2fit=len(mag_good_list)
            variable_label_list=np.append(self.mag_label_list[mag_good_list][0:self.nmag2fit],self.color_label_list[color_good_list])
            # variable_label_list=np.append(self.mag_label_list[mag_good_list][0:self.nmag2fit],self.color_label_list[color_good_list][-1])
            if self.blobs==True: self.dtype = [("%s"%i,np.float) for i in variable_label_list]
            else: self.dtype=None
            print('> Magnitude and Color fit')
            data_variable_list=[np.append(mag_list[mag_good_list][0:self.nmag2fit],color_list[color_good_list]),np.append(emag_list[mag_good_list][0:self.nmag2fit],ecolor_list[color_good_list])]
            Av1_variable_list=np.append(self.Av_list[mag_good_list][0:self.nmag2fit],Av1_color_list[color_good_list])
            variables_interp=np.append(self.interp_mags[mag_good_list][0:self.nmag2fit],self.interp_colors[color_good_list])           
            # data_variable_list=[np.append(mag_list[mag_good_list][0:self.nmag2fit],color_list[color_good_list][-1]),np.append(emag_list[mag_good_list][0:self.nmag2fit],ecolor_list[color_good_list][-1])]
            # Av1_variable_list=np.append(self.Av_list[mag_good_list][0:self.nmag2fit],Av1_color_list[color_good_list][-1])
            # variables_interp=np.append(self.interp_mags[mag_good_list][0:self.nmag2fit],self.interp_colors[color_good_list][-1])           
        elif self.color_fit:
            print('> Color fit')
            variable_label_list=self.color_label_list[color_good_list]
            if self.blobs==True: self.dtype = [("%s"%i,np.float) for i in variable_label_list]
            else: self.dtype=None
            data_variable_list=[color_list[color_good_list],ecolor_list[color_good_list]]
            Av1_variable_list=Av1_color_list[color_good_list]
            variables_interp=self.interp_colors[color_good_list]
            if len(color_list[color_good_list])<1: 
                raise ValueError('Need at least 1 color to perform the fit')
        elif self.magnitude_fit:
            print('> Magnitude fit')
            variable_label_list=self.mag_label_list[mag_good_list]
            if self.blobs==True: self.dtype = [("%s"%i,np.float) for i in variable_label_list]
            else: self.dtype=None
            data_variable_list=[mag_list[mag_good_list],emag_list[mag_good_list]]
            Av1_variable_list=self.Av_list[mag_good_list]
            variables_interp=self.interp_mags[mag_good_list]
            if len(mag_list[mag_good_list])<2: 
                raise ValueError('Need at least 2 magnitudes to perform the fit')

    
        if any(color_good_list):
            pos = np.array([np.array([np.random.uniform(self.var_min,self.var_max),np.random.uniform(self.Av_min,self.Av_max),np.random.uniform(self.t_min,self.t_max)]) for i in range(self.nwalkers)])
            # self.find_starting_minimum()
            
            if self.parallelize_sampler:
                 with Pool(processes=self.workers) as pool:
                    # if self.blobs: sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability,blobs_dtype=dtype, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                    # else: sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])    
                    # sampler.run_mcmc(pos, self.niters, progress=self.progress)
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, pool=pool,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                    sampler.run_mcmc(pos, self.niters, progress=self.progress)
            else:
                if self.blobs:sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, blobs_dtype=self.dtype,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                else: sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability,moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                sampler.run_mcmc(pos, self.niters, progress=self.progress)
            tau = sampler.get_autocorr_time(tol=0)
            return(tau,sampler,data_variable_list,variable_label_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list)
    
        else:
            return(np.nan,[],[],variable_label_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,[])
      
    def aggregated_tasks(self,ID,avg_df):
        global mu_T,sig_T

        mag_list,emag_list,mu_T,sig_T=self.pre_task(avg_df,ID)
        tau,sampler,data,variable_label_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list=self.task(mag_list,emag_list,avg_df)
        return(tau,sampler,data,variable_label_list,mag_list,emag_list,mag_good_list,color_list,ecolor_list,color_good_list,ID,mu_T,sig_T)       

    def update_dataframe(self,ID,MCMC_sim_df,sampler,data,tau,variable_label_list,mag_good_list,color_good_list,mag_list,emag_list,color_list,ecolor_list,mu_T,sig_T):
        if len(data):
            if self.discard==None: self.discard=int(self.niters/2)
            # if self.thin==None: self.thin=int(0.5*np.min(tau))
            # if self.discard==None: self.discard=int(2 * np.max(tau))
            if self.thin==None: self.thin=int(0.5 * np.min(tau))
    
            blobs = sampler.get_blobs(discard=self.discard, thin=self.thin)
            flat_blobs = sampler.get_blobs(discard=self.discard, thin=self.thin,flat=True)
            samples = sampler.get_chain(discard=self.discard, thin=self.thin)
            flat_samples = sampler.get_chain(discard=self.discard, thin=self.thin, flat=True)
            mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=mcmc_utils.star_properties(flat_samples,self.ndim,self.interp_star_properties,self.vlabel)

            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,['variables','truths','samples','flat_samples','blobs','flat_blobs','good_mags','good_cols','tau','mags','emags','cols','ecols','mass','emass_u','emass_d','Av','eAv_u','eAv_d','A','eA_u','eA_d','Teff','eTeff','T','eT_u','eT_d','L','eL_u','eL_d']]=[[variable_label_list.tolist(),[self.var2simulate,self.Av2simulate,self.age2simulate],samples.tolist(),flat_samples.tolist(),blobs.tolist(),flat_blobs.tolist(),mag_good_list.tolist(),color_good_list,tau.tolist(),mag_list,emag_list,color_list,ecolor_list,mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,mu_T,sig_T,T,eT_u,eT_d,L,eL_u,eL_d]]
        return(MCMC_sim_df)

    ##################################
    # Probability functions          #
    # var can be either mass or Teff #
    ##################################

    def log_likelihood(self,params,y,yerr):
        if self.magnitude_color_fit:
            ym=self.magnitude_color_from_models(params)
        elif self.color_fit:
            ym=self.color_from_models(params)
        elif self.magnitude_fit:
            ym=self.mags_from_models(params)

        delta2=(y-ym)**2
        sigma2=yerr**2
        # return(-0.5*np.sum((y-ym)**2/yerr**2),ym)
        return(-0.5*np.sum(delta2/sigma2+np.log(sigma2)),ym)
    
    def log_prior(self,params):
        var_x,Av_x,t_x=params
        if (self.var_min <= var_x <= self.var_max) and (self.Av_min <= Av_x <= self.Av_max) and (self.t_min <= t_x <= self.t_max):
            if self.physical_prior==True:
                if self.Av_prior:
                    if self.Rv=='a': 
                        lp_Av=np.log(self.DaRio_pdf(Av_x)[0])
                    else: lp_Av=0
                else: lp_Av=0
                
                if self.t_prior: lp_t=np.log(lognormal_dist(x=t_x,mu=self.mu_t,sig=self.sig_t))
                else: lp_t=0
                
                if self.var_prior: 
                    if self.vlabel=='1': lp_var=np.log(kroupa_dist(mass=var_x)) 
                    elif self.vlabel=='2' or self.vlabel == '3': lp_var=np.log(chabrier_dist(mass=var_x,cc=self.cc,cmu=self.cmu,csig=self.csig))
                    elif self.vlabel=='4': lp_var=np.log(normal_dist(x=var_x,mu=mu_T,sig=sig_T))
                else: lp_var=0
                return(lp_var+lp_Av+lp_t)
                    
            else:         
                return(0.0)
        return(-np.inf)
    
    def log_probability(self,params):
        y,yerr=data_variable_list
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            if not self.blobs:return(-np.inf)
            else:
                if len(y)==1:return(-np.inf,-np.inf)
                elif len(y)==2:return(-np.inf,-np.inf,-np.inf)
                elif len(y)==3:return(-np.inf,-np.inf,-np.inf,-np.inf)
                elif len(y)==4:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
                elif len(y)==5:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
                elif len(y)==6:return(-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
        ll,ym=self.log_likelihood(params,y,yerr)
        if not self.blobs:return(lp+ll)
        else:
            if len(y)==1: return(lp+ll,ym[0])
            elif len(y)==2: return(lp+ll,ym[0],ym[1])
            elif len(y)==3: return(lp+ll,ym[0],ym[1],ym[2])
            elif len(y)==4: return(lp+ll,ym[0],ym[1],ym[2],ym[3])
            elif len(y)==5: return(lp+ll,ym[0],ym[1],ym[2],ym[3],ym[4])
            elif len(y)==6: return(lp+ll,ym[0],ym[1],ym[2],ym[3],ym[4],ym[5])
    
    ######################
    # Ancillary routines #
    ######################
    
    def magnitude_color_from_models(self,params):
        var_x,Av_x,t_x=params
        variables_list=[]

        for elno in range(0,self.nmag2fit):
            x=variables_interp[elno](np.log10(var_x),np.log10(t_x))
            variables_list.append(x+Av_x*Av1_variable_list[elno]+self.DM)

        for elno in range(self.nmag2fit,len(variables_interp)):
            x=variables_interp[elno](np.log10(var_x),np.log10(t_x))
            variables_list.append(x+Av_x*Av1_variable_list[elno])
        variables_list=np.array(variables_list)
        # print(variables_list)
        return(variables_list)
    
    def color_from_models(self,params):
        var_x,Av_x,t_x=params
        variables_list=[]
        for elno in range(len(variables_interp)):
            x=variables_interp[elno](np.log10(var_x),np.log10(t_x))
            variables_list.append(x+Av_x*Av1_variable_list[elno])
        variables_list=np.array(variables_list)
        return(variables_list)
    
    def mags_from_models(self,params):
        var_x,Av_x,t_x=params
        variables_list=[]
        for elno in range(len(variables_interp)):
            x=variables_interp[elno](np.log10(var_x),np.log10(t_x))
            variables_list.append(x+Av_x*Av1_variable_list[elno]+self.DM)
        variables_list=np.array(variables_list)
        return(variables_list)

    def find_starting_minimum(self):
        y,yerr=data_variable_list
        params = np.array([0.3,1,1]) + 0.1 * np.random.randn(3)
        nll = lambda *args: -self.log_likelihood(*args)[0]
        soln = minimize(nll, params, args=(y,yerr))
        sys.exit(soln.x)
        
