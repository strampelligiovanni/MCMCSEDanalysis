#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:36:26 2021

@author: giovanni
"""
import sys,os
from kde import KDE
import numpy as np

from astropy import units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import bz2
from itertools import repeat
import _pickle as cPickle
import concurrent.futures
from IPython.display import display
from numpy import trapz
from astropy.stats import sigma_clip
from mcmc_plots import sample_posteriors
from synphot.units import FLAM
from dust_extinction.parameter_averages import CCM89
from synphot.reddening import ExtinctionCurve
from astropy import units as u
from synphot import ExtinctionModel1D,Observation,SourceSpectrum,SpectralElement,Empirical1D
from astropy.time import Time
import stsynphot as stsyn

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

########################
# Simulated photometry #
########################


def simulate_mag_star(ID,sat_list,variables_interp_in,mag_variable_in,Av1_list,ID_label='avg_ids',mag_list=[],emag_list=[],var=None,Av1=None,age=None,logSPacc=None,parallax=None,err=None,err_min=0.01,err_max=0.1,avg_df=None,f_spx_max=3):#,sim_label='Teff'):
    if len(mag_list)==0 and len(emag_list)==0: 
        distance=(parallax* u.pc).to(u.mas, equivalencies=u.parallax()).value
        DM=5*np.log10(distance/10)
        mag_temp_list=np.round(np.array([float(variables_interp_in[i](np.log10(var),np.log10(age),logSPacc))+Av1_list[i]*Av1+DM for i in mag_variable_in]),3)
        emag_temp_list=[]
        if err!=None:
            emag_list=np.array([err]*len(mag_temp_list))
        else:
            emag_list=[]
            for i in range(len(mag_variable_in)):
                mag_label='m%s'%(mag_variable_in[i][1:4])
                emag_label='e%s'%(mag_variable_in[i][1:4])
                if mag_label in avg_df.columns: df=avg_df
                else: raise ValueError('%s found in no dataframe'%mag_label)
                err_temp=np.nanmedian(df.loc[(df[mag_label]>=mag_temp_list[i]-0.5)&(df[mag_label]<=mag_temp_list[i]+0.5),emag_label].values)            
                emag_list.append(err_temp+0.02)
        emag_list=np.array(emag_list)   
        mag_list=np.array([mag_temp_list[i]+np.random.normal(0,scale=emag_list[i]) for i in range(len(mag_temp_list))])
    else:
        mag_temp_list=np.copy(mag_list)
        if err!=None: emag_list=np.array([err]*len(mag_temp_list))
        # else: emag_list+=0.02

    emag_temp_list=np.copy(emag_list)
    mag_good_list=[True]*len(mag_variable_in)
    for elno in range(len(mag_variable_in)):
        if sat_list[elno] == 'N/A':
            f_spx=avg_df.loc[avg_df[ID_label]==ID,'f_%s'%mag_variable_in[elno]].values[0]
            if (np.isnan(mag_list[elno]))|(np.isnan(emag_list[elno]))|(emag_list[elno] >=err_max)|(f_spx==f_spx_max): mag_good_list[elno]=False
        else:
            if (np.isnan(mag_list[elno]))|(np.isnan(emag_list[elno]))|(emag_list[elno] >=err_max)|(mag_list[elno]<=sat_list[elno]): 
                mag_good_list[elno]=False
            
    # mag_good_list=(emag_list<=err_max)&(mag_list>=sat_list)
    mag_good_list=np.array(mag_good_list)
    mag_list[~mag_good_list]=np.nan
    emag_list[~mag_good_list]=np.nan
    # return(np.round(mag_list,4),np.round(emag_list,4),np.round(mag_temp_list,4),np.round(emag_temp_list,4),mag_good_list)
    return(mag_list,emag_list,mag_temp_list,emag_temp_list,mag_good_list)

def simulate_color_star(mag_list,emag_list,Av1_list,mag_label_list,color_label_list):
    color_list=[]
    ecolor_list=[]
    Av1_color_list=[]
    color_good_list=[]
    for color_label in color_label_list:
        mag1_label=color_label.split('-')[0]
        mag2_label=color_label.split('-')[1]
        j=np.where(mag1_label == mag_label_list)[0][0]
        k=np.where(mag2_label == mag_label_list)[0][0]
        color_list.append(mag_list[j]-mag_list[k])
        ecolor_list.append(np.sqrt(emag_list[j]**2+emag_list[k]**2))
        # Av1_color_list.append(Av1_list[j]-Av1_list[k])
        Av1_color_list.append(Av1_list[mag1_label]-Av1_list[mag2_label])
        if np.isnan(color_list[-1]): color_good_list.append(False)
        else: color_good_list.append(True)
    return(np.array(color_list),np.array(ecolor_list),np.array(Av1_color_list),np.array(color_good_list))

def get_Av_list(filter_label_list, date='2005-01-1',xlim=[3000, 15000], ylim=[1e-10, 1e-8], verbose=False, Av=1, Rv=3.1):
    obsdate = Time(date).mjd
    vegaspec = SourceSpectrum.from_vega()
    Dict = {}

    wav = np.arange(3000, 15000, 10) * u.AA
    extinct = CCM89(Rv=Rv)
    ex = ExtinctionCurve(ExtinctionModel1D, points=wav, lookup_table=extinct.extinguish(wav, Av=Av))
    vegaspec_ext = vegaspec * ex

    band = SpectralElement.from_filter('johnson_v')  # 555
    sp_obs = Observation(vegaspec_ext, band)
    sp_obs_before = Observation(vegaspec, band)

    sp_stim_before = sp_obs_before.effstim(flux_unit='vegamag', vegaspec=vegaspec)
    sp_stim = sp_obs.effstim(flux_unit='vegamag', vegaspec=vegaspec)

    if verbose:
        print('before dust, V =', np.round(sp_stim_before, 4))
        print('after dust, V =', np.round(sp_stim, 4))
        flux_spectrum_norm = vegaspec(wav).to(FLAM, u.spectral_density(wav))
        flux_spectrum_ext = vegaspec_ext(wav).to(FLAM, u.spectral_density(wav))

        plt.semilogy(wav, flux_spectrum_norm, label='Av = 0')
        plt.semilogy(wav, flux_spectrum_ext, label='Av = %s' % Av)
        plt.legend()
        plt.ylabel('Flux [FLAM]')
        plt.xlabel('Wavelength [A]')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()

        # Calculate extinction and compare to our chosen value.
        Av_calc = sp_stim - sp_stim_before
        print('Av = ', np.round(Av_calc, 4))

    if any('johnson' in string for string in filter_label_list):
        for filter in filter_label_list:
            obs = Observation(vegaspec, SpectralElement.from_filter(filter))
            obs_ext = Observation(vegaspec_ext, SpectralElement.from_filter(filter))
            if verbose:
                # print('AV=0 %s'%filter,obs.effstim('vegamag',vegaspec=vegaspec))
                print('AV=1 %s' % filter, np.round(
                    obs_ext.effstim('vegamag', vegaspec=vegaspec) - obs.effstim('vegamag', vegaspec=vegaspec), 4))
            Dict[filter] = np.round(
                (obs_ext.effstim('vegamag', vegaspec=vegaspec) - obs.effstim('vegamag', vegaspec=vegaspec)).value, 4)
    else:
        for filter in filter_label_list:
            if filter in ['F130N', 'F139M']:
                obs = Observation(vegaspec, stsyn.band('wfc3,ir,%s' % filter.lower()))
                obs_ext = Observation(vegaspec_ext, stsyn.band('wfc3,ir,%s' % filter.lower()))
            elif filter in ['F336W', 'F439W', 'F656N', 'F814W']:
                obs = Observation(vegaspec, stsyn.band('acs,wfpc2,%s' % filter.lower()))
                obs_ext = Observation(vegaspec_ext, stsyn.band('acs,wfpc2,%s' % filter.lower()))
            elif filter in ['F110W', 'F160W']:
                obs = Observation(vegaspec, stsyn.band('nicmos,3,%s' % filter.lower()))
                obs_ext = Observation(vegaspec_ext, stsyn.band('nicmos,3,%s' % filter.lower()))
            else:
                obs = Observation(vegaspec, stsyn.band(f'acs,wfc1,%s,mjd#{obsdate}' % filter.lower()))
                obs_ext = Observation(vegaspec_ext, stsyn.band(f'acs,wfc1,%s,mjd#{obsdate}' % filter.lower()))

            if verbose:
                # print('AV=0 %s'%filter,obs.effstim('vegamag',vegaspec=vegaspec))
                print('AV=1 %s' % filter, np.round(
                    obs_ext.effstim('vegamag', vegaspec=vegaspec) - obs.effstim('vegamag', vegaspec=vegaspec), 4))
            Dict['m%s' % filter[1:4]] = np.round(
                (obs_ext.effstim('vegamag', vegaspec=vegaspec) - obs.effstim('vegamag', vegaspec=vegaspec)).value, 4)
    return (Dict)

def truth_list(mass,Av,age,mass_lim=[0.1,0.9],Av_lim=[0,10],age_lim=[0,100]):
    if mass==None:mass=np.round(random.uniform(mass_lim[0],mass_lim[1]),4)
    if Av==None: Av=np.round(random.uniform(Av_lim[0],Av_lim[1]),4)
    if age==None:age=np.round(random.uniform(age_lim[0],age_lim[1]),4)
    return(mass,Av,age)
    
def read_samples(filename):
    with bz2.BZ2File(filename, 'r') as f:
         x = cPickle.load(f)
    return(x)

####################
# Update dataframe #
####################

def update_dataframe(df,file_list,interp,workers=10,chunksize = 30,ID_label='avg_ids',kde_fit=False,discard=0,thin=1,label_list=['logMass','logAv','logAge','logSPacc','Parallax'],path2savedir=None,path2loaddir=None,pmin=1.66,pmax=3.30,verbose=False,showplots=False,sigma=3.5, parallel_runs=False):
    ntarget=len(file_list)
    if kde_fit:
        for file in tqdm(file_list):
            ID,logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,Dist,eDist_u,eDist_d,area_r=task(file,interp,kde_fit,discard,thin,label_list,path2savedir=path2savedir,path2loaddir=path2loaddir,pmin=pmin,pmax=pmax,verbose=verbose,showplots=showplots,sigma=sigma)
            df.loc[df[ID_label]==ID,['MCMC_mass','MCMC_emass_u','MCMC_emass_d','MCMC_Av','MCMC_eAv_u','MCMC_eAv_d','MCMC_A','MCMC_eA_u','MCMC_eA_d','MCMC_T','MCMC_eT_u','MCMC_eT_d','MCMC_logL','MCMC_elogL_u','MCMC_elogL_d','MCMC_logSPacc','MCMC_elogSPacc_u','MCMC_elogSPacc_d','MCMC_logLacc','MCMC_elogLacc_u','MCMC_elogLacc_d','MCMC_logMacc','MCMC_elogMacc_u','MCMC_elogMacc_d','MCMC_Parallax','MCMC_eParallax_d','MCMC_eParallax_u','MCMC_d','MCMC_ed_u','MCMC_ed_d','MCMC_area_r']]=[[10**logMass, 10**(logMass+elogMass_u)-10**logMass, 10**logMass-10**(logMass-elogMass_d),10**logAv, 10**(logAv+elogAv_u)-10**logAv, 10**logAv-10**(logAv-elogAv_d),10**logAge, 10**(logAge+elogAge_u)-10**logAge, 10**logAge-10**(logAge-elogAge_d),T,eT_u,eT_d,logL,elogL_u,elogL_d,logSPacc,elogSPacc_u,elogSPacc_d,logLacc,elogLacc_u,elogLacc_d,logMacc,elogMacc_u,elogMacc_d,Parallax, eParallax_u, eParallax_d,Dist,eDist_u,eDist_d,area_r]]
    else:
        if parallel_runs:
            print('> workers %i,chunksize %i,ntarget %i'%(workers,chunksize,ntarget))
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for ID,logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,Dist,eDist_u,eDist_d,area_r in tqdm(executor.map(task,file_list,repeat(interp),repeat(kde_fit),repeat(discard),repeat(thin),repeat(label_list),repeat(path2savedir),repeat(path2loaddir),repeat(pmin),repeat(pmax),repeat(verbose),repeat(showplots),repeat(sigma),chunksize=chunksize)):
                    df.loc[df[ID_label]==ID,['MCMC_mass','MCMC_emass_u','MCMC_emass_d','MCMC_Av','MCMC_eAv_u','MCMC_eAv_d','MCMC_A','MCMC_eA_u','MCMC_eA_d','MCMC_T','MCMC_eT_u','MCMC_eT_d','MCMC_logL','MCMC_elogL_u','MCMC_elogL_d','MCMC_logSPacc','MCMC_elogSPacc_u','MCMC_elogSPacc_d','MCMC_logLacc','MCMC_elogLacc_u','MCMC_elogLacc_d','MCMC_logMacc','MCMC_elogMacc_u','MCMC_elogMacc_d','MCMC_Parallax','MCMC_eParallax_d','MCMC_eParallax_u','MCMC_d','MCMC_ed_u','MCMC_ed_d','MCMC_area_r']]=[[10**logMass, 10**(logMass+elogMass_u)-10**logMass, 10**logMass-10**(logMass-elogMass_d),10**logAv, 10**(logAv+elogAv_u)-10**logAv, 10**logAv-10**(logAv-elogAv_d),10**logAge, 10**(logAge+elogAge_u)-10**logAge, 10**logAge-10**(logAge-elogAge_d),T,eT_u,eT_d,logL,elogL_u,elogL_d,logSPacc,elogSPacc_u,elogSPacc_d,logLacc,elogLacc_u,elogLacc_d,logMacc,elogMacc_u,elogMacc_d,Parallax, eParallax_u, eParallax_d,Dist,eDist_u,eDist_d,area_r]]
        else:
            for file in tqdm(file_list):
                ID,logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,Dist,eDist_u,eDist_d,area_r=task(file,interp,kde_fit,discard,thin,label_list,path2savedir=path2savedir,path2loaddir=path2loaddir,pmin=pmin,pmax=pmax,verbose=verbose,showplots=showplots,sigma=sigma)
                df.loc[df[ID_label]==ID,['MCMC_mass','MCMC_emass_u','MCMC_emass_d','MCMC_Av','MCMC_eAv_u','MCMC_eAv_d','MCMC_A','MCMC_eA_u','MCMC_eA_d','MCMC_T','MCMC_eT_u','MCMC_eT_d','MCMC_logL','MCMC_elogL_u','MCMC_elogL_d','MCMC_logSPacc','MCMC_elogSPacc_u','MCMC_elogSPacc_d','MCMC_logLacc','MCMC_elogLacc_u','MCMC_elogLacc_d','MCMC_logMacc','MCMC_elogMacc_u','MCMC_elogMacc_d','MCMC_Parallax','MCMC_eParallax_d','MCMC_eParallax_u','MCMC_d','MCMC_ed_u','MCMC_ed_d','MCMC_area_r']]=[[10**logMass, 10**(logMass+elogMass_u)-10**logMass, 10**logMass-10**(logMass-elogMass_d),10**logAv, 10**(logAv+elogAv_u)-10**logAv, 10**logAv-10**(logAv-elogAv_d),10**logAge, 10**(logAge+elogAge_u)-10**logAge, 10**logAge-10**(logAge-elogAge_d),T,eT_u,eT_d,logL,elogL_u,elogL_d,logSPacc,elogSPacc_u,elogSPacc_d,logLacc,elogLacc_u,elogLacc_d,logMacc,elogMacc_u,elogMacc_d,Parallax, eParallax_u, eParallax_d,Dist,eDist_u,eDist_d,area_r]]

    return(df)

def task(file,interp,kde_fit=False,discard=0,thin=1,label_list=['logMass','logAv','logAge','logSPacc','Parallax'],path2savedir=None,path2loaddir=None,pmin=1.66,pmax=3.30,verbose=False,showplots=False,sigma=3.5):

    ndim=len(label_list)
    ID=float(file.split('_')[-1])
    mcmc_dict=read_samples(file)  
    samples=np.array(mcmc_dict['samples'])
    if len(samples)>0:
        if not verbose:
            with HiddenPrints():
                logMass, elogMass_u, elogMass_d, logAv, elogAv_u, elogAv_d, logAge, elogAge_u, elogAge_d, logSPacc, elogSPacc_u, elogSPacc_d, Parallax, eParallax_u, eParallax_d, T, eT_u, eT_d, logL, elogL_d, elogL_u, logLacc, elogLacc_d, elogLacc_u, logMacc, elogMacc_d, elogMacc_u, kde_list, area_r = sample_posteriors(
                    interp, float(ID), ndim, verbose=verbose, fx=10, fy=10, show_samples=False, showplots=showplots,
                    bins=10, kde_fit=kde_fit, return_fig=False, return_variables=True, path2savedir=path2savedir,
                    path2loaddir=path2loaddir, pranges=None, pmin=pmin, pmax=pmax, sigma=sigma)

        else:
            logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,kde_list,area_r=sample_posteriors(interp,float(ID),ndim,verbose=verbose,fx=10,fy=10,show_samples=False,showplots=showplots,bins=10,kde_fit=kde_fit,return_fig=False,return_variables=True,path2savedir=path2savedir,path2loaddir=path2loaddir,pranges=None,pmin=pmin,pmax=pmax,sigma=sigma)
        Dist=(Parallax* u.mas).to(u.parsec, equivalencies=u.parallax()).value
        eDist_d=Dist-((Parallax+eParallax_u)*u.mas).to(u.parsec, equivalencies=u.parallax()).value
        eDist_u=((Parallax-eParallax_d)*u.mas).to(u.parsec, equivalencies=u.parallax()).value-Dist
        return([ID,logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,Dist,eDist_u,eDist_d,area_r])
    else:
        return([ID,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        
################################
# Star and accretion proprties #
################################
def star_properties(flat_samples,ndim,interp,pmin=1.66,pmax=3.30,mass_label='mass',T_label='teff',logL_label='logL',logLacc_label='logLacc',logMacc_label='logMacc',label_list=['mass','Av','Age','logSPacc','Parallax'],bw_method=None,kernel='linear',bandwidth2fit=np.linspace(0.01, 1, 100),kde_fit=False,path2savedir=None,return_fig=False):
    val_list=[]
    q_d_list=[]
    q_u_list=[]
    kde_list=[]
    for i in range(ndim):
        # x=np.sort(flat_samples[:,i][~flat_samples[:,i].mask])
        x=np.sort(flat_samples[:,i])
        mcmc = np.percentile(x, [16, 50, 84])
        if kde_fit:
            xlinspace = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
            kde=KDE(np.sort(x), xlinspace, bandwidth=bw_method,kernel=kernel,bandwidth2fit=bandwidth2fit)
            kde.kde_sklearn()
            kde_list.append(kde)
    
            pdf_max=np.max(kde.pdf(xlinspace))
            w=np.where(kde.pdf(xlinspace)==pdf_max)
            val=np.nanmedian(xlinspace[w])
            
            if label_list[i]=='Parallax':
                xlinspace2=xlinspace[(xlinspace>pmin)&(xlinspace<pmax)]
                try: 
                    area2 = trapz(kde.pdf(xlinspace2), dx=0.01)
                    area = trapz(kde.pdf(xlinspace), dx=0.01)
                    area_r=area2/area
                except: 
                    area_r = 0
                
        else: val =mcmc[1]
        q = np.diff([mcmc[0],val,mcmc[-1]])
        if label_list[i]=='Parallax':
            if (val-q[0]<0) or (mcmc[0]>val):q[0]=np.nan
            if (val+q[1]<0) or (mcmc[-1]<val):q[1]=np.nan
            val_list.append(val)
            q_d_list.append(q[0])
            q_u_list.append(q[1])
        else:    
            if (10**(val-q[0])<0) or (10**(mcmc[0])>10**val):q[0]=np.nan
            if (10**(val+q[1])<0) or (10**(mcmc[-1])<10**val):q[1]=np.nan
            val_list.append(val)
            q_d_list.append(q[0])
            q_u_list.append(q[1])
    
    logMass,logAv,logAge,logSPacc,Parallax=val_list
    # logMass=-0.77-0.98
    elogMass_d,elogAv_d,elogAge_d,elogSPacc_d,eParallax_d=q_d_list
    elogMass_u,elogAv_u,elogAge_u,elogSPacc_u,eParallax_u=q_u_list
    
    mass_u=10**(logMass+elogMass_u)
    mass_d=10**(logMass-elogMass_d)

    Age_u=10**(logAge+elogAge_u)
    Age_d=10**(logAge-elogAge_d)

    SPacc_u=10**(logSPacc+elogSPacc_u)
    SPacc_d=10**(logSPacc-elogSPacc_d)
    
    T=round(float(interp[T_label](logMass,logAge,logSPacc)),4)
    eT_u=round(float(interp[T_label](np.log10(mass_u),np.log10(Age_u),np.log10(SPacc_u)))-T,4)
    eT_d=round(T-float(interp[T_label](np.log10(mass_d),np.log10(Age_d),np.log10(SPacc_d))),4)
    if np.isnan(eT_d):eT_d=eT_u
    elif np.isnan(eT_u):eT_u=eT_d

    L=10**float(interp[logL_label](logMass,logAge,logSPacc))
    L_u=10**float(interp[logL_label](np.log10(mass_u),np.log10(Age_u),np.log10(SPacc_u)))
    L_d=10**float(interp[logL_label](np.log10(mass_d),np.log10(Age_d),np.log10(SPacc_d)))
    elogL_u=round(np.log10(L+L_u)-np.log10(L),4)
    elogL_d=round(np.log10(L)-np.log10(L-L_d),4)
    if np.isnan(elogL_d):elogL_d=elogL_u
    elif np.isnan(elogL_u):elogL_u=elogL_d
    logL=round(np.log10(L),4)

    Lacc=10**float(interp[logLacc_label](logMass,logAge,logSPacc))
    Lacc_u=10**float(interp[logLacc_label](np.log10(mass_u),np.log10(Age_u),np.log10(SPacc_u)))
    Lacc_d=10**float(interp[logLacc_label](np.log10(mass_d),np.log10(Age_d),np.log10(SPacc_d)))
    elogLacc_u=round(np.log10(Lacc+Lacc_u)-np.log10(Lacc),4)
    elogLacc_d=round(np.log10(Lacc)-np.log10(Lacc-Lacc_d),4)
    if np.isnan(elogLacc_d):elogLacc_d=elogLacc_u
    elif np.isnan(elogLacc_u):elogLacc_u=elogLacc_d
    logLacc=round(np.log10(Lacc),4)    
    
    Macc=10**float(interp[logMacc_label](logMass,logAge,logSPacc))
    Macc_u=10**float(interp[logMacc_label](np.log10(mass_u),np.log10(Age_u),np.log10(SPacc_u)))
    Macc_d=10**float(interp[logMacc_label](np.log10(mass_d),np.log10(Age_d),np.log10(SPacc_d)))
    elogMacc_u=round(np.log10(Macc+Macc_u)-np.log10(Macc),4)
    elogMacc_d=round(np.log10(Macc)-np.log10(Macc-Macc_d),4)
    if np.isnan(elogMacc_d):elogMacc_d=elogMacc_u
    elif np.isnan(elogMacc_u):elogMacc_u=elogMacc_d
    logMacc=round(np.log10(Macc),4)    

    return(logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,kde_list,area_r)

def lum_corr(MCMC_sim_df,ID,interp_mags,interp_cols,Av_list,DM,L,mag_label_list,color_label_list,verbose=False,truths=[None,None,None]):
    if verbose: display(MCMC_sim_df.loc[MCMC_sim_df.ID==ID])
    mag_good_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'good_mags'].values[0]
    col_good_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'good_cols'].values[0]
    mag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mags'].values[0]
    col_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'cols'].values[0]
    emag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'emags'].values[0]
    if np.all(np.array(truths) == None):
        mass=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mass'].values[0]
        Av=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'Av'].values[0]
        age=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'A'].values[0]
    else:
        mass,Av,age=truths
        
    dmag_corr_list=[]
    emag_corr_list=[]
    good_mag_label_list=[]
    q=np.where(mag_good_list)[0]
    w=np.where(col_good_list)[0]
    
    if verbose:
        print(ID,mag_good_list,col_good_list,Av_list)
        print('DM: %.3f, mass: %.3f, Av: %.3f, age: %.3f'%(DM,mass,Av,age))
        print('\nMatched Colors:')
        
    for elno in w:
        color_label=color_label_list[elno]
        mag1_label=color_label.split('-')[0]
        mag2_label=color_label.split('-')[1]
        good_mag_label_list.extend([mag1_label,mag2_label])
        j=np.where(mag1_label == mag_label_list)[0][0]
        k=np.where(mag2_label == mag_label_list)[0][0]
        Av_color=Av_list[j]-Av_list[k]
        iso_col=float(interp_cols[elno](np.log10(mass),np.log10(age)))
        col=col_list[elno]-Av*Av_color
        dcol=float((col-iso_col))
        if verbose:
            print('%s :'%color_label)
            print('orig col: %.3f, Av_color: %.3f, Av*Av_color: %.3f '%(col_list[elno],Av_color,Av*Av_color))
            print('iso: %.3f, derived col: %.3f, dcol: %.3f'%(iso_col,col,dcol))
            print(' ')

    if verbose:print('Derived Magnitude')
    for elno in q:
        mag_label=mag_label_list[elno]
        if mag_label in good_mag_label_list:
            iso_mag=float(interp_mags[elno](np.log10(mass),np.log10(age)))
            mag=mag_list[elno]-DM-Av*Av_list[elno]
            dmag=float((mag-iso_mag))
            dmag_corr_list.append(dmag)
            emag=emag_list[elno]
            emag_corr_list.append(emag**(-2))
            if verbose:
                print('%s :'%mag_label)
                print('orig mag: %.3f, Av_mag: %.3f, Av*Av_mag: %.3f '%(mag_list[elno],Av_list[elno],Av*Av_list[elno]))
                print('iso: %.3f, derived mag: %.3f, dmag: %.3f'%(iso_mag,mag,dmag))
                print(' ')

    dmag_corr_list=np.array(dmag_corr_list)
    emag_corr_list=np.array(emag_corr_list)
    # dmag_mean=np.average(dmag_corr_list,weights=emag_corr_list)
    dmag_median=np.median(dmag_corr_list)
    df_f=-dmag_median/1.087 
    if verbose:
        # print('Delta distance?!:')
        # print(dmag_mean,10**(abs(dmag_mean)/5),10**(abs(dmag_corr_list)/5))
        print('Delta Mag/Flux:')
        print('dmag_median: %.3f, dflux: %.3f'%(dmag_median,df_f))
        print('Delta L:')
        print('L: %.3f, 1+df: %.3f, Lf: %.3f'%(L,1+df_f,L*(1+df_f)))
        print('###################')
    return(L*(1+df_f))

def accr_stats(MCMC_sim_df,ID,m658_c,m658_d,e658,e658_c,zpt658,photlam658,Msun,Lsun,eLsun,Rsun,d,ed,sigma,RW,label='',s685=3,EQ_th=10):
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

    if (m658_c-m658_d>= s685*e658) and (EQW>=EQ_th):

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

def star_accrention_properties(self,MCMC_sim_df,avg_df,interp_mags,interp_cols,interp_658,DM,Av_list,Av_658,zpt658,photlam658,Msun,Lsun,eLsun,Rsun,d,ed,sigma,RW,ID_label='avg_ids',showplot=False,verbose=False,ID_list=[],p='',s685=3,EQ_th=10): 
    MCMC_sim_df[['emass']]=MCMC_sim_df[['emass_d','emass_u']].mean(axis=1)
    MCMC_sim_df[['eAv']]=MCMC_sim_df[['eAv_d','eAv_u']].mean(axis=1)
    MCMC_sim_df[['eA']]=MCMC_sim_df[['eA_d','eA_u']].mean(axis=1)
    MCMC_sim_df[['eT']]=MCMC_sim_df[['eT_d','eT_u']].mean(axis=1)
    MCMC_sim_df[['eL']]=MCMC_sim_df[['eL_d','eL_u']].mean(axis=1)
    MCMC_sim_df[['eL_corr']]=MCMC_sim_df[['eL_d','eL_u']].mean(axis=1)

    if len(ID_list) == 0: ID_list=MCMC_sim_df['ID'].unique()
    for ID in tqdm(ID_list): 
        color_good_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'good_cols'].values[0]
        if not np.all(np.isnan(color_good_list)) and np.any(color_good_list):
            L=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'L'].values[0]
            MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'L_corr']=lum_corr(MCMC_sim_df,ID,interp_mags,interp_cols,Av_list,DM,L,self.mag_label_list,self.color_label_list,verbose=verbose)
            mag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'mags'].tolist()[0][:-2]
            emag_list=MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'emags'].tolist()[0][:-2]
            m435,m555,m775,m850=mag_list
            e435,e555,e775,e850=emag_list

            m658=avg_df.loc[avg_df[ID_label]==ID,'m658%s'%p].values[0]
            e658=avg_df.loc[avg_df[ID_label]==ID,'e658%s'%p].values[0]

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
                    # MCMC_sim_df=accr_stats(MCMC_sim_df,ID,m658_c,m658_d,e658,e658_c,label='')
                    MCMC_sim_df=accr_stats(MCMC_sim_df,ID,m658_c,m658_d,e658,e658_c,zpt658,photlam658,Msun,Lsun,eLsun,Rsun,d,ed,sigma,RW,s685=s685,EQ_th=EQ_th)
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


#############
# Ancillary #
#############

def plot_ND(args, plotND=True, xerror=None, yerror=None, zerror=None, x_label='x', y_label='y', z_label='z',
            name_label='var', elno=0, fig=None, color='k', pad=1, w_pad=1, h_pad=1, fx=1000, fy=750, size=3, width=1,
            row=1, col=1, showplot=False, subplot_titles=['Plot1'], marker_color='black', aspectmode='cube'):
    if showplot:
        if plotND:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"type": "scatter3d"}]],
                subplot_titles=subplot_titles)
        else:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=subplot_titles)

    error_x = dict(type='data', array=xerror, visible=True)
    error_y = dict(type='data', array=yerror, visible=True)
    error_z = dict(type='data', array=zerror, visible=True)
    if plotND:
        fig.add_trace(go.Scatter3d(args, error_x=error_x, error_y=error_y, error_z=error_z,
                                   mode='markers',
                                   marker=dict(size=size, line=dict(width=width),
                                               color=marker_color),
                                   name=name_label),
                      row=row, col=col)
        fig.update_layout(autosize=False, width=fx, height=fy, margin=dict(l=10, r=10, b=10, t=22, pad=4),
                          paper_bgcolor="LightSteelBlue")
        fig.update_scenes(xaxis=dict(title_text=x_label), yaxis=dict(title_text=y_label),
                          zaxis=dict(title_text=z_label), row=row, col=col, aspectmode=aspectmode)
    else:
        fig.add_trace(go.Scatter(args, error_x=error_x, error_y=error_y,
                                 mode='markers',
                                 marker=dict(size=size * 3, line=dict(width=width),
                                             color=marker_color),
                                 name=name_label),
                      row=row, col=col)
        fig.update_layout(autosize=False, width=fx, height=fy, margin=dict(l=10, r=10, b=10, t=22, pad=4),
                          paper_bgcolor="LightSteelBlue")
        # fig.update_scenes(xaxis=dict(title_text=x_label),yaxis=dict(title_text=y_label),row=row,col=col)
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    if showplot:
        fig.show()
    else:
        return (fig)


def interpND_task(elno, args, method, smooth):
    args_reshaped = [args[0][i] for i in range(len(args[0][:-1]))]
    if method == 'nearest':
        node_list = args[0][-1]
        args2interp = [list(zip(*args_reshaped)), node_list[elno]]
        interpolation = LinearNDInterpolator(*args2interp)
    else:
        args_reshaped.append(args[0][-1][elno])
        interpolation = Rbf(*args_reshaped, smooth=smooth, function=method)
    return (interpolation, args_reshaped, elno)


def interpND_plots(elno, args, args_reshaped, node_list, interpolations_list, interpolation, Dict, z_label, x_label,
                   y_label, new_coords, color_labels, row=1, col=1, fig=None, fx=7, fy=7, ncols=1, showplot=False):
    interpolations_list.append(interpolation)
    Dict[z_label[elno]] = interpolation
    if showplot:
        if len(*args) - 1 == 3:
            thisdict = {
                "x": new_coords[0],
                "y": new_coords[1],
                "z": new_coords[2]}
            # "z": interpolation(*new_coords)}
            thisdict2 = {
                "x": args_reshaped[0].ravel(),
                "y": args_reshaped[1].ravel(),
                "z": args_reshaped[2].ravel()}
            # "z": node_list[elno].ravel()}
            # marker_color1=new_coords[2]
            # marker_color2=args_in[2].ravel()
            marker_color1 = interpolation(*new_coords)
            marker_color2 = node_list[elno].ravel()
            z_label_ND = z_label[elno]
            label_ND2 = z_label[elno] + '_o'
            label_ND1 = z_label[elno] + '_i'
            plotND = True
        elif len(*args) - 1 == 2:
            thisdict = {
                "x": new_coords[0],
                "y": new_coords[1],
                "z": interpolation(*new_coords)}
            thisdict2 = {
                "x": args_reshaped[0].ravel(),
                "y": args_reshaped[1].ravel(),
                "z": node_list[elno].ravel()}

            marker_color1 = 'lightskyblue'
            marker_color2 = 'black'
            z_label_ND = z_label[elno]
            label_ND2 = z_label[elno] + '_o'
            label_ND1 = z_label[elno] + '_i'
            plotND = True
        elif len(*args) - 1 == 1:
            thisdict = {
                "x": new_coords[0],
                "y": interpolation(*new_coords)}
            thisdict2 = {
                "x": args_reshaped[0].ravel(),
                "y": node_list[elno].ravel()}

            marker_color1 = 'lightskyblue'
            marker_color2 = 'black'
            z_label_ND = []
            y_label = z_label[elno]
            label_ND2 = z_label[elno] + '_o'
            label_ND1 = z_label[elno] + '_i'
            plotND = False

        fig = plot_ND(thisdict2, plotND=plotND, row=row, col=col, showplot=False, marker_color=marker_color2, fig=fig,
                      name_label=label_ND2)
        fig = plot_ND(thisdict, plotND=plotND, showplot=False, row=row, col=col, marker_color=marker_color1, fx=fx,
                      fy=fy, fig=fig, x_label=x_label, y_label=y_label, z_label=z_label_ND, name_label=label_ND1)
        col += 1
        if col > ncols:
            row += 1
            col = 1
    return (fig, Dict)


def interpND(*args, smooth=0, method='nearest', x_label='x', y_label='y', z_label=None, color_labels=None,
             showplot=False, radial=True, fx=1450, fy=1000, w_pad=3, h_pad=1, pad=3, npoints=50, nrows=1, surface=True,
             workers=None, progress=True, horizontal_spacing=0.01, vertical_spacing=0.01):
    '''Calculate 2d interpolation for the z axis along x and y variables.
        Parameters:
            args: x, y, z, where x, y, z, ... are the coordinates of the nodes
            node_list: list of lists of values at the nodes to interpolate. The rotuine will perfom a different interpolation (using the same x and y) for each sublist of node_list.
            x_range,y_range: range of x,y variables in the from [in,end] over which evaluate the interpolation. If not provide will automaticaly take min,max as ranges.
            smooth: smoothnes parametr for Rbf routine
            method:
                if 'nearest' selected, then use LinearNDInterpolator to interpolate, otherwise use the Rbf routine where method parameters are:
                'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                'gaussian': exp(-(r/self.epsilon)**2)
                'linear': r
                'cubic': r**3
                'quintic': r**5
                'thin_plate': r**2 * log(r)

            x_labe,y_label: labels  for x,y axis
            z_label: list of labels for the z axis
            showplot: if true show plot for the interpolation
            radial: use Rbf insead of interp2d
        Returns:
            interpolations_list: list of interpolatations functions to be called as new_z=interpolations_list(x0,y0)
    '''
    if workers == None:
        ncpu = multiprocessing.cpu_count()
        if ncpu >= 3:
            workers = ncpu - 2
        else:
            workers = 1
        print('> Selected Workers:', workers)

    if np.all(z_label == None): z_label = ['z'] * len(args[0][-1])
    if surface:
        meshed_coords = np.meshgrid(
            *[np.linspace(np.min(args[0][i]), np.max(args[0][i]), npoints) for i in range(len(args[0][:-1]))])
        new_coords = [meshed_coords[i].ravel() for i in range(len(meshed_coords))]
    else:
        new_coords = np.array(
            [np.linspace(np.min(args[0][i]), np.max(args[0][i]), npoints) for i in range(len(args[0][:-1]))])
    node_list = args[0][-1]
    if showplot:

        ncols = int(round_up(len(node_list) / nrows))
        if len(*args) - 1 != 1:
            fig = make_subplots(rows=nrows, cols=ncols,
                                specs=[[{"type": "surface"} for i in range(ncols)] for j in range(nrows)],
                                horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)
        else:
            fig = make_subplots(rows=nrows, cols=ncols,
                                horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)
    else:
        fig = None
        ncols = 1
    interpolations_list = []
    elno_list = [i for i in range(len(node_list))]
    row = 1
    col = 1
    Dict = {}

    if progress:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for interpolation, args_reshaped, elno in tqdm(
                    executor.map(interpND_task, elno_list, repeat(args), repeat(method), repeat(smooth))):
                fig, Dict = interpND_plots(elno, args, args_reshaped, node_list, interpolations_list, interpolation,
                                           Dict, z_label, x_label, y_label, new_coords, color_labels, row=row, col=col,
                                           fig=fig, fx=fx, fy=fy, ncols=ncols, showplot=showplot)
    else:
        if workers == 0:
            for elno in elno_list:
                interpolation, args_reshaped, elno = interpND_task(elno, args, method, smooth)
                fig, Dict = interpND_plots(elno, args, args_reshaped, node_list, interpolations_list, interpolation,
                                           Dict, z_label, x_label, y_label, new_coords, color_labels, row=row,
                                           col=elno + 1, fig=fig, fx=fx, fy=fy, ncols=ncols, showplot=showplot)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for interpolation, args_reshaped, elno in executor.map(interpND_task, elno_list, repeat(args),
                                                                       repeat(method), repeat(smooth)):
                    fig, Dict = interpND_plots(elno, args, args_reshaped, node_list, interpolations_list, interpolation,
                                               Dict, z_label, x_label, y_label, new_coords, color_labels, row=row,
                                               col=col, fig=fig, fx=fx, fy=fy, ncols=ncols, showplot=showplot)

    if showplot:
        fig.show()
    return (Dict)