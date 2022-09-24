#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:36:26 2021

@author: giovanni
"""
import sys
sys.path.append('./')
sys.path.append('/mnt/Storage/Lavoro/GitHub/imf-master/imf/')
# from config import path2data
# from miscellaneus import chunks
from kde import KDE
import numpy as np

from astropy import units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import bz2
# import pickle
from itertools import repeat
# import multiprocessing as mp
import _pickle as cPickle
import concurrent.futures
from IPython.display import display
from numpy import trapz
from astropy.stats import sigma_clip
# from mcmc_plots import *
from mcmc_plots import sample_posteriors

########################
# Simulated photometry #
########################


def simulate_mag_star(ID,sat_list,variables_interp_in,mag_variable_in,Av1_list,ID_label='avg_ids',mag_list=[],emag_list=[],var=None,Av1=None,age=None,logSPacc=None,parallax=None,err=None,err_min=0.01,err_max=0.1,avg_df=None):#,sim_label='Teff'):
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
            if (np.isnan(mag_list[elno]))|(np.isnan(emag_list[elno]))|(emag_list[elno] >=err_max)|(f_spx==3): mag_good_list[elno]=False
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



# def get_Av_list(interp_mags,interp_Tlogg,filter_label_list,mag_label_list,photflam,Rv,date='2005-01-1',DM=0,mass=1,age=1,Av=1,showplot=False,photlam658=1.977e-18):
#     obsdate = Time(date).mjd

#     bp435=stsyn.band(f'acs,wfc1,f435w,mjd#{obsdate}')
#     bp555=stsyn.band(f'acs,wfc1,f555w,mjd#{obsdate}')
#     bp658=stsyn.band(f'acs,wfc1,f658n,mjd#{obsdate}')
#     bp775=stsyn.band(f'acs,wfc1,f775w,mjd#{obsdate}')
#     bp850=stsyn.band(f'acs,wfc1,f850lp,mjd#{obsdate}')

#     bp130=stsyn.band('wfc3,ir,f130n')
#     bp139=stsyn.band('wfc3,ir,f139m')

#     bp_list=[bp435,bp555,bp775,bp850,bp130,bp139]

#     mag_sel_list=[interp_mags[i](np.log10(mass),np.log10(age))+DM for i in range(len(mag_label_list))]
#     T=interp_Tlogg[0](np.log10(mass),np.log10(age))
#     logg=interp_Tlogg[1](np.log10(mass),np.log10(age))
#     wavelengths_list=[]
#     sp = SourceSpectrum(BlackBodyNorm1D, temperature=T)
#     binset = range(1000, 30001)

#     for bp in bp_list:
#         wavelengths_list.append(Observation(sp, bp, binset=binset).effective_wavelength())

#     wavelengths_658=Observation(sp, bp658, binset=binset).effective_wavelength()

#     sp = stsyn.grid_to_spec('phoenix', T, 0, logg)

#     band =stsyn.band('acs,wfc1,f555w') # SpectralElement.from_filter('johnson_v')#555
#     vega = SourceSpectrum.from_vega()
#     mag = np.array(mag_sel_list)[np.array(filter_label_list)=='F555W'][0]* units.VEGAMAG
#     sp_norm = sp.normalize(mag , band, vegaspec=vega)

#     wav = binset * u.AA
#     flux = sp_norm(wav).to(FLAM, u.spectral_density(wav))#*1e3/dist_ist[-1]

#     if showplot:
#         plt.figure(figsize=(7,7))
#         plt.loglog(wav.value, flux.value)
#         plt.title('Blackbody T=%.2f'%T)
#         plt.ylabel('FLAM')
#         plt.xlabel('$\lambda$ [AA]')
#         plt.show()
        
        
#     ext = CCM89(Rv=Rv)
    
#     # Make the extinction model in synphot using a lookup table.
#     ex = ExtinctionCurve(ExtinctionModel1D,
#                          points=wav, lookup_table=ext.extinguish(wav, Av=Av))
#     sp_ext = sp_norm*ex

#     flux_ext = sp_ext(wav).to(FLAM, u.spectral_density(wav))

#     if showplot:
#         plt.figure(figsize=(7,7))
#         plt.loglog(wav, flux_ext)
#         plt.title('Blackbody T=%.2f Av=%.3f'%(T,Av))
#         plt.ylabel('FLAM')
#         plt.xlabel('$\lambda$ [AA]')
#         plt.show()
        
#     Av_list=[]

#     for n in range(len(filter_label_list)):
#         qq=np.where(np.array(wav.value)==round(wavelengths_list[n].value))[0]
#         Av1_mag=-2.5*np.log10((sp_ext(wav).to(FLAM, u.spectral_density(wav))[qq]/photflam[n]).value)[0]
#         Av0_mag=-2.5*np.log10((sp_norm(wav).to(FLAM, u.spectral_density(wav))[qq]/photflam[n]).value)[0]
#         Av_list.append(round(Av1_mag-Av0_mag,5))
    
#     qq=np.where(np.array(wav.value)==round(wavelengths_658.value))[0]
#     Av1_mag=-2.5*np.log10((sp_ext(wav).to(FLAM, u.spectral_density(wav))[qq]/(photlam658)).value)[0]
#     Av0_mag=-2.5*np.log10((sp_norm(wav).to(FLAM, u.spectral_density(wav))[qq]/(photlam658)).value)[0]
#     Av_658=round(Av1_mag-Av0_mag,5)
        
#     return(np.array(Av_list),Av_658)

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

def update_dataframe(df,file_list,interp,workers=10,chunksize = 30,ID_label='avg_ids',kde_fit=False,discard=0,thin=1,label_list=['logMass','logAv','logAge','logSPacc','Parallax'],pmin=1.66,pmax=3.30,path2savedir=None,return_fig=False):
    ntarget=len(file_list)
    if kde_fit:
        for file in tqdm(file_list):
            ID,logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,Dist,eDist_u,eDist_d,area_r=task(file,interp,kde_fit,discard,thin,label_list,pmin=pmin,pmax=pmax,path2savedir=path2savedir,return_fig=return_fig)
            df.loc[df[ID_label]==ID,['MCMC_mass','MCMC_emass_u','MCMC_emass_d','MCMC_Av','MCMC_eAv_u','MCMC_eAv_d','MCMC_A','MCMC_eA_u','MCMC_eA_d','MCMC_T','MCMC_eT_u','MCMC_eT_d','MCMC_logL','MCMC_elogL_u','MCMC_elogL_d','MCMC_logSPacc','MCMC_elogSPacc_u','MCMC_elogSPacc_d','MCMC_logLacc','MCMC_elogLacc_u','MCMC_elogLacc_d','MCMC_logMacc','MCMC_elogMacc_u','MCMC_elogMacc_d','MCMC_Parallax','MCMC_eParallax_d','MCMC_eParallax_u','MCMC_d','MCMC_ed_u','MCMC_ed_d','MCMC_area_r']]=[[10**logMass, 10**(logMass+elogMass_u)-10**logMass, 10**logMass-10**(logMass-elogMass_d),10**logAv, 10**(logAv+elogAv_u)-10**logAv, 10**logAv-10**(logAv-elogAv_d),10**logAge, 10**(logAge+elogAge_u)-10**logAge, 10**logAge-10**(logAge-elogAge_d),T,eT_u,eT_d,logL,elogL_u,elogL_d,logSPacc,elogSPacc_u,elogSPacc_d,logLacc,elogLacc_u,elogLacc_d,logMacc,elogMacc_u,elogMacc_d,Parallax, eParallax_u, eParallax_d,Dist,eDist_u,eDist_d,area_r]]
    else:
        print('> workers %i,chunksize %i,ntarget %i'%(workers,chunksize,ntarget))    
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for ID,logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,Dist,eDist_u,eDist_d,area_r in tqdm(executor.map(task,file_list,repeat(interp),repeat(kde_fit),repeat(discard),repeat(thin),repeat(label_list),repeat(pmin),repeat(pmax),repeat(path2savedir),repeat(return_fig),chunksize=chunksize)):
                df.loc[df[ID_label]==ID,['MCMC_mass','MCMC_emass_u','MCMC_emass_d','MCMC_Av','MCMC_eAv_u','MCMC_eAv_d','MCMC_A','MCMC_eA_u','MCMC_eA_d','MCMC_T','MCMC_eT_u','MCMC_eT_d','MCMC_logL','MCMC_elogL_u','MCMC_elogL_d','MCMC_logSPacc','MCMC_elogSPacc_u','MCMC_elogSPacc_d','MCMC_logLacc','MCMC_elogLacc_u','MCMC_elogLacc_d','MCMC_logMacc','MCMC_elogMacc_u','MCMC_elogMacc_d','MCMC_Parallax','MCMC_eParallax_d','MCMC_eParallax_u','MCMC_d','MCMC_ed_u','MCMC_ed_d','MCMC_area_r']]=[[10**logMass, 10**(logMass+elogMass_u)-10**logMass, 10**logMass-10**(logMass-elogMass_d),10**logAv, 10**(logAv+elogAv_u)-10**logAv, 10**logAv-10**(logAv-elogAv_d),10**logAge, 10**(logAge+elogAge_u)-10**logAge, 10**logAge-10**(logAge-elogAge_d),T,eT_u,eT_d,logL,elogL_u,elogL_d,logSPacc,elogSPacc_u,elogSPacc_d,logLacc,elogLacc_u,elogLacc_d,logMacc,elogMacc_u,elogMacc_d,Parallax, eParallax_u, eParallax_d,Dist,eDist_u,eDist_d,area_r]]
    return(df)

def task(file,interp,kde_fit=False,discard=0,thin=1,label_list=['logMass','logAv','logAge','logSPacc','Parallax'],pmin=1.66,pmax=3.30,path2savedir=None,return_fig=False):
    ndim=len(label_list)
    ID=float(file.split('_')[-1])
    mcmc_dict=read_samples(file)  
    samples=np.array(mcmc_dict['samples'])
    if len(samples)>0:
        # if discard!=None: samples=samples[discard:, :, :]
        # if thin!=None: samples=samples[::thin, :, :]        
        # flat_samples=samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
        # filtered_flat_sample=sigma_clip(flat_samples, sigma=3.5, maxiters=5,axis=0)
        # flat_samples=filtered_flat_sample.copy()
        logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,kde_list,area_r=sample_posteriors(interp,float(ID),ndim,verbose=False,fx=10,fy=10,show_samples=False,showplots=False,bins=10,kde_fit=kde_fit,return_fig=False,return_variables=True,path2savedir=path2savedir,pranges=None)
        # a=sample_posteriors(interp,float(ID),ndim,verbose=False,fx=10,fy=10,show_samples=False,bins=10,kde_fit=kde_fit,return_fig=return_fig,return_variables=True,path2savedir=path2savedir,pranges=None)
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
    # flat_samples=samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
    val_list=[]
    q_d_list=[]
    q_u_list=[]
    kde_list=[]
    for i in range(ndim):
        x=np.sort(flat_samples[:,i][~flat_samples[:,i].mask])
        mcmc = np.percentile(x, [16, 50, 84])
        area_r = 0
        if kde_fit:
            xlinspace=np.linspace(min(x),max(x),1000)
            # xlinspace=np.linspace(min(flat_samples[:,i]),max(flat_samples[:,i]),1000)
            kde=KDE(np.sort(x), xlinspace, bandwidth=bw_method,kernel=kernel,bandwidth2fit=bandwidth2fit)
            kde.kde_sklearn()
            kde_list.append(kde)
    
            pdf_max=np.max(kde.pdf(xlinspace))
            w=np.where(kde.pdf(xlinspace)==pdf_max)
            val=np.nanmedian(xlinspace[w])
            
            if i == ndim-1:
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

# def round_up(n, decimals=0):
#     multiplier = 10 ** decimals
#     return math.ceil(n * multiplier) / multiplier

# def get_kde_pdf(x_sort,bw_method,kernel='linear',auto_adjust=True):
#     xlinspace=np.linspace(min(x_sort),max(x_sort),1000)
    
#     kde=KDE(x_sort, xlinspace, bandwidth=bw_method,kernel=kernel)
    
#     # my_pdf = gaussian_kde(x_sort,bw_method=bw_method)
#     # my_pdf=kde_sklearn(x_sort, xlinspace, bandwidth=bw_method,kernel=kernel)
#     # if auto_adjust:
#     #     grid = GridSearchCV(KernelDensity(),
#     #                 {'bandwidth': np.linspace(0.01, 10, 1000)},
#     #                 cv=20,n_jobs=10) # 20-fold cross-validation
#     #     grid.fit(x_sort[:, None])
#     return(kde)
        # pdf_max=np.max(my_pdf(xlinspace))
        # w=np.where(my_pdf(xlinspace)==pdf_max)
        # val=np.round(np.nanmedian(xlinspace[w]),2)
        # spline = UnivariateSpline(xlinspace, my_pdf(xlinspace)-np.nanmax(my_pdf(xlinspace))/2, s=0)
        # r1= spline.roots()[0] 
        # r2= spline.roots()[-1]  # find the roots
        # if r2<val: r2=np.nanmax(xlinspace)
        # if r1>val: r1=np.nanmin(xlinspace)
        # bw_method2=round((abs(r2-r1)/abs(max(xlinspace)-min(xlinspace)))/2,2)
        # if bw_method2<=0.1: bw_method2=0.1
        # my_new_pdf = gaussian_kde(x_sort,bw_method=bw_method2)
        # return(my_new_pdf,r1,r2)
    # else: return(my_pdf,np.nan,np.nan)
