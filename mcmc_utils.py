#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:36:26 2021

@author: giovanni
"""
import sys,math
sys.path.append('./')
sys.path.append('/mnt/Storage/Lavoro/GitHub/imf-master/imf/')

import numpy as np
import stsynphot as stsyn
from synphot import units,ExtinctionModel1D,Observation,SourceSpectrum
from synphot.units import FLAM
from dust_extinction.parameter_averages import CCM89
from synphot.reddening import ExtinctionCurve
from synphot.models import BlackBodyNorm1D
from astropy.time import Time
from astropy import units as u
import matplotlib.pyplot as plt
# from tqdm import tqdm
# from IPython.display import display

########################
# Simulated photometry #
########################

def simulate_mag_star(sat_list,variables_interp_in,mag_variable_in,Av1_list,mag_list=[],emag_list=[],mass=None,Av1=None,age=None,distance=None,err=None,err_min=0.01,err_max=0.1,mvs_df=None,avg_df=None):
    if len(mag_list)==0 and len(emag_list)==0: 
        mag_temp_list=np.round(np.array([float(variables_interp_in[i](np.log10(mass),np.log10(age)))+Av1_list[i]*Av1+5*np.log10(distance/10) for i in range(len(variables_interp_in))]),3)
        emag_temp_list=[]
        if err!=None:
            emag_list=np.array([err]*len(mag_temp_list))
        else:
            emag_list=[]
            for i in range(len(mag_variable_in)):
                mag_label='m%s'%(mag_variable_in[i][1:4])
                emag_label='e%s'%(mag_variable_in[i][1:4])
                if mag_label in mvs_df.columns: df=mvs_df
                elif mag_label in avg_df.columns: df=avg_df
                else: raise ValueError('%s found in no dataframe'%mag_label)
                err=np.nanmedian(df.loc[(df[mag_label]>=mag_temp_list[i]-0.5)&(df[mag_label]<=mag_temp_list[i]+0.5),emag_label].values)
                emag_list.append(np.round(err,3))
        emag_list=np.array(emag_list)   
        mag_list=np.array([round(mag_temp_list[i]+np.random.normal(0,scale=emag_list[i]),3) for i in range(len(variables_interp_in))])
    else:mag_temp_list=np.copy(mag_list)

    emag_temp_list=np.copy(emag_list)

    mag_good_list=(emag_list<=err_max)&(mag_list>=sat_list)
    mag_list[~mag_good_list]=np.nan
    emag_list[~mag_good_list]=np.nan
    return(np.round(mag_list,4),np.round(emag_list,4),np.round(mag_temp_list,4),np.round(emag_temp_list,4),mag_good_list)

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
        ecolor_list.append(np.round(np.sqrt(emag_list[j]**2+emag_list[k]**2),3))
        Av1_color_list.append(Av1_list[j]-Av1_list[k])
        if np.isnan(color_list[-1]): color_good_list.append(False)
        else: color_good_list.append(True)
    return(np.round(np.array(color_list),4),np.round(np.array(ecolor_list),4),np.round(np.array(Av1_color_list),4),color_good_list)

def get_Av_list(interp_mags,interp_Tlogg,filter_label_list,mag_label_list,photflam,Rv,DM=8.02,mass=1,age=1,showplot=False,photlam658=1.977e-18):
    obsdate = Time('2005-01-1').mjd

    bp435=stsyn.band(f'acs,wfc1,f435w,mjd#{obsdate}')
    bp555=stsyn.band(f'acs,wfc1,f555w,mjd#{obsdate}')
    bp658=stsyn.band(f'acs,wfc1,f658n,mjd#{obsdate}')
    bp775=stsyn.band(f'acs,wfc1,f775w,mjd#{obsdate}')
    bp850=stsyn.band(f'acs,wfc1,f850lp,mjd#{obsdate}')

    bp130=stsyn.band('wfc3,ir,f130n')
    bp139=stsyn.band('wfc3,ir,f139m')

    bp_list=[bp435,bp555,bp775,bp850,bp130,bp139]

    mag_sel_list=[interp_mags[i](np.log10(mass),np.log10(age))+DM for i in range(len(mag_label_list))]
    T=interp_Tlogg[0](np.log10(mass),np.log10(age))
    logg=interp_Tlogg[1](np.log10(mass),np.log10(age))
    wavelengths_list=[]
    sp = SourceSpectrum(BlackBodyNorm1D, temperature=T)
    binset = range(1000, 30001)

    for bp in bp_list:
        wavelengths_list.append(Observation(sp, bp, binset=binset).effective_wavelength())

    wavelengths_658=Observation(sp, bp, binset=binset).effective_wavelength()

    sp = stsyn.grid_to_spec('phoenix', T, 0, logg)

    band =stsyn.band('acs,wfc1,f555w') # SpectralElement.from_filter('johnson_v')#555
    vega = SourceSpectrum.from_vega()
    mag = mag_sel_list[-1]* units.VEGAMAG
    sp_norm = sp.normalize(mag , band, vegaspec=vega)

    wav = binset * u.AA
    flux = sp_norm(wav).to(FLAM, u.spectral_density(wav))#*1e3/dist_ist[-1]

    if showplot:
        plt.figure(figsize=(7,7))
        plt.plot(wav.value, flux.value)
        plt.title('Blackbody T=%.2f'%T)
        plt.ylabel('FLAM')
        plt.xlabel('$\lambda$ [AA]')
        plt.show()
        
        
    ext = CCM89(Rv=Rv)
    # ext = CCM89(Rv=5)
    Av = 1

    # Make the extinction model in synphot using a lookup table.
    ex = ExtinctionCurve(ExtinctionModel1D,
                         points=wav, lookup_table=ext.extinguish(wav, Av=Av))
    sp_ext = sp_norm*ex

    flux_ext = sp_ext(wav).to(FLAM, u.spectral_density(wav))

    if showplot:
        plt.figure(figsize=(7,7))
        plt.loglog(wav, flux_ext)
        plt.title('Blackbody T=%.2f Av=%.3f'%(T,Av))
        plt.ylabel('FLAM')
        plt.xlabel('$\lambda$ [AA]')
        plt.show()
        
    Av_list=[]

    for n in range(len(filter_label_list)):
        qq=np.where(np.array(wav.value)==round(wavelengths_list[n].value))[0]
        Av1_mag=-2.5*np.log10((sp_ext(wav).to(FLAM, u.spectral_density(wav))[qq]/photflam[n]).value)[0]
        Av0_mag=-2.5*np.log10((sp_norm(wav).to(FLAM, u.spectral_density(wav))[qq]/photflam[n]).value)[0]
        Av_list.append(round(Av1_mag-Av0_mag,5))
    
    qq=np.where(np.array(wav.value)==round(wavelengths_658.value))[0]
    Av1_mag=-2.5*np.log10((sp_ext(wav).to(FLAM, u.spectral_density(wav))[qq]/(photlam658)).value)[0]
    Av0_mag=-2.5*np.log10((sp_norm(wav).to(FLAM, u.spectral_density(wav))[qq]/(photlam658)).value)[0]
    Av_658=round(Av1_mag-Av0_mag,5)
    
    return(np.array(Av_list),Av_658)

def truth_list(mass,Av,age,mass_lim=[0.1,0.9],Av_lim=[0,10],age_lim=[0,100]):
    if mass==None:mass=np.round(random.uniform(mass_lim[0],mass_lim[1]),4)
    if Av==None: Av=np.round(random.uniform(Av_lim[0],Av_lim[1]),4)
    if age==None:age=np.round(random.uniform(age_lim[0],age_lim[1]),4)
    return(mass,Av,age)
    


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

#############
# Ancillary #
#############

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

