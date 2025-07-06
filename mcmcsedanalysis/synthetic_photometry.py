#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 08:56:17 2021

@author: giovanni
"""
from synphot import Observation,SourceSpectrum,units,ExtinctionModel1D
from astropy import units as u
import numpy as np
import os,sys,math
# os.environ["PYSYN_CDBS"] = "/home/gstrampelli/anaconda3/lib/python3.8/site-packages/stsynphot/grp/hst/cdbs/"
from IPython.display import display
from astropy.table import Table
from tqdm import tqdm
from astropy.io import ascii
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import concurrent.futures
from itertools import repeat
import multiprocessing
from astropy.io import fits
from astropy.visualization import quantity_support
from specutils.manipulation import FluxConservingResampler#, LinearInterpolatedResampler, SplineInterpolatedResampler
from specutils import Spectrum1D
from plotly.subplots import make_subplots
from decimal import *
from dust_extinction.parameter_averages import CCM89
from synphot.reddening import ExtinctionCurve
import platform
from pathlib import Path

def load_spectra_task(file,acc):
    OS=platform.system()
    if OS == 'Windows': sep='\\'
    else: sep='/'

    if acc:

        temp=float(file.split(sep)[-1].split('_')[3].split('teff')[1])
        logg=float(file.split(sep)[-1].split('_')[4].split('logg')[1])
        logacc=float(file.split(sep)[-1].split('_')[6].split('logacc')[1].split('.dat')[0])

        spectrum=SourceSpectrum.from_file(file)#,wave_col='lam',flux_col='flux')
        return(spectrum,temp,logg,logacc)#,L,Lacc,Lf)
    else:
        temp=float(file.split(sep)[-1].split('_')[3].split('teff')[1])
        logg=float(file.split(sep)[-1].split('_')[4].split('logg')[1])

        spectrum=SourceSpectrum.from_file(file)#,wave_col='lam',flux_col='flux')
        return(spectrum,temp,logg)

def syntetic_photometry(spectrum,vega_spectrum,bp,wav,r,d=10,return_flux=False):
    '''default is TA-DA eq 2'''
    R=r*u.cm.to(u.pc)*u.pc
    D=d*u.pc
    obs=Observation(spectrum, bp, binset=wav)
    vobs=Observation(vega_spectrum, bp,binset=wav)
    vega_flux=vobs.effstim(wavelengths=wav,flux_unit='flam')
    star_flux=obs.effstim(wavelengths=wav,flux_unit='flam')
    mag=-2.5*np.log10((R/D)**2*(star_flux/vega_flux))
    if return_flux:
        wavelength=obs.effective_wavelength(wavelengths=wav)
        return(mag,star_flux,wavelength)
    else: return(mag)

def from_teff_to_str(teff):
    s='000'
    if teff%100==0:
        t_s=int(teff/100)
        temp = list(s)
        temp[-len(str(t_s)):]=str(t_s)
        s = "".join(temp)
    else:
        t_s=teff/100
        temp = list(s)
        temp[-len(str(t_s))+2:]=str(t_s)
        s = "".join(temp)
    return(s)

def interpolate_spectra2D(spectrum_with_acc_df,teff,logg):
    tmin=math.floor(teff/100)*100
    tmax=math.ceil(teff/100)*100

    if (logg)-math.floor(logg)>0.5:
        loggmin=math.floor(logg)+0.5
        loggmax=math.ceil(logg)
    elif (logg)-math.floor(logg)<0.5:
        loggmin=math.floor(logg)
        loggmax=math.ceil(logg)-0.5
    else:
        loggmin=logg
        loggmax=logg

    if tmax==tmin:
        tmin=tmax-0.001
    if loggmax==loggmin:
        loggmin=loggmax-0.001

    A1=np.array([0,0])
    B1=np.array([0,1])
    C1=np.array([1,0])
    D1=np.array([1,1])

    S=np.array([(teff-tmin)/(tmax-tmin),(logg-loggmin)/(loggmax-loggmin)])

    CA1=np.linalg.norm(S-A1)
    CB1=np.linalg.norm(S-B1)
    CC1=np.linalg.norm(S-C1)
    CD1=np.linalg.norm(S-D1)

    spA1=spectrum_with_acc_df.loc[np.round(tmin,1),np.round(loggmin,2)].Spectrum
    spB1=spectrum_with_acc_df.loc[np.round(tmin,1),np.round(loggmax,2)].Spectrum
    spC1=spectrum_with_acc_df.loc[np.round(tmax,1),np.round(loggmin,2)].Spectrum
    spD1=spectrum_with_acc_df.loc[np.round(tmax,1),np.round(loggmax,2)].Spectrum
    spS=(spA1/CA1+spB1/CB1+spC1/CC1+spD1/CD1)/np.sum([1/CA1,1/CB1,1/CC1,1/CD1])
    return(spS)

def interpolate_spectra3D(spectrum_with_acc_df,teff,logg,logSPacc):
    tmin=math.floor(teff/100)*100
    tmax=math.ceil(teff/100)*100
    if (logg)-math.floor(logg)>0.5:
        loggmin=math.floor(logg)+0.5
        loggmax=math.ceil(logg)
    elif (logg)-math.floor(logg)<0.5:
        loggmin=math.floor(logg)
        loggmax=math.ceil(logg)-0.5
    else:
        loggmin=logg
        loggmax=logg

    logSPaccmin=math.floor(logSPacc)
    logSPaccmax=math.ceil(logSPacc)
    if tmax==tmin:
        tmin=tmax-0.001
    if loggmax==loggmin:
        loggmin=loggmax-0.001
    if logSPaccmax==logSPaccmin:
        logSPaccmin=logSPaccmax-0.001

    A1=np.array([0,0,0])
    B1=np.array([0,1,0])
    C1=np.array([1,0,0])
    D1=np.array([1,1,0])

    A2=np.array([0,0,1])
    B2=np.array([0,1,1])
    C2=np.array([1,0,1])
    D2=np.array([1,1,1])

    S=np.array([(teff-tmin)/(tmax-tmin),(logg-loggmin)/(loggmax-loggmin),(logSPacc-logSPaccmin)/(logSPaccmax-logSPaccmin)])

    CA1=np.linalg.norm(S-A1)
    CB1=np.linalg.norm(S-B1)
    CC1=np.linalg.norm(S-C1)
    CD1=np.linalg.norm(S-D1)

    CA2=np.linalg.norm(S-A2)
    CB2=np.linalg.norm(S-B2)
    CC2=np.linalg.norm(S-C2)
    CD2=np.linalg.norm(S-D2)

    spA1=spectrum_with_acc_df.loc[np.round(tmin,1),np.round(loggmin,2),np.round(logSPaccmin,1)].Spectrum
    spB1=spectrum_with_acc_df.loc[np.round(tmin,1),np.round(loggmax,2),np.round(logSPaccmin,1)].Spectrum
    spC1=spectrum_with_acc_df.loc[np.round(tmax,1),np.round(loggmin,2),np.round(logSPaccmin,1)].Spectrum
    spD1=spectrum_with_acc_df.loc[np.round(tmax,1),np.round(loggmax,2),np.round(logSPaccmin,1)].Spectrum

    spA2=spectrum_with_acc_df.loc[np.round(tmin,1),np.round(loggmin,2),np.round(logSPaccmax,1)].Spectrum
    spB2=spectrum_with_acc_df.loc[np.round(tmin,1),np.round(loggmax,2),np.round(logSPaccmax,1)].Spectrum
    spC2=spectrum_with_acc_df.loc[np.round(tmax,1),np.round(loggmin,2),np.round(logSPaccmax,1)].Spectrum
    spD2=spectrum_with_acc_df.loc[np.round(tmax,1),np.round(loggmax,2),np.round(logSPaccmax,1)].Spectrum

    spS=(spA1/CA1+spB1/CB1+spC1/CC1+spD1/CD1+spA2/CA2+spB2/CB2+spC2/CC2+spD2/CD2)/np.sum([1/CA1,1/CB1,1/CC1,1/CD1,1/CA2,1/CB2,1/CC2,1/CD2])
    return(spS)

def task4syntetic_photometry(logacc,iso_df,spectrum_with_acc_df,spectrum_without_acc_df,age,m_list,vega_spectrum,interp,bp_dict,mag_label_list,spectrum_T_list,spectrum_logg_list,showplot,inst_list,filter_list,T_label='Teff',logg_label='logg',mass_label='mass',dist=10,A=np.array([0,0]),B=np.array([0,1]),C=np.array([1,1]),D=np.array([1,0]),rn=4,area=45238.93416):
    area*= units.AREA
    mag_mass_list=[]
    log_star_acc_lum_list=[]
    if len(m_list)==0: m_list=iso_df[mass_label].unique()
    for m in m_list:
        try:
            mag_list=[]
            teff=iso_df.loc[(iso_df.mass==m),T_label].values[0]
            logg=iso_df.loc[(iso_df.mass==m),logg_label].values[0]
            tmin=spectrum_T_list[spectrum_T_list<teff].max()
            tmax=spectrum_T_list[spectrum_T_list>teff].min()
            try:loggmin=spectrum_logg_list[spectrum_logg_list<logg].max()
            except:loggmin=logg
            try:loggmax=spectrum_logg_list[spectrum_logg_list>logg].min()
            except:loggmax=logg

            S=np.array([(teff-tmin)/(tmax-tmin),(logg-loggmin)/(loggmax-loggmin)])
            CA=np.linalg.norm(S-A)
            CB=np.linalg.norm(S-B)
            CC=np.linalg.norm(S-C)
            CD=np.linalg.norm(S-D)

            if not (loggmin in spectrum_with_acc_df.loc[tmin].index.get_level_values('logg').unique()):
                loggmin_tmin=loggmin+0.5#spectrum_with_acc_df.loc[tmin].index.get_level_values('logg').unique().min()
            else:
                loggmin_tmin=loggmin
            if not (loggmax in spectrum_with_acc_df.loc[tmin].index.get_level_values('logg').unique()):
                loggmax_tmin=loggmax-0.5#spectrum_with_acc_df.loc[tmin].index.get_level_values('logg').unique().max()
            else:
                loggmax_tmin=loggmax

            if not (loggmin in spectrum_with_acc_df.loc[tmax].index.get_level_values('logg').unique()):
                loggmin_tmax=loggmin+0.5#spectrum_with_acc_df.loc[tmax].index.get_level_values('logg').unique().min()
            else:
                loggmin_tmax=loggmin
            if not (loggmax in spectrum_with_acc_df.loc[tmax].index.get_level_values('logg').unique()):
                loggmax_tmax=loggmax-0.5#spectrum_with_acc_df.loc[tmax].index.get_level_values('logg').unique().max()
            else:
                loggmax_tmax=loggmax
            # print(spectrum_with_acc_df)
            # print(tmin,loggmin_tmin)
            spA=spectrum_with_acc_df.loc[tmin,loggmin_tmin].Spectrum
            spB=spectrum_with_acc_df.loc[tmin,loggmax_tmin].Spectrum
            spC=spectrum_with_acc_df.loc[tmax,loggmin_tmax].Spectrum
            spD=spectrum_with_acc_df.loc[tmax,loggmax_tmax].Spectrum
            spS=(spA/CA+spB/CB+spC/CC+spD/CD)/np.sum([1/CA,1/CB,1/CC,1/CD])

            # display(spectrum_without_acc_df.loc[tmin,loggmin_tmin,slice(None)].Spectrum.values[0])
            # spA_s=spectrum_without_acc_df.loc[tmin,loggmin_tmin,slice(None)].Spectrum.values[0]
            # spB_s=spectrum_without_acc_df.loc[tmin,loggmax_tmin,slice(None)].Spectrum.values[0]
            # spC_s=spectrum_without_acc_df.loc[tmax,loggmin_tmax,slice(None)].Spectrum.values[0]
            # spD_s=spectrum_without_acc_df.loc[tmax,loggmax_tmax,slice(None)].Spectrum.values[0]

            spA_s=spectrum_without_acc_df.loc[tmin,loggmin_tmin].Spectrum
            spB_s=spectrum_without_acc_df.loc[tmin,loggmax_tmin].Spectrum
            spC_s=spectrum_without_acc_df.loc[tmax,loggmin_tmax].Spectrum
            spD_s=spectrum_without_acc_df.loc[tmax,loggmax_tmax].Spectrum
            spS_s=(spA_s/CA+spB_s/CB+spC_s/CC+spD_s/CD)/np.sum([1/CA,1/CB,1/CC,1/CD])
            star_wl=spS_s.waveset
            star_wl=star_wl[star_wl>=3000*u.AA]

            star_flux=(np.trapz(units.convert_flux(star_wl,spS_s(star_wl),u.erg/u.cm**2/u.s/u.AA),x=star_wl)*area).to(u.Lsun)
            ratio=10**(interp['logL'](np.log10(teff),np.log10(age),logg))*u.Lsun/star_flux

            star_acc_wl=spS.waveset
            star_acc_wl=star_acc_wl[star_acc_wl>=3000*u.AA]

            star_total_flux=(np.trapz(units.convert_flux(star_acc_wl,spS(star_acc_wl),u.erg/u.cm**2/u.s/u.AA),x=star_acc_wl)*area).to(u.Lsun)
            log_star_acc_lum_list.append(np.log10((star_total_flux.value-star_flux.value)*ratio))

            R=interp['R'](np.log10(teff),np.log10(age),logg)*1e9

            for mag in mag_label_list:
                bp=bp_dict[mag]
                if mag=='m336_unleak': waveset=np.arange(np.nanmin(spS.waveset.value),5000,1)*u.AA
                else: waveset=np.arange(np.nanmin(spS.waveset.value), np.nanmax(spS.waveset.value), 1) * u.AA
                mag=syntetic_photometry(spS,vega_spectrum,bp,waveset,R)
                mag_list.append(np.round(mag.value,rn))
            mag_mass_list.append(mag_list)
            if showplot:
                print('Showing test:')
                wavelenghts=[]
                phot_df=pd.DataFrame({'filters':filter_list,'bp':list(bp_dict.values()),'inst':inst_list}).reset_index().set_index(['inst','index'])

                phot_df['wavelenghts']=np.nan
                phot_df['mag']=np.nan
                phot_df['magA']=np.nan
                phot_df['magB']=np.nan
                phot_df['magC']=np.nan
                phot_df['magD']=np.nan
                phot_df['mag_average']=np.nan
                phot_df['flux']=np.nan
                phot_df['fluxA']=np.nan
                phot_df['fluxB']=np.nan
                phot_df['fluxC']=np.nan
                phot_df['fluxD']=np.nan
                phot_df['flux_average']=np.nan
                n=0
                for mag in mag_label_list:
                    bp=bp_dict[mag]
                    if mag=='m336_unleak': waveset=np.arange(np.nanmin(spS.waveset.value),5000,1)*u.AA
                    else: waveset=np.arange(np.nanmin(spS.waveset.value), np.nanmax(spS.waveset.value), 1) * u.AA
                    magA,fluxA,_=syntetic_photometry(spA,vega_spectrum,bp,waveset,interp['R'](np.log10(tmin),np.log10(age),loggmin_tmin)*1e9,return_flux=True)
                    magB,fluxB,_=syntetic_photometry(spB,vega_spectrum,bp,waveset,interp['R'](np.log10(tmin),np.log10(age),loggmax_tmin)*1e9,return_flux=True)
                    magC,fluxC,_=syntetic_photometry(spC,vega_spectrum,bp,waveset,interp['R'](np.log10(tmax),np.log10(age),loggmin_tmax)*1e9,return_flux=True)
                    magD,fluxD,_=syntetic_photometry(spD,vega_spectrum,bp,waveset,interp['R'](np.log10(tmax),np.log10(age),loggmax_tmax)*1e9,return_flux=True)
                    magS,fluxS,wavelength=syntetic_photometry(spS,vega_spectrum,bp,waveset,interp['R'](np.log10(teff),np.log10(age),logg)*1e9,return_flux=True)
                    # print(wavelength)
                    # phot_df['wavelenghts'].iloc[n]=np.round(wavelenghts.value,rn)
                    phot_df['mag'].iloc[n]=np.round(magS.value,rn)
                    phot_df['magA'].iloc[n]=np.round(magA.value,rn)
                    phot_df['magB'].iloc[n]=np.round(magB.value,rn)
                    phot_df['magC'].iloc[n]=np.round(magC.value,rn)
                    phot_df['magD'].iloc[n]=np.round(magD.value,rn)
                    phot_df['mag_average'].iloc[n]=np.round(np.average([magA.value,magB.value,magC.value,magD.value],weights=[1/CA,1/CB,1/CC,1/CD]),rn)
                    phot_df['flux'].iloc[n]=np.round(fluxS.value,rn)
                    phot_df['fluxA'].iloc[n]=np.round(fluxA.value,rn)
                    phot_df['fluxB'].iloc[n]=np.round(fluxB.value,rn)
                    phot_df['fluxC'].iloc[n]=np.round(fluxC.value,rn)
                    phot_df['fluxD'].iloc[n]=np.round(fluxD.value,rn)
                    phot_df['flux_average'].iloc[n]=np.round(np.average([fluxA.value,fluxB.value,fluxC.value,fluxD.value],weights=[1/CA,1/CB,1/CC,1/CD]),rn)
                    n+=1
                display(phot_df)

                wav = np.arange(np.nanmin(spS.waveset.value), np.nanmax(spS.waveset.value), 1) * u.AA
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=wav.value, y=units.convert_flux(wav,spA(wav),units.FLAM),mode='lines',name='Spectrum teff= %s logg= %s'%(tmin,loggmin_tmin), marker=dict(size=40,color='grey')))
                fig.add_trace(go.Scatter(x=wav.value, y=units.convert_flux(wav,spB(wav),units.FLAM),mode='lines',name='Spectrum teff= %s logg= %s'%(tmin,loggmax_tmin), marker=dict(size=40,color='grey')))
                fig.add_trace(go.Scatter(x=wav.value, y=units.convert_flux(wav,spC(wav),units.FLAM),mode='lines',name='Spectrum teff= %s logg= %s'%(tmax,loggmin_tmax), marker=dict(size=40,color='grey')))
                fig.add_trace(go.Scatter(x=wav.value, y=units.convert_flux(wav,spD(wav),units.FLAM),mode='lines',name='Spectrum teff= %s logg= %s'%(tmax,loggmax_tmax), marker=dict(size=40,color='grey')))
                fig.add_trace(go.Scatter(x=wav.value, y=units.convert_flux(wav,spS(wav),units.FLAM),mode='lines',name='Spectrum teff= %s logg= %s'%(teff,logg), marker=dict(size=40,color='black')))

                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="fluxA", hover_data=["filters"],color_discrete_sequence=['grey']).data[0])
                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="fluxB", hover_data=["filters"],color_discrete_sequence=['grey']).data[0])
                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="fluxC", hover_data=["filters"],color_discrete_sequence=['grey']).data[0])
                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="fluxD", hover_data=["filters"],color_discrete_sequence=['grey']).data[0])

                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="flux", color="inst", hover_data=["filters"]).data[0])
                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="flux", color="inst", hover_data=["filters",]).data[1])
                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="flux", color="inst", hover_data=["filters"]).data[2])
                # fig.add_trace(px.scatter(phot_df.reset_index(), x="wavelenghts", y="flux", color="inst", hover_data=["filters",]).data[3])

                fig.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
                fig.update_yaxes(type="log")
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="LightSteelBlue",xaxis_title="Wavelenghts [A]",xaxis=dict(range=[1000, 20000]),yaxis_title="FLAM",width=1600, height=700,yaxis_tickformat = '.0e',xaxis_tickformat = 'i')
                fig.show()
        except:
            print(logacc,m,teff,tmin,tmax,logg,loggmin,loggmax,loggmin_tmin,loggmax_tmin,loggmin_tmax,loggmax_tmax)
            # print(spectrum_with_acc_df.loc[tmin].index.get_level_values('logg').unique())
            # print(spectrum_with_acc_df.loc[tmax].index.get_level_values('logg').unique())
            sys.exit()

    return(np.array(mag_mass_list),logacc,m_list,log_star_acc_lum_list)

def build_the_isochrones(input_spectrum_with_acc_df,input_spectrum_without_acc_df,input_iso_df,interp,bp_dict,mag_label_list,age_list=[],logacc_list=[],m_list=[],showplot=False,sp_Teff_label='Teff',sp_logg_label='logg',iso_age_label='Age',iso_logAcc_label='logAcc',iso_mass_label='mass',workers=None,inst_list=['WFPC2','WFPC2','WFPC2','WFPC2','WFPC2','ACS','ACS','ACS','ACS','ACS','NICMOS','NICMOS','WFC3','WFC3'],filter_list=['F336W_unleak','F336W','F439W','F656N','F814W','F435W','F555W','F658N','F775W','F850LP','F110W','F160W','F130N','F139M']):
    if workers==None:
        ncpu=multiprocessing.cpu_count()
        if ncpu>=3: workers=ncpu-2
        else: workers=1
    vega_spectrum = SourceSpectrum.from_vega()
    spectrum_T_list=np.sort(input_spectrum_with_acc_df.index.get_level_values(sp_Teff_label).unique())
    spectrum_logg_list=np.sort(input_spectrum_with_acc_df.index.get_level_values(sp_logg_label).unique())
    if len(age_list)==0:age_list=input_iso_df.index.get_level_values(iso_age_label).unique().values

    if len(logacc_list)==0: logacc_list=input_iso_df.index.get_level_values(iso_logAcc_label).unique().values
    spectrum_without_acc_df=input_spectrum_without_acc_df.loc[(slice(None),slice(None))]
    # display(input_spectrum_with_acc_df)
    for age in age_list:
        print('> Age: %s'%age)
        iso_df=[input_iso_df.loc[age,i].reset_index(drop=True) for i in logacc_list]
        spectrum_with_acc_df=[input_spectrum_with_acc_df.loc[(slice(None),slice(None),i)] for i in logacc_list]
        # display(spectrum_with_acc_df)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for outs,logacc,m_out_list,log_star_acc_lum_list in tqdm(executor.map(task4syntetic_photometry,logacc_list,iso_df,spectrum_with_acc_df,repeat(spectrum_without_acc_df),repeat(age),repeat(m_list),repeat(vega_spectrum),repeat(interp),repeat(bp_dict),repeat(mag_label_list),repeat(spectrum_T_list),repeat(spectrum_logg_list),repeat(showplot),repeat(inst_list),repeat(filter_list))):
                for elno in range(len(m_out_list)):
                    m=m_out_list[elno]
                    input_iso_df.loc[(input_iso_df.index.get_level_values('Age')==age)&(input_iso_df.index.get_level_values('logAcc')==logacc)&(input_iso_df['mass']==m),mag_label_list]=outs[elno]
                    input_iso_df.loc[(input_iso_df.index.get_level_values('Age')==age)&(input_iso_df.index.get_level_values('logAcc')==logacc)&(input_iso_df['mass']==m),'logLacc']=log_star_acc_lum_list[elno]
    return(input_iso_df)

def degrade_resolution(wavelengths, flux, center_wave, spec_res, disp, px_tot):
### To coadd the spectra I degrade their resolution, using the following function"#double check
#purpose is to mimic what output would look like based on the inherent flaws of optics of telescope
#basic process: convert everything to velocity, make a gaussian kernal and convolve it with the rest of the function,
#               then convert back to wavelength

    # print('in degrade resolution')
    # print(wavelengths)
#[1.5  1.76] [1. 1.]
#1.63
#0.40471199999999996 1.34252 4096
    #allow more px for K band
#     Npix_spec=px_tot * 3./2.  # if px_tot=4096, this is 6144 pixels
    Npix_spec=px_tot #* 3./2.  # if px_tot=4096, this is 6144 pixels

    #the log of speed of light in cm/s
    logc=np.log10(29979245800.)

    # make velocity array from -300,000 to 300,000 Km/s
    vel=(np.arange(600001)-300000) # big array of integers....

    # the array of wavelengths coming in input is converted in velocity difference vs. central wavelength, in km/s
    in_vel=(wavelengths/center_wave-1)*10.**(1*logc-5)

    # we can get non-physical velocities: kill them and their relative input flux array
    #create vectors in velocity space, picking out realistic values (keeping good indices)
    in_vel_short = in_vel[ np.where( (in_vel > vel[0]) & (in_vel < vel[600000]) )[0] ]
    in_flux_short = flux[ np.where( (in_vel > vel[0]) & (in_vel < vel[600000]) )[0] ]

    # this new arrays of velocities from the center_wave, and relative fluxes, are not uniformly sampled...
    # interpolate to equally spaced km/s, i.e. 600000 points
    interp_flux = np.interp(vel, in_vel_short, in_flux_short)
#    print(interp_flux,vel, in_vel_short, in_flux_short)
#[1. 1. 1. ... 1. 1. 1.] [-300000 -299999 -299998 ...  299998  299999  300000] [-23909.82793865  23909.82793865] [1. 1.]

    # sigma  = the resolution of the spectrograph in km/s, expressed as sigma of the Gaussian response.
    # it is Delta_lam/lam = Delta_v/c = FWHM/c = 2SQRT(2log2)/c, since
    # FWHM = 2*SQRT(2*log2) = 2.35
    sigma = (10.**(logc-5)/spec_res)/(2*np.sqrt(2*np.log(2)))
    #for R=1000 it is sigma = 127.31013507066515 [km/s]; instead of 300km/s, that would be the FWHM
    # make a smaller velocity array with
    # the same "resolution" as the steps in
    # vel, above
    n = round(8.*sigma) # = 1012kms
    # make sure that n is even...
    if (n % 2 == 0):
        n = n + 1
    #create an array -4*sigma to +4*sigma in km/s, e.g. from -509 to +508
    vel_kernel = np.arange(n) - np.floor(n/2.0)

    # create a normalized gaussian (unit area) with width=sigma
    gauss_kernel = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*vel_kernel**2.0/sigma**2.0)
        # shape the kernel
        # look up the equation for gaussian kernal and figure out the significance of sigma used here
        # like how does resolution shape/define the kernel
#    print(sigma,vel_kernel,gauss_kernel)
#314569.7065336959
#[-1258279. -1258278. -1258277. ...  1258277.  1258278.  1258279.]
#[4.25438073e-10 4.25443483e-10 4.25448893e-10 ... 4.25448893e-10 4.25443483e-10 4.25438073e-10]
    # convolve flux with gaussian kernel
    convol_flux = np.convolve(interp_flux, gauss_kernel , mode="same")
    # convert old moving kernel
    convol_wave = center_wave * (vel*10.**(-1*logc+5.0) + 1.0 ) # [micron]
    # convert back to wavelength

    # and the real pixel scale
    real_wave = np.arange(Npix_spec) * disp * 10.**(-4.)     #6000pix * 10A/pix * 1E-4mic/A  => [micron]
    real_wave = real_wave - real_wave[int(np.round(Npix_spec/2.))]
    real_wave = real_wave + center_wave # [pixel]    print(real_wave)
#     real_wave=wavelengths
    # wavelength to px

    # interpolate onto the pixel scale of the detector
    out_wave = real_wave
#    print('out_wave',out_wave)
    out_flux = np.interp(real_wave , convol_wave, convol_flux)
        # interpolating to number of px (b/c working from km/px or lam/px)

    out = {"lam": out_wave, #[micron]
          "flux": out_flux} #same unit in input e.g. erg/cm2/s/micron

    return(out)

def line_flux_correction(path2spectrum,spectrum_filename,area=45238.93416,shift=0,A=1.5,B=1.12,rl_min=6559,rl_max=6561,rl=6560,rl_sub=6559,savefile=False,verbose=False):
    # A,B da relazione di Alcala' 2014 tra Lacc e LHa

    area=area* units.AREA
    sp_acc = SourceSpectrum.from_file(path2spectrum+'/'+spectrum_filename)

    #we need to correct the flux in Ha for this acc spectrum
    #flusso totale dello spettro di accrescimento

    acc_sp_wl = sp_acc.waveset
    acc_sp_fl = sp_acc(acc_sp_wl)# in PHOTLAM

    acc_line_wl = np.arange(rl_min,rl_max+1,1)*u.AA
    acc_line_fl = sp_acc(acc_line_wl)# in PHOTLAM
    acc_line_lum=(np.trapz(units.convert_flux(acc_line_wl,acc_line_fl,u.erg/u.cm**2/u.s/u.AA),x=acc_line_wl)*area).to(u.Lsun)

    photlam_cont1=acc_line_fl[acc_line_wl==rl_min*u.AA]
    photlam_cont2=acc_line_fl[acc_line_wl==rl_max*u.AA]
    photlam_peak=acc_line_fl[acc_line_wl==rl*u.AA]
    photlam_ratio=(photlam_peak.value/photlam_cont1.value)[0]


    flam_cont1=units.convert_flux(rl_min*u.AA,acc_line_fl[acc_line_wl==rl_min*u.AA],u.erg/u.cm**2/u.s/u.AA)
    flam_cont2=units.convert_flux(rl_max*u.AA,acc_line_fl[acc_line_wl==rl_max*u.AA],u.erg/u.cm**2/u.s/u.AA)
    flam_peak=units.convert_flux(rl*u.AA,acc_line_fl[acc_line_wl==rl*u.AA],u.erg/u.cm**2/u.s/u.AA)
    flam_ratio=(flam_peak.value/flam_cont1.value)[0]

    if verbose:
        print('> line cont1 ',photlam_cont1,flam_cont1)
        print('> line peak', photlam_peak,flam_peak)
        print('> line cont2', photlam_cont2,flam_cont2)
        print('> ratio line/cont: %s PHOTLAM, %s FLAM'%(photlam_ratio,flam_ratio))

    # flusso della riga Ha + spettro
    acc_lum=(np.trapz(units.convert_flux(acc_sp_wl,acc_sp_fl,u.erg/u.cm**2/u.s/u.AA),x=acc_sp_wl)*area).to(u.Lsun)

    if verbose: print('> AccSp integrated luminosity: %e'%acc_lum.value)


    # # flusso totale dello spettro senza riga Ha (ottenuto mettendo al valore del continuo il picco Ha)

    acc_noline_wl = np.arange(rl_min,rl_max+1,1)*u.AA
    acc_noline_fl = sp_acc(acc_noline_wl)# in PHOTLAM
    acc_noline_fl[acc_noline_wl==rl*u.AA]=acc_noline_fl[acc_noline_wl==rl_sub*u.AA]
    acc_noline_lum=(np.trapz(units.convert_flux(acc_noline_wl,acc_noline_fl,u.erg/u.cm**2/u.s/u.AA),x=acc_noline_wl)*area).to(u.Lsun)
    if verbose: print('> AccSp-line integrated luminosity: %s'%acc_noline_lum)

    # # Flusso nella linea

    flux_line=abs(acc_line_lum-acc_noline_lum)
    if verbose: print('> line excees luminosity: %s'%flux_line)


    log_lum_acc=A+B*np.log10(flux_line.value)
    ratio=acc_lum.value/(10**log_lum_acc)
    if verbose: print('> ratio (AccSp/excess line) integrated luminosity: %e'%ratio)

    # flusso della spettro con nuova riga Ha

    # new_acc_sp_wl = sp_acc.waveset+shift*u.AA
    # new_acc_sp_fl = sp_acc(acc_sp_wl)# in PHOTLAM
    # new_acc_sp_fl[new_acc_sp_wl.value==rl+shift]=new_acc_sp_fl[new_acc_sp_wl.value==rl+shift]*photlam_ratio
    # if verbose:
    #     print('> new line peak', new_acc_sp_fl[new_acc_sp_wl.value==rl+shift],units.convert_flux((rl+shift)*u.AA,new_acc_sp_fl[new_acc_sp_wl.value==rl+shift],u.erg/u.cm**2/u.s/u.AA))
        # print('> new line integrated luminosity: %s'% (np.trapz(units.convert_flux(acc_line_wl,new_acc_sp_fl[(new_acc_sp_wl>=rl_min*u.AA)&(new_acc_sp_wl<=rl_max*u.AA)],u.erg/u.cm**2/u.s/u.AA),x=acc_line_wl)*area).to(u.Lsun))
        # print('> new AccSp integrated luminosity: %s'% (np.trapz(units.convert_flux(new_acc_sp_wl,new_acc_sp_fl,u.erg/u.cm**2/u.s/u.AA),x=new_acc_sp_wl)*area).to(u.Lsun))

    # if savefile:
    #     col1 = fits.Column(name='WAVELENGTH', format='f8', array=new_acc_sp_wl)
    #     col2 = fits.Column(name='FLUX', format='f8', array=units.convert_flux(new_acc_sp_wl,new_acc_sp_fl,u.erg/u.cm**2/u.s/u.AA))
    #     cols = fits.ColDefs([col1, col2])
    #     hdu = fits.BinTableHDU.from_columns(cols)
    #     hdu.writeto(path2spectrum+'/doctored_'+spectrum_filename,overwrite=True)


def rebin_spectrum(spectrum_wl,spectrum_fl,wmin=None,wmax=None,wstep=10000,step=10,shift=0,ncycles=None,savefile=False,showplots=False,show_final_plot=False):
    flux_list=np.array([])
    wavelength_list=np.array([])

    # spectrum = SourceSpectrum.from_file(path2spectrum+'/'+spectrum_filename)
    spectrum_wl = spectrum_wl.value
    spectrum_fl = units.convert_flux(spectrum_wl,spectrum_fl,u.erg/u.cm**2/u.s/u.AA).value

    if wmin==None: wmin=np.nanmin(spectrum_wl)
    if wmax==None: wmax=np.nanmax(spectrum_wl)
    mask=((spectrum_wl>=wmin)&(spectrum_wl<=wmax))
    spectrum_fl=spectrum_fl[mask]
    spectrum_wl=spectrum_wl[mask]


    col1 = fits.Column(name='WAVELENGTH', format='f8', array=spectrum_wl*u.AA)
    col2 = fits.Column(name='FLUX', format='f8', array=spectrum_fl* u.erg/u.cm**2/u.s/u.AA)
    cols = fits.ColDefs([col1, col2])
    spec = fits.BinTableHDU.from_columns(cols)
    specdata = spec.data



    if show_final_plot or showplots:
        # spectrum = SourceSpectrum.from_file(path2spectrum+'/'+spectrum_filename)
        # spectrum.plot(flux_unit='flam')

        spec1 = Spectrum1D(spectral_axis=specdata['WAVELENGTH'] * u.AA , flux=specdata['FLUX'] * u.erg/u.cm**2/u.s/u.AA)
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_title('Original Spectrum')
        ax.step(spec1.spectral_axis, spec1.flux)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()


    i=0
    cc=0

    while i < wmax:

        specdata_sel=specdata[[(specdata['WAVELENGTH']>=i)&(specdata['WAVELENGTH']<=wstep+i)]]
        lamb = specdata_sel['WAVELENGTH'] * u.AA
        flux = specdata_sel['FLUX'] * u.erg/u.cm**2/u.s/u.AA
        input_spec = Spectrum1D(spectral_axis=lamb, flux=flux)

        if showplots:
            quantity_support()  # for getting units on the axes below
            f, ax = plt.subplots(figsize=(10,7))
            ax.set_title('Selected Spectrum')
            ax.step(input_spec.spectral_axis, input_spec.flux)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.show()

        new_disp_grid = np.arange(i, wstep+i+step, step) * u.AA
        fluxcon = FluxConservingResampler()
        new_spec_fluxcon = fluxcon(input_spec,new_disp_grid)

        if showplots:
            f, ax = plt.subplots(figsize=(10,7))
            ax.set_title('Rebinned Selected Spectrum')
            ax.step(new_spec_fluxcon.spectral_axis, new_spec_fluxcon.flux)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.show()
            print('######################################################')

        flux_list=np.append(flux_list,new_spec_fluxcon.flux.value)
        wavelength_list=np.append(wavelength_list,new_spec_fluxcon.spectral_axis.value)


        i+=wstep
        if ncycles!=None and cc>=ncycles: break
        cc+=1

    if show_final_plot or showplots:
        spec2 = Spectrum1D(spectral_axis=wavelength_list * u.AA, flux=flux_list* u.erg/u.cm**2/u.s/u.AA)
        f, ax = plt.subplots(figsize=(10,7))
        ax.set_title('Combined Rebinned Spectra')
        ax.step(spec2.spectral_axis, spec2.flux)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()


    mask=~np.isnan(flux_list)
    wavelength_list=wavelength_list[mask]+shift
    flux_list=flux_list[mask]

    # if savefile:
    #     col1 = fits.Column(name='WAVELENGTH', format='f8', array=wavelength_list*u.AA)
    #     col2 = fits.Column(name='FLUX', format='f8', array=flux_list* u.erg/u.cm**2/u.s/u.AA)
    #     cols = fits.ColDefs([col1, col2])
    #     hdu = fits.BinTableHDU.from_columns(cols)
    #     hdu.writeto(path2spectrum+'/rebinned_'+spectrum_filename,overwrite=True)


    # else:

    return(dict({'flux':flux_list* u.erg/u.cm**2/u.s/u.AA,'lam':wavelength_list*u.AA}))


def star_plus_accretion_model(file_list,accr_list,path2accr_spect,acc_spec_filename='',path2models='/media/giovanni/DATA_store/Data/Giovanni/Synthetic Photometry/Models/bt_settl_agss2009',prename='',path2savedir='./',Amin=1000,Amax=300000,wmin=1000,wmax=25000,wstep=10000,showplot=False,bp_dict=[],star_wl_shift=0,acc_wl_shift=0,savefile=False,parallelize_runs=True,workers=5):
    # accr spectrum
    print('> loading AccSp: ',path2accr_spect+'/'+acc_spec_filename)
    sp_acc = SourceSpectrum.from_file(path2accr_spect+'/'+acc_spec_filename)

    Path(path2savedir).mkdir(parents=True, exist_ok=True)
    Path(path2models+'/Rebinned/').mkdir(parents=True, exist_ok=True)

    for file in tqdm(file_list):
        combine_star_model_and_accretion_task(file,accr_list,sp_acc,path2accr_spect,acc_spec_filename,prename,path2models,path2savedir,Amin,Amax,wmin,wmax,wstep,showplot,bp_dict,star_wl_shift,acc_wl_shift,savefile)

def combine_star_model_and_accretion_task(file,accr_list,sp_acc,path2accr_spect,acc_spec_filename,prename,path2models,path2savedir,Amin,Amax,wmin,wmax,wstep,showplot,bp_dict,star_wl_shift,acc_wl_shift,savefile):
    star_model=file.split('/')[-1]
    Teff=float(star_model.split('-')[0].split('lte')[1])*100
    logg=float(star_model.split('-')[1])
    wavelengths=np.arange(Amin,Amax+1,1)*u.AA
    area=1* units.AREA

    #star spectrum
    sp_star = SourceSpectrum.from_file(file) # This output in in PHOTLAM!!!
    star_wl = sp_star.waveset
    star_fl = sp_star(star_wl)# in PHOTLAM
    star_wl+=star_wl_shift*u.AA

    #total flux of the original star spectrum in PHOTLAM
    norm_factor_star = sp_star.integrate(wavelengths=wavelengths).value
    star_flux=sp_star.integrate(wavelengths=wavelengths).value

    #rebbinned star spectrum
    star_rebinned=rebin_spectrum(star_wl,star_fl,wmin=wmin,wmax=wmax,wstep=wstep,ncycles=None,showplots=False,show_final_plot=False)
    star_rebinned['flux_rn']=list(star_rebinned['flux'].value)*u.erg/u.cm**2/u.s/u.AA
    star_reb_flux=(np.trapz(star_rebinned['flux_rn'],x=star_rebinned["lam"])*area).to(u.Lsun)

    #Acc spectrum
    acc_wl = sp_acc.waveset
    acc_fl = sp_acc(acc_wl)# in PHOTLAM
    acc_wl+=acc_wl_shift*u.AA

    #total flux of the accretium spectrum in PHOTLAM
    norm_factor_sp_acc = sp_acc.integrate(wavelengths=wavelengths).value

    #normalized accretium spectrum in PHOTLAM
    sp_acc_rn = sp_acc/norm_factor_sp_acc
    acc_rn_wl = sp_acc_rn.waveset
    acc_rn_fl = sp_acc_rn(acc_rn_wl)# in PHOTLAM
    acc_rn_wl+=acc_wl_shift*u.AA

    #normalize accretium spectrum to total area of star spectrum
    sp_acc=sp_acc_rn*norm_factor_star
    acc_wl = sp_acc.waveset
    acc_fl = sp_acc(acc_wl)# in PHOTLAM
    acc_flux=sp_acc.integrate(wavelengths=wavelengths).value

    #rebbinned accretium spectrum
    acc_rebinned=rebin_spectrum(acc_wl,acc_fl,wmin=wmin,wmax=wmax,wstep=wstep,ncycles=None,showplots=False,show_final_plot=False)
    acc_rebinned['flux_rn']=list(acc_rebinned['flux'].value)*u.erg/u.cm**2/u.s/u.AA
    acc_reb_flux=(np.trapz(acc_rebinned['flux_rn'],x=acc_rebinned["lam"])*area).to(u.Lsun)

    for logacc in accr_list:
        # #combine the normalized star and ACC specturm
        star_acc_rebinned=(star_rebinned['flux_rn']+acc_rebinned['flux_rn']*float(10)**(logacc))

        # #flux of the star + accretion spectrum rescaled to the orignal flux of the star and coverted to Lsun
        star_acc_reb_flux=(np.trapz(star_acc_rebinned,x=acc_rebinned["lam"])*area).to(u.Lsun)

        if showplot:

            fig = make_subplots(rows=2, cols=1,shared_xaxes=True, vertical_spacing=0.02)

            fig.add_trace(go.Scatter(mode="lines", x=star_wl, y=star_fl.to(units.FLAM, u.spectral_density(star_wl)),name="OStar"),row=1, col=1)
            fig.add_trace(go.Scatter(mode="lines", x=star_rebinned["lam"], y=star_rebinned["flux_rn"],name="Star rebinned"),row=1, col=1)
            fig.add_trace(go.Scatter(mode="lines", x=acc_wl, y=acc_fl.to(units.FLAM, u.spectral_density(acc_wl)),name="AccSp"),row=1, col=1)
            fig.add_trace(go.Scatter(mode="lines", x=acc_rebinned["lam"], y=acc_rebinned["flux_rn"],name="AccSp rebinned"),row=1, col=1)
            fig.add_trace(go.Scatter(mode="lines", x=acc_rebinned["lam"], y=star_acc_rebinned,name="Star+AccSp rebinned"),row=1, col=1)
            fig.add_vline(x=Amin,line_width=3, line_dash="dash")
            fig.add_vline(x=Amax,line_width=3, line_dash="dash")
            fig.update_yaxes(type="log",row=1, col=1) # log range: 10^0=1, 10^5=100000
            fig.update_xaxes(type="log") # linear range

            for filter, bp in bp_dict.items():
                fig.add_trace(go.Scatter(mode="lines", x=bp.waveset, y=bp(bp.waveset),name=filter),row=2, col=1)


            fig.update_layout(xaxis_tickformat = 'E',
                              yaxis_tickformat = 'E',
                              autosize=False,
                              width=1500,
                              height=800,
                              margin=dict(
                                          l=50,
                                          r=50,
                                          b=50,
                                          t=50,
                                          pad=4
                                          ),
                              paper_bgcolor="LightSteelBlue",
                              )
            fig.show()
            print('> check if the accretium spectrum is rescaled correctly to the star spectrum (whitin %e - %e A, shoud be one): %f'%(Amin,Amax,star_flux/acc_flux))
            print('1 check if this is equal to the input logAcc %.10f'%np.log10(abs(star_reb_flux.value-star_reb_flux.value+acc_reb_flux.value*10**logacc)/acc_reb_flux.value))
            print('2 check if this is equal to the input logAcc %.10f'%np.log10(abs(star_reb_flux.value-star_acc_reb_flux.value)/acc_reb_flux.value))
            print('############################################################################')
        if savefile:
            data = Table([acc_rebinned["lam"], units.convert_flux(acc_rebinned["lam"],star_acc_rebinned,units.FLAM)], names=['#lam[Angstrom]', 'flux[erg/cm2/s/A]'])
            filename=prename+'_teff%i_logg%s_feh0.0_logacc%.1f.dat'%(Teff,logg,logacc)
            ascii.write(data, path2savedir+'/'+filename, overwrite=True)

            data = Table([acc_rebinned["lam"], units.convert_flux(star_rebinned["lam"],star_rebinned["flux_rn"],units.FLAM)], names=['#lam[Angstrom]', 'flux[erg/cm2/s/A]'])
            filename=prename+'_teff%i_logg%s_feh0.0_star_rebinned.dat'%(Teff,logg)
            ascii.write(data, path2models+'/Rebinned/'+filename, overwrite=True)

# def combine_star_model_and_accretion_task(file,accr_list,sp_acc,path2accr_spect,acc_spec_filename,prename,path2models,path2savedir,wmin,wmax,wstep,showplot,bp_dict,star_wl_shift,acc_wl_shift,savefile):
#     star_model=file.split('/')[-1]
#     Teff=float(star_model.split('-')[0].split('lte')[1])*100
#     logg=float(star_model.split('-')[1])

#     area=1* units.AREA
#     #star spectrum
#     sp_star = SourceSpectrum.from_file(file) # This output in in PHOTLAM!!!

#     star_wl = sp_star.waveset
#     star_fl = sp_star(star_wl)# in PHOTLAM
#     star_wl+=star_wl_shift*u.AA

#     #total flux of the original spectrum in PHOTLAM
#     wave=star_wl[(star_wl>1000*u.AA)&(star_wl<250000*u.AA)]
#     norm_factor_star = sp_star.integrate(wavelengths=wave).value
#     print('>',norm_factor_star)

#     #normalized spectrum in PHOTLAM
#     sp_star_rn = sp_star / norm_factor_star
#     print('>',sp_star_rn.integrate(wavelengths=wave).value )

#     star_rn_wl = sp_star_rn.waveset
#     star_rn_fl = sp_star_rn(star_rn_wl)# in PHOTLAM
#     star_rn_wl+=star_wl_shift*u.AA


#     acc_wl = sp_acc.waveset
#     acc_fl = sp_acc(acc_wl)# in PHOTLAM
#     acc_wl+=acc_wl_shift*u.AA
#     #total flux of the accretium spectrum in PHOTLAM
#     norm_factor_sp_acc = sp_acc.integrate(wavelengths=wave).value
#     print('>',norm_factor_sp_acc)

#     #normalized accretium spectrum in PHOTLAM
#     sp_acc_rn = sp_acc/norm_factor_sp_acc
#     print('>',sp_acc_rn.integrate(wavelengths=wave).value )

#     acc_rn_wl = sp_acc_rn.waveset
#     acc_rn_fl = sp_acc_rn(acc_rn_wl)# in PHOTLAM
#     acc_rn_wl+=acc_wl_shift*u.AA

#     star_rebinned=rebin_spectrum(star_rn_wl,star_rn_fl,wmin=wmin,wmax=wmax,wstep=wstep,ncycles=None,showplots=False,show_final_plot=False)
#     acc_rebinned=rebin_spectrum(acc_rn_wl,acc_rn_fl,wmin=wmin,wmax=wmax,wstep=wstep,ncycles=None,showplots=False,show_final_plot=False)

#     star_rebinned['flux_rn']=list(star_rebinned['flux'].value)*u.erg/u.cm**2/u.s/u.AA
#     acc_rebinned['flux_rn']=list(acc_rebinned['flux'].value)*u.erg/u.cm**2/u.s/u.AA

#     #total flux of the original spectrum in Lsun
#     star_flux=(np.trapz(units.convert_flux(star_wl,star_fl,u.erg/u.cm**2/u.s/u.AA),x=star_wl)*area).to(u.Lsun)

#     #star spectrum coverted to total flux in Lsun and rescaled to the total flux of the original spectrum
#     star_reb_flux=(np.trapz(star_rebinned['flux_rn'],x=star_rebinned["lam"])*area).to(u.Lsun)
#     star_reb_flux_ratio=(star_flux/star_reb_flux)

#     star_rebinned['flux_rn']*=star_reb_flux_ratio
#     star_reb_flux*=star_reb_flux_ratio

#     # accretion spectrum coverted to total flux in Lsun and rescaled to the total flux of the original spectrum
#     acc_flux=(np.trapz(units.convert_flux(acc_wl,acc_fl,u.erg/u.cm**2/u.s/u.AA),x=acc_wl)*area).to(u.Lsun)

#     #degarded accretion spectrum coverted to total flux in Lsun and rescaled to the total flux of the original spectrum
#     acc_reb_flux=(np.trapz(acc_rebinned['flux_rn'],x=acc_rebinned["lam"])*area).to(u.Lsun)

#     # acc_reb_ratio=(units.convert_flux(acc_rebinned["lam"],sp_acc(acc_rebinned["lam"]),u.erg/u.cm**2/u.s/u.AA)/acc_rebinned['flux_rn'])
#     acc_reb_flux_ratio=(acc_flux/acc_reb_flux)

#     acc_rebinned['flux_rn']*=acc_reb_flux_ratio
#     acc_reb_flux*=acc_reb_flux_ratio

#     for logacc in accr_list:
#         #combine the normalized star and ACC specturm
#         star_acc_rebinned=(star_rebinned['flux_rn']+acc_rebinned['flux_rn']*float(10)**(logacc))

#         #flux of the star + accretion spectrum rescaled to the orignal flux of the star and coverted to Lsun
#         star_acc_deg_flux=(np.trapz(star_acc_rebinned,x=acc_rebinned["lam"])*area).to(u.Lsun)

#         # print(star_reb_flux.value+acc_reb_flux.value*10**(logacc),star_acc_deg_flux.value)
#         if showplot:

#             fig = make_subplots(rows=2, cols=1,shared_xaxes=True, vertical_spacing=0.02)

#             fig.add_trace(go.Scatter(mode="lines", x=star_wl, y=star_fl.to(units.FLAM, u.spectral_density(star_wl)),name="OStar"),row=1, col=1)
#             fig.add_trace(go.Scatter(mode="lines", x=star_rn_wl, y=star_rn_fl.to(units.FLAM, u.spectral_density(star_rn_wl))*norm_factor_star,name="N2OStar"),row=1, col=1)
#             fig.add_trace(go.Scatter(mode="lines", x=star_rebinned["lam"], y=star_rebinned["flux_rn"],name="Star rebinned"),row=1, col=1)
#             fig.add_trace(go.Scatter(mode="lines", x=acc_rn_wl, y=acc_rn_fl.to(units.FLAM, u.spectral_density(acc_rn_wl))*norm_factor_sp_acc,name="AccSp"),row=1, col=1)
#             fig.add_trace(go.Scatter(mode="lines", x=acc_rebinned["lam"], y=acc_rebinned["flux_rn"],name="AccSp rebinned"),row=1, col=1)
#             fig.add_trace(go.Scatter(mode="lines", x=acc_rebinned["lam"], y=star_acc_rebinned,name="Star+AccSp rebinned"),row=1, col=1)
#             fig.update_yaxes(type="log",row=1, col=1) # log range: 10^0=1, 10^5=100000
#             fig.update_xaxes(type="log") # linear range

#             for filter, bp in bp_dict.items():
#                 fig.add_trace(go.Scatter(mode="lines", x=bp.waveset, y=bp(bp.waveset),name=filter),row=2, col=1)


#             fig.update_layout(xaxis_tickformat = 'E',
#                               yaxis_tickformat = 'E',
#                               autosize=False,
#                               width=1500,
#                               height=800,
#                               margin=dict(
#                                           l=50,
#                                           r=50,
#                                           b=50,
#                                           t=50,
#                                           pad=4
#                                           ),
#                               paper_bgcolor="LightSteelBlue",
#                               )
#             fig.show()
#             print('> check if the rebbined star flux is rescaled correctly to the original (shoud be one): ', star_flux/star_reb_flux)
#             print('> check if the rebbined accr spectrum flux is rescaled correctly to the original (shoud be one): ', acc_flux/acc_reb_flux)
#             print('> check if the total flux of the degraded spectrum + x% of accretion is rescaled correctly to the original (shoud be one): ', star_acc_deg_flux.value/(star_flux.value+acc_flux.value*10**(logacc)))
#             print('1 check if this is equal to the input logAcc',np.log10(abs(star_reb_flux.value-star_reb_flux.value+acc_reb_flux.value*10**logacc)/acc_reb_flux.value))
#             print('2 check if this is equal to the input logAcc',np.log10(abs(star_reb_flux.value-star_acc_deg_flux.value)/acc_reb_flux.value))
#             print('############################################################################')
#         if savefile:
#             data = Table([acc_rebinned["lam"], units.convert_flux(acc_rebinned["lam"],star_acc_deg_flux,units.FLAM)], names=['#lam[Angstrom]', 'flux[erg/cm2/s/A]'])
#             filename=prename+'_teff%i_logg%s_feh0.0_logacc%.1f.dat'%(Teff,logg,logacc)
#             ascii.write(data, path2savedir+'/'+filename, overwrite=True)

#             data = Table([acc_rebinned["lam"], units.convert_flux(star_rebinned["lam"],star_rebinned["flux_rn"],units.FLAM)], names=['#lam[Angstrom]', 'flux[erg/cm2/s/A]'])
#             filename=prename+'_teff%i_logg%s_feh0.0_star_rebinned.dat'%(Teff,logg)
#             ascii.write(data, path2models+'Rebinned/'+filename, overwrite=True)

def plot_SEDfit(sp_acc,spectrum_without_acc_df,vega_spectrum,bp_dict,sat_dict,interp_btsettl,row,mag_labels=[],fig=None,ax=None,Rv=3.1,ms=2,showplot=True,loc='lower right',plot_spSWO_ext=True, plot_spAcc_ext=True,sp_color='k',color='k',outlier_list=[],arrow_kwargs=dict(),dxy={},spaccfit=True):
    # Star's parameters
    if len(mag_labels)==0:
        mag_labels=np.array(['m336','m439','m656','m814','m435','m555','m658','m775','m850','m110','m160','m130','m139'])
    flam=u.erg*u.s**(-1)*u.cm**(-2)*u.AA**(-1)
    logSPacc=row.MCMC_logSPacc.values[0]
    logLacc=row.MCMC_logLacc.values[0]
    logMacc=row.MCMC_logMacc.values[0]
    teff=row.MCMC_T.values[0]
    if spaccfit:
        logg=np.round(interp_btsettl['logg'](np.log10(row.MCMC_mass),np.log10(row.MCMC_A),row.MCMC_logSPacc)[0],2)
    else:
        logg = np.round(interp_btsettl['logg'](np.log10(row.MCMC_mass), np.log10(row.MCMC_A))[0], 2)

    Av=row.MCMC_Av.values[0]
    Age=row.MCMC_A.values[0]
    Mass=row.MCMC_mass.values[0]
    if spaccfit:
        r=interp_btsettl['R'](np.log10(row.MCMC_mass.values[0]),np.log10(row.MCMC_A.values[0]),row.MCMC_logSPacc.values[0])*1e9
    else:
        r = interp_btsettl['R'](np.log10(row.MCMC_mass.values[0]), np.log10(row.MCMC_A.values[0])) * 1e9

    R=r*u.cm.to(u.pc)*u.pc
    d=row.MCMC_d.values[0]
    D=d*u.pc
    DM=5*np.log10(d/10)
    sp_acc=0

    # Star Spectrum /wo accr
    spSWO=interpolate_spectra2D(spectrum_without_acc_df,teff,logg)
    wavelengths=spSWO.waveset
    if spaccfit:
        #total flux of the accretium spectrum in PHOTLAM
        norm_factor_sp_acc = sp_acc.integrate(wavelengths=wavelengths).value
        #normalized accretium spectrum in PHOTLAM
        sp_acc_rn = sp_acc/norm_factor_sp_acc
        norm_factor_star = spSWO.integrate(wavelengths=wavelengths).value
        #normalize accretium spectrum to total area of star spectrum
        sp_acc=sp_acc_rn*norm_factor_star
        sp_acc*=10**(logSPacc)

        # Star Spectrum /w accr
        spSum=sp_acc+spSWO
    else:
        # Star Spectrum /wo accr
        spSum=spSWO

    waveset=spSum.waveset[(spSum.waveset.value>=3e3)&(spSum.waveset.value<=1.7e4)]

    # Star's extinction
    extinct = CCM89(Rv=Rv)
    ex = ExtinctionCurve(ExtinctionModel1D,points=waveset, lookup_table=extinct.extinguish(waveset, Av=Av))

    if showplot:
        fig,ax=plt.subplots(1,1,figsize=(15,15))
    if spaccfit:
        spSWO_ext = spSWO * ex
        if plot_spSWO_ext:
            ax.plot(waveset, units.convert_flux(waveset,spSWO_ext(waveset),flam)*(1/D)**2, c='r', ls= '-', label='Teff %i, logg %.2f'%(teff,logg),ms=ms)

        # Acc spectrum
        spAcc_ext = sp_acc*ex
        if plot_spAcc_ext:
            ax.plot(waveset, units.convert_flux(waveset,spAcc_ext(waveset),flam)*(1/D)**2, c='b', ls= '-', label='logSPacc %.2f'%logSPacc,ms=ms)

    # Star spectrum /w accr
    spSum_ext = spSum*ex
    if spaccfit:
        ax.plot(waveset, units.convert_flux(waveset,spSum_ext(waveset),flam)*(1/D)**2, c=sp_color,lw=3, ls= '-', label='Teff %i, logg %.2f, logMass %.2f, logAge %.2f, logAv %.2f\nlogSPacc %.2f, logLacc %.2f, logMacc %.2f'%(teff,logg,np.log10(Mass),np.log10(Age),np.log10(Av),logSPacc,logLacc,logMacc),ms=ms)
    else:
        ax.plot(waveset, units.convert_flux(waveset,spSum_ext(waveset),flam)*(1/D)**2, c=sp_color,lw=3, ls= '-', label='Teff %i, logg %.2f, logMass %.2f, logAge %.2f, logAv %.2f'%(teff,logg,np.log10(Mass),np.log10(Age),np.log10(Av)),ms=ms)

    ax.plot(0,0,'o',label='WFPC2',mfc='b',ms=20,mec='k',mew=2)
    ax.plot(0,0,'o',label='ACS',mfc='g',ms=20,mec='k',mew=2)
    ax.plot(0,0,'o',label='NICMOS',mfc='r',ms=20,mec='k',mew=2)
    ax.plot(0,0,'o',label='WFC3-IR',mfc='y',ms=20,mec='k',mew=2)
    deltas_dict={}
    Ndict={}
    for key in list(bp_dict.keys()):
        if key not in mag_labels:
            Ndict[filter]=False
        else:
            if key in list(bp_dict.keys()):
                if key in ['m336','m439','m656','m814']: color='b'
                elif key in ['m435','m555','m658','m775','m850']: color='g'
                elif key in ['m110','m160']: color='r'
                elif key in ['m130','m139']: color='y'
                mag=row[key].values[0]
                emag=row['e%s'%key[1:]].values[0]

                if isinstance(sat_dict[key],str):
                    f_spx=row['f_%s'%key].values[0]
                    if f_spx!=3 and emag<=0.1:
                        vobs=Observation(vega_spectrum, bp_dict[key],binset=bp_dict[key].waveset)
                        wav=vobs.effective_wavelength(wavelengths=bp_dict[key].waveset)
                        vega_flux=vobs.effstim(wavelengths=bp_dict[key].waveset,flux_unit='flam').value*flam

                        flux=10**(-mag/2.5)*vega_flux*(1/R)**2
                        ax.plot(wav,flux.value,'o',mfc=color,ms=20,mec='k',mew=2)

                        sp_flux=units.convert_flux(wav,spSum_ext(wav),units.FLAM)*(1/D)**2
                        deltas_dict[key]=(abs(flux-sp_flux)/sp_flux).value
                        Ndict[key]=True
                    else:Ndict[key]=False

                else:
                    if mag>=sat_dict[key] and emag<=0.1:

                        vobs=Observation(vega_spectrum, bp_dict[key],binset=bp_dict[key].waveset)
                        wav=vobs.effective_wavelength(wavelengths=bp_dict[key].waveset)
                        vega_flux=vobs.effstim(wavelengths=bp_dict[key].waveset,flux_unit='flam').value*flam

                        flux=10**(-mag/2.5)*vega_flux*(1/R)**2
                        ax.plot(wav,flux,'o',mfc=color,ms=20,mec='k',mew=2)

                        sp_flux=units.convert_flux(wav,spSum_ext(wav),units.FLAM)*(1/D)**2
                        deltas_dict[key]=(abs(flux-sp_flux)/sp_flux).value
                        Ndict[key] = True
                    else:
                        Ndict[key]=False

            if key in outlier_list:
                bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="b", lw=2)
                t = ax.text(wav.value+dxy[key][0],flux.value+dxy[key][1], "Outlier",bbox=bbox_props,**arrow_kwargs[key])

                bb = t.get_bbox_patch()

                bb.set_boxstyle("rarrow", pad=0.6)


    ax.set_yscale('log')
    ax.legend(loc=loc,fontsize=15)
    ax.set_xlim(3e3,1.7e4)
    ax.set_ylabel('FLAM [erg/cm2/s/A]')#,fontsize=20)
    ax.set_xlabel('Wavelenght [A]')#,fontsize=20)
    ax.grid()

    if showplot:
        plt.show()
    else:
        return(fig,ax,Ndict)