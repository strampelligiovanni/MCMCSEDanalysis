# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
sys.path.append('./')
from priors import mass_distributions,generator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats.kde import gaussian_kde
import random


class Show:
    def distance_prior(dm0,edm0):
        samples = np.random.normal(dm0,edm0,10000)
        bins = np.linspace((min(samples)), max(samples), 200)
        bin_centers = 0.5*(bins[:-1] + bins[1:])
        gauss_dist=np.array([generator().normal_pdf(i,dm0,edm0) for i in  bin_centers])
        gauss_dist/=(sum(gauss_dist)*np.diff(bin_centers)[0])
        bins = np.linspace((min(samples)), max(samples), 20)
        
        fig,ax=plt.subplots(figsize=(7,7))
        ax.hist(samples, bins=bins, density=True,label='DM',alpha=0.7)
        ax.plot(bin_centers, gauss_dist, label="PDF")
        ax.set_xlabel('Distance')
    #     ax.set_xlim(dm_min,dm_max)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def T_prior(T0,eT0):
        samples = np.random.normal(T0,eT0,10000)
        bins = np.linspace((min(samples)), max(samples), 200)
        bin_centers = 0.5*(bins[:-1] + bins[1:])
        gauss_dist=np.array([generator().normal_pdf(i,T0,eT0) for i in  bin_centers])
        gauss_dist/=(sum(gauss_dist)*np.diff(bin_centers)[0])
        bins = np.linspace((min(samples)), max(samples), 20)
        
        fig,ax=plt.subplots(figsize=(7,7))
        ax.hist(samples, bins=bins, density=True,label='T',alpha=0.7)
        ax.plot(bin_centers, gauss_dist, label="PDF")
        ax.set_xlabel('Temperature')
    #     ax.set_xlim(dm_min,dm_max)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        print('MEAN:',np.nanmean(samples))
        print('MEDIAN:',np.nanmedian(samples))
    
    def m_prior(m0,em0):
        mu=np.log(m0**2/np.sqrt(m0**2+em0**2))
        sig=np.sqrt(np.log(1+em0**2/m0**2))
    
        samples = np.random.lognormal(mu,sig,10000)
        
        bins = np.linspace((min(samples)), max(samples), 100)
        bin_centers = 0.5*(bins[:-1] + bins[1:])
        gauss_dist=np.array([generator().lognormal_pdf(i,mu,sig) for i in  bin_centers])
        gauss_dist/=(sum(gauss_dist)*np.diff(bin_centers)[0])
        logbins = np.logspace(np.log10(min(samples)), np.log10(max(samples)), 20)
    
        fig,ax=plt.subplots(figsize=(7,7))
        ax.hist(samples, bins=logbins, density=True,label='Age',alpha=0.7)
        ax.plot(bin_centers, gauss_dist, label="PDF")
        ax.set_xlabel('Mass [Msun]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        print('MODE:',np.exp(mu-sig**2))
        print('MEAN:',np.nanmean(samples))
        print('MEDIAN:',np.nanmedian(samples))
        
    def t_prior(t0,et0):
        mu=np.log(t0**2/np.sqrt(t0**2+et0**2))
        sig=np.sqrt(np.log(1+et0**2/t0**2))
    
        samples = np.random.lognormal(mu,sig,10000)
        
        bins = np.linspace((min(samples)), max(samples), 100)
        bin_centers = 0.5*(bins[:-1] + bins[1:])
        gauss_dist=np.array([generator().lognormal_pdf(i,mu,sig) for i in  bin_centers])
        gauss_dist/=(sum(gauss_dist)*np.diff(bin_centers)[0])
        logbins = np.logspace(np.log10(min(samples)), np.log10(max(samples)), 20)
    
        fig,ax=plt.subplots(figsize=(7,7))
        ax.hist(samples, bins=logbins, density=True,label='Age',alpha=0.7)
        ax.plot(bin_centers, gauss_dist, label="PDF")
        ax.set_xlabel('Age [Myr]')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        print('MODE:',np.exp(mu-sig**2))
        print('MEAN:',np.nanmean(samples))
        print('MEDIAN:',np.nanmedian(samples))
        
    def DaRio_Av_prior(Av_min,Av_max,DaRio_finename='/media/giovanni/DATA_backup/Lavoro/Giovanni/NGC1976/ACS/DaRio_ACS_matched.csv'):
        DaRio_df = pd.read_csv(DaRio_finename) 
        DaRio_Av_sort=np.sort(DaRio_df.Av.values)
        DaRio_Av_sort=DaRio_Av_sort[(DaRio_Av_sort>=Av_min)&(DaRio_Av_sort<=Av_max)]
    
        ecdf = ECDF(DaRio_Av_sort)
        x=ecdf.y
        y=ecdf.x
        yfit = interp1d(x, y)
        xfit=np.linspace(np.nanmin(x), np.nanmax(x), 200)
    
        n=random.uniform(min(x), max(x))
    
        fig,ax=plt.subplots()
        ax.plot(xfit,yfit(xfit),'-r')
        ax.plot(ecdf.y, ecdf.x,'ob',ms=2)
    
        ax.axhline(yfit(n))
        ax.axvline(n)
        ax.set_ylabel('Av')
        ax.set_xlabel('N')
        plt.tight_layout()
    
        samples_extr=[]
        for xxx in range(len(DaRio_Av_sort)):
            n=random.uniform(min(x), max(x))
            samples_extr.append(yfit(n))
        samples_extr=np.array(samples_extr).ravel()
        samples_extr=np.sort(samples_extr[(~np.isnan(samples_extr))&(~np.isinf(samples_extr))])
        
        bins = np.linspace(0,13, 25)
        my_pdf = gaussian_kde(DaRio_Av_sort)
    
        fig,ax=plt.subplots(figsize=(7,7))
        ax.hist(samples_extr,bins=bins,density=True,alpha=0.7, label="Ext")
        ax.hist(DaRio_Av_sort,bins=bins,density=True,alpha=0.7, label="DaRio")
        ax.plot(DaRio_Av_sort,my_pdf(DaRio_Av_sort), label="PDF")
    
        ax.set_xlabel('Av')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def mass_prior(label,showplot=False):
        cluster,massfunc=mass_distributions(label,showplot=showplot,Nmass=500)
        cluster=np.sort(cluster)
        if label=='singles': emass_dist=np.array([generator().chabrier_pdf(i,cc=0,cmu=0.079,csig=0.69) for i in  cluster])
        elif label=='systems': emass_dist=np.array([generator().chabrier_pdf(i,cc=0.,cmu=0.22,csig=0.57) for i in  cluster])
        elif label=='kroupa': emass_dist=np.array([generator().kroupa_pdf(i) for i in  cluster])
    
        logbins = np.logspace(np.log10(min(cluster)), np.log10(max(cluster)),20)
        
        fig,ax=plt.subplots(figsize=(10,7))
        ax.hist(cluster, bins=logbins, density=True,label='Mass')
        ax.plot(cluster, emass_dist, label="Mario PDF")
        ax.plot(cluster, massfunc(cluster), label="Test PDF")
        ax.set_ylabel('dN/dM')
        ax.set_xlabel('mass')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
        fig,ax=plt.subplots(figsize=(10,7))
        ax.plot(np.log10(cluster), np.log10(massfunc(cluster)*cluster*np.log(10)), label="Test PDF")
        ax.plot(np.log10(cluster), np.log10(emass_dist*cluster*np.log(10)), label="Mario PDF")
        ax.set_ylabel('log(dN/dlog(M))')
        ax.set_xlabel('log(M)')
        ax.set_xlim(-2.5,1)
        ax.grid()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()