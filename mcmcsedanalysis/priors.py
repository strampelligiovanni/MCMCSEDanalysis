#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:09:40 2021

@author: giovanni
"""
import numpy as np
import scipy.stats as ss
import pandas as pd
from scipy.stats.kde import gaussian_kde

class generator(ss.rv_continuous):
    # def normal_pdf(self, x,mu,sig):
    #     pdf=normal_dist(x,mu,sig)
    #     return(pdf)

    # def lognormal_pdf(self, x,mu,sig):
    #     pdf=lognormal_dist(x,mu,sig)
    #     return(pdf)

    def chabrier_pdf(self,mass,cc,cmu,csig):       
        pdf=chabrier_dist(mass,cc,cmu,csig)
        return(pdf)
    
    def kroupa_pdf(self,mass):       
        pdf=kroupa_dist(mass)
        return(pdf)

# def normal_dist(x,mu,sig):
#     # k=(sig*np.sqrt(2*np.pi))
#     # f=k*np.exp(-0.5*((x-mu)/sig)**2)
#     f=norm.pdf(x,mu,sig)
#     return(f)

# def lognormal_dist(x,mu,sig):
#     # k=(x*sig*np.sqrt(2*np.pi))
#     # f=k*np.exp(-0.5*((np.log(x)-mu)/sig)**2)
#     f=lognorm.pdf(x,s=sig,loc=mu,scale=np.exp(mu))
#     return(f)

def chabrier_dist(mass,cc=0.158,cmu=0.079,csig=0.69):
    if mass <=1 :
        const = cc/np.log(10)
        exponent = -0.5 * np.square((np.log10(mass)-np.log10(cmu))/csig)
        k=const/mass
        return(k * np.exp(exponent))
    else:
        k =chabrier_dist(1,cc,cmu,csig)
        return(k * np.power(mass,-2.3))

def kroupa_dist(mass,a0=-0.3,a1=-1.3,a2=-2.4):
    if mass <=0.08: return(np.power(mass,a0))
    elif mass >0.08 and mass <=0.5:
        ka = np.power(0.08,a0-a1)
        return(ka*np.power(mass,a1))
    elif mass > 0.5: 
        ka = np.power(0.08,a0-a1)
        kb = np.power(0.5,a1-a2)
        return(kb*ka*np.power(mass,a2))   

# def mass_distributions(label,Nmass=500,showplot=False):
#     if label=='singles':
#         imf_in=imf.chabrier_single
#     elif label=='systems':
#         imf_in=imf.chabrier_not_resolved
#     elif label == 'kroupa':
#         imf_in=imf.kroupa
#     cluster,massfunc=show_cluster(imf_in,Nmass,showplot=showplot)
#     return(cluster,massfunc)

# def DaRio_dist(Av_min,Av_max,DaRio_finename='./DaRio_ACS_matched.csv'):
#     DaRio_df = pd.read_csv(DaRio_finename) 
#     DaRio_Av_sort=np.sort(DaRio_df.Av.values)
#     DaRio_pdf = gaussian_kde(DaRio_Av_sort[(DaRio_Av_sort>=Av_min)&(DaRio_Av_sort<=Av_max)])
#     return(DaRio_pdf)

def gaussian_kde_distribution(var,finename,var_minmax=[None,None]):
    df = pd.read_csv(finename) 
    df_sort=np.sort(df[var].values)
    if np.all(np.array(var_minmax)!=None): df_pdf = gaussian_kde(df_sort[(df_sort>=var_minmax[0])&(df_sort<=var_minmax[1])])
    else:df_pdf = gaussian_kde(df_sort)
    return(df_pdf)