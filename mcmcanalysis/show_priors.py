# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys,statistics,pickle
sys.path.append('../')
from mcmcanalysis.priors import generator
# import mcmc_utils 
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats.kde import gaussian_kde
from scipy.stats import lognorm,skewnorm
# from twopiece.scale import tpnorm
# import scipy
# from matplotlib.ticker import PercentFormatter
# import matplotlib.ticker
from mcmcanalysis.kde import KDE

# def tpnormal_prior(x0,ex0,ex1,xlabel='x',size = 10000, nbins=20):
#     dist = tpnorm(loc=x0, sigma1=ex0, sigma2 =ex1)
#     sample = dist.random_sample(size = size)
#     x = np.linspace(min(sample), max(sample), size)
#     y = dist.pdf(x)
#     bins=np.linspace((min(x)), max(x),nbins)
#     fig,ax=plt.subplots(1,1,figsize=(7,7))
#     counts,_,_=ax.hist(sample, histtype='stepfilled', label='tpnormal dist',density=True,bins=bins,alpha=0.2)
#     # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
#     ax.plot(x, y,'k-', label='PDF',lw=2)  

#     # bins=np.logspace(np.log10(min(x)),np.log10(max(x)),nbins)
#     # ax[1].hist(sample, density=True, histtype='stepfilled', alpha=0.2,bins=bins,label='tpnormal dist')
#     # ax[1].plot(x, y, 'k-', lw=2, label='frozen pdf')
#     # ax[1].legend(loc='best', frameon=False)
#     # ax[1].set_xscale('log')
#     plt.show()


#     plt.show()   
#     density = counts / (sum(counts) * np.diff(bins))
#     print('area under the histogram:',np.sum(density * np.diff(bins)))
#     print('MEAN:',np.nanmean(sample))
#     print('MEDIAN:',np.nanmedian(sample))
#     print('MODE:',statistics.mode(sample))
 
def skewnormal_prior(x0,ex0,a=0,xlabel='x',size=10000,nbins=50):
    rv = skewnorm(a,loc=x0,scale=ex0)
    r = rv.rvs(size=size)
    x = np.linspace(min(r),max(r), 100)
    bins=np.linspace((min(r)), max(r),nbins)
    fig,ax=plt.subplots(1,1,figsize=(7,7))
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    counts,_,_=ax.hist(r, density=True, histtype='stepfilled', alpha=0.2,bins=bins,label='normal dist')
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc='best', frameon=False)
    plt.show()    
    density = counts / (sum(counts) * np.diff(bins))
    print('area under the histogram:',np.sum(density * np.diff(bins)))
    print('MEAN:',np.nanmean(r))
    print('MEDIAN:',np.nanmedian(r))
    print('MODE:',statistics.mode(r))
    print('Ax ratio',(max(r)-np.nanmedian(r))/(np.nanmedian(r)-min(r)))
    print('Skewness:',a)

def lognormal_prior(x0,ex0,xlabel='x',size=2000,nbins=50):
    rv = lognorm(ex0,scale=x0)
    r = rv.rvs(size=10000)
    x = np.linspace(0.01,20, size)
    fig,ax=plt.subplots(1,2,figsize=(14,7))
    
    bins=np.linspace(min(x),max(x),nbins)
    ax[0].plot(x, rv.pdf(x), 'k-', lw=2)
    counts,_,_=ax[0].hist(r, density=True, histtype='stepfilled', alpha=0.2,bins=bins)
    # ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    bins=np.logspace(np.log10(min(x)),np.log10(max(x)),nbins)
    ax[1].plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    ax[1].hist(r, density=True, histtype='stepfilled', alpha=0.2,bins=bins,label='lognormal dist')
    # ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax[1].legend(loc='best', frameon=False)
    ax[1].set_xscale('log')
    plt.show()

    density = counts / (sum(counts) * np.diff(bins))
    print('area under the histogram:',np.sum(density * np.diff(bins)))
    print('MEAN:',np.nanmean(r))
    print('MEDIAN:',np.nanmedian(r))
    print('MODE:',statistics.mode(r))
    
 
def kernel_prior(x,xlabel='x',ylabel='p',bins=None,nbins=13,showplot=True,bw_method=None,kernel='gaussian',bandwidth2fit=np.linspace(0.01, 1, 100),xlogscale=False,density=True,return_prior=False,savename='kde',path2savedir='./'):
    x_sort=np.sort(x)
    fig,ax=plt.subplots(figsize=(10,7))
    if not isinstance(bins, (list,np.ndarray)):
        if xlogscale:
            bins=np.logspace(np.log10(min(x_sort)),np.log10(max(x_sort)),nbins)
            # print(np.diff(np.log10(bins)))
        else:
            bins=np.linspace(min(x_sort),max(x_sort),nbins)
            # print(np.diff(bins))
    # my_kde=mcmc_utils.get_kde(x_sort,bw_method,auto_adjust=auto_adjust)
        
    # my_new_pdf,r1,r2=mcmc_utils.get_kde_pdf(x_sort,bw_method,auto_adjust=auto_adjust)
    kde=KDE(x_sort, bins, bandwidth=bw_method,kernel=kernel,bandwidth2fit=bandwidth2fit)
    kde.kde_sklearn()

    counts,_=np.histogram(x_sort,bins=bins)

    ax.hist(x_sort, histtype='stepfilled',bins=bins,density=density,alpha=0.2, label=xlabel)
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.plot(x_sort,kde.pdf(x_sort), label="PDF",color='k',lw=2)
    
    # ax.axvline(r1,linestyle='-.')
    # ax.axvline(r2,linestyle='-.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if xlogscale: 
        ax.set_xscale("log")
    plt.tight_layout()
    if showplot:
        density = counts / (sum(counts) * np.diff(bins))
        plt.show() 
        print('bandwidth:',kde.bandwidth)
        print('area under the histogram:',np.sum(density * np.diff(bins)))
        print('MEAN:',np.nanmean(x_sort))
        print('MEDIAN:',np.nanmedian(x_sort))
    else: plt.close()
    

    if return_prior: return(kde)
    else:
        print('Saving %s.pck in %s'%(savename,path2savedir))
        with open(path2savedir+"/%s.pck"%savename, 'wb') as file_handle:
            pickle.dump(kde , file_handle)


def IMF_prior(label,showplot=False):
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