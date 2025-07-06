#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:40:55 2021

@author: giovanni
"""
import os

import corner
import numpy as np
import matplotlib.pyplot as plt
from mcmcsedanalysis import mcmc_utils
from astropy import units as u
from glob import glob
from astropy.stats import sigma_clip

#############
# Ancillary #
#############

# def show_cluster(massfunc,mcluster,showplot=False):
#
#     cluster,yax,colors = coolplot(mcluster, massfunc=massfunc)
#
#     logspace_mass = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)
#     if showplot:
#         plt.figure(1, figsize=(10,8))
#         # plt.clf()
#         plt.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
#                     linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
#         plt.gca().set_xscale('log')
#         plt.plot(logspace_mass,np.log10(massfunc(logspace_mass)),'r--',linewidth=2,alpha=0.5)
#         plt.xlabel("Stellar Mass")
#         plt.ylabel("log(dN(M)/dM)")
#         plt.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
#         plt.show()
#     return(cluster,massfunc)

def sample_posteriors(interp,ID,ndim,verbose=True,path2loaddir='./',truths=[None,None,None,None,None],bins=20,pranges=None,labelpad=10,discard=None,thin=None,label_list=['logMass','logAv','logAge','logSPacc','Parallax'],kde_fit=False,showID=False,path2savedir=None,showplots=True,return_fig=False,return_variables=False,sigma=3.5,pmin=1.66,pmax=3.30,spaccfit=True):
        path2file=path2loaddir+'/samplerID_%i'%ID
        filename=glob(path2file)[0]
        if verbose:print(filename)
        mcmc_dict= mcmc_utils.read_samples(filename)
        samples=np.array(mcmc_dict['samples'])
        if discard!=None: samples=samples[discard:, :, :]
        if thin!=None: samples=samples[::thin, :, :]        
        flat_samples=samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
        filtered_flat_sample=sigma_clip(flat_samples, sigma=sigma, maxiters=5,axis=0)
        flat_samples=filtered_flat_sample.copy()
        if pranges==None: 
            pranges=[]
            for i in  range(flat_samples.shape[1]):
                pranges.append((np.nanmin(flat_samples[:,i][np.isfinite(flat_samples[:,i])]),np.nanmax(flat_samples[:,i][np.isfinite(flat_samples[:,i])])))
        if verbose:print('tau:', mcmc_dict['tau'])
        
        logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,kde_list,area_r= mcmc_utils.star_properties(flat_samples, ndim, interp, label_list=label_list, kde_fit=kde_fit, pmin=pmin, pmax=pmax, spaccfit=spaccfit)
        if verbose:
            print('\nStar\'s principal parameters:')

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(10 ** logMass, 10 ** logMass - 10 ** (logMass - elogMass_d),
                             10 ** (logMass + elogMass_u) - 10 ** logMass, 'mass')
            print(txt)

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(10 ** logAv, 10 ** logAv - 10 ** (logAv - elogAv_d),
                             10 ** (logAv + elogAv_u) - 10 ** logAv, 'Av')
            print(txt)

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(10 ** logAge, 10 ** logAge - 10 ** (logAge - elogAge_d),
                             10 ** (logAge + elogAge_u) - 10 ** logAge, 'A')
            print(txt)

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(logSPacc, elogSPacc_d, elogSPacc_u, r"logSPacc")
            print(txt)

            txt = r"\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
            txt = txt.format(Parallax, eParallax_d, eParallax_u, 'Parallax')
            print(txt)

            txt = r"\mathrm{{{1}}} = {0:.5f}"
            txt = txt.format(area_r, 'Area Ratio')
            print(txt)

    
            print('\nStar\'s derived parameters:')
            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            Dist=(Parallax* u.mas).to(u.parsec, equivalencies=u.parallax()).value
            eDist_d=Dist-((Parallax+eParallax_u)*u.mas).to(u.parsec, equivalencies=u.parallax()).value
            eDist_u=((Parallax-eParallax_d)*u.mas).to(u.parsec, equivalencies=u.parallax()).value-Dist
            txt = txt.format(Dist, eDist_d, eDist_u, 'Distance')
            print(txt)
    
            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(T, eT_d, eT_u, 'T')
            print(txt)

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(logL, elogL_d, elogL_u, 'logL')
            print(txt)

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(logLacc, elogLacc_d, elogLacc_u, r"logL_{acc}")
            print(txt)

            txt = r"\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(logMacc, elogMacc_d, elogMacc_u, r"logM_{acc}")
            print(txt)

        fig=plt.figure(figsize=(13, 13))
        figure = corner.corner(np.asarray(flat_samples),truths=truths,range=pranges,labels=label_list,plot_contours=True,fig=fig,bins=bins,hist_kwargs={'histtype':'stepfilled','color':'#6A5ACD','density':True,'alpha':0.35},contour_kwargs={'colors':'k','labelpad':labelpad},color='#6A5ACD')#,label_kwargs={'fontsize':20},title_kwargs={'fontsize':20})
        if showID: figure.suptitle('Star ID %i'%ID)
        axes = np.array(figure.axes).reshape((ndim, ndim))
        for i in range(ndim):
            if i == 0: 
                val= logMass
                eval_u= elogMass_u
                eval_d= elogMass_d
            elif i == 1: 
                val= logAv
                eval_u= elogAv_u
                eval_d= elogAv_d
            elif i == 2: 
                val= logAge
                eval_u= elogAge_u
                eval_d= elogAge_d
            elif i == 3: 
                val= logSPacc
                eval_u= elogSPacc_u
                eval_d= elogSPacc_d
            elif i == 4: 
                val= Parallax
                eval_u= eParallax_u
                eval_d= eParallax_d
    
            txt = r"$\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}$"
            txt = txt.format(val, eval_d, eval_u, label_list[i])
            ax = axes[i, i]
            ax.set_title(txt,size=22,fontweight="bold")
            ax.axvline(val-eval_d, color="k",linestyle='-.')
            ax.axvline(val, color="b",linestyle='-.',lw=2)    
            ax.axvline(eval_u+val, color="k",linestyle='-.')  
            if len(kde_list)>0:
                x=np.sort(flat_samples[:,i][~flat_samples[:,i].mask])
                y=kde_list[i].pdf(np.sort(x))
                ax.plot(x,y, color='k',lw=2)
        
        plt.tight_layout(w_pad=0.,h_pad=0.)

        os.makedirs(path2savedir, exist_ok=True)
        if isinstance(path2savedir, str): plt.savefig(path2savedir+'/cornerID%i.png'%ID, bbox_inches='tight')
        if return_fig: 
            plt.close()
            return(axes)
        else:
            if showplots: plt.show()
            else:plt.close('all')
        if return_variables:
            return(logMass,elogMass_u,elogMass_d,logAv,elogAv_u,elogAv_d,logAge,elogAge_u,elogAge_d,logSPacc,elogSPacc_u,elogSPacc_d,Parallax,eParallax_u,eParallax_d,T,eT_u,eT_d,logL,elogL_d,elogL_u,logLacc,elogLacc_d,elogLacc_u,logMacc,elogMacc_d,elogMacc_u,kde_list,area_r)

def sample_blobs(ID,path2loaddir='./',showID=False,show_samples=False,sigma=3,bins=20,pranges=None,fx=3,fy=3,discard=None,thin=None,labelpad=10):
    filename=glob(path2loaddir+'/*ID_%s'%ID)[0]
    print('> ',filename)
    mcmc_dict= mcmc_utils.read_samples(filename)
    blobs=np.array(mcmc_dict['blobs'])
    if discard!=None: blobs=blobs[discard:, :, :]
    if thin!=None: blobs=blobs[::thin, :, :]        
    flat_blobs=blobs.reshape(blobs.shape[0]*blobs.shape[1],blobs.shape[2])
    filtered_flat_blobs=sigma_clip(flat_blobs, sigma=sigma, maxiters=5,axis=0)
    flat_blobs=filtered_flat_blobs.copy()
    if pranges==None: 
        pranges=[]
        for i in  range(flat_blobs.shape[1]):
            pranges.append((np.nanmin(flat_blobs[:,i][np.isfinite(flat_blobs[:,i])]),np.nanmax(flat_blobs[:,i][np.isfinite(flat_blobs[:,i])])))
    label_list=[]
    truths=[]
    for label in mcmc_dict['variables_label']:
        label_list.append(label) 
    for variables in mcmc_dict['variables']:
        truths.append(variables)
            
    ndim=len(truths)
    figsize=(ndim*fx,ndim*fy)
    blobs_avg=[]
    if show_samples:

        fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(blobs[:, :, i], "k", alpha=0.3)
            blobs_avg.append(np.round(np.percentile(flat_blobs[:, i], [50])[0],4))
            ax.axhline(truths[i])
            ax.set_xlim(0, len(blobs))
            ax.set_ylabel(label_list[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        axes[-1].set_xlabel("step number")
        plt.tight_layout()
    figure=plt.figure(figsize=figsize)
    figure = corner.corner(flat_blobs,truths=truths,range=pranges,labels=label_list,plot_contours=True,fig=None,bins=bins,linewidth=3,hist_kwargs={'histtype':'stepfilled','color':'#6A5ACD','density':True,'alpha':0.35},contour_kwargs={'colors':'k','labelpad':labelpad},color='#6A5ACD')
    if showID: figure.suptitle('Star ID %i'%ID)
    axes = np.array(figure.axes).reshape((ndim, ndim))
    
    for i in range(ndim):
        x=np.sort(flat_blobs[:,i][~flat_blobs[:,i].mask])
        mcmc = np.percentile(x, [16, 50, 84])
        val =mcmc[1]
        eval_d,eval_u = np.diff([mcmc[0],val,mcmc[-1]])
        
        txt = r"$\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}$"
        txt = txt.format(val, eval_d, eval_u, label_list[i])
        ax = axes[i, i]
        ax.set_title(txt,size=18,fontweight="bold")
        ax.axvline(val-eval_d, color="k",linestyle='-.')
        ax.axvline(val, color="b",linestyle='-.',lw=2)    
        ax.axvline(eval_u+val, color="k",linestyle='-.')
    plt.tight_layout()
    plt.show()
