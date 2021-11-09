#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:40:55 2021

@author: giovanni
"""
import sys,corner,math
sys.path.append('/mnt/Storage/Lavoro/GitHub/imf-master/imf/')
from imf import coolplot
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy.interpolate import LinearNDInterpolator,Rbf
import plotly.graph_objects as go
import mcmc_utils
from IPython.display import display, Math

#############
# Ancillary #
#############

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def show_cluster(massfunc,mcluster,showplot=False):

    cluster,yax,colors = coolplot(mcluster, massfunc=massfunc)

    logspace_mass = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)
    if showplot:
        plt.figure(1, figsize=(10,8))
        # plt.clf()
        plt.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        plt.gca().set_xscale('log')
        plt.plot(logspace_mass,np.log10(massfunc(logspace_mass)),'r--',linewidth=2,alpha=0.5)
        plt.xlabel("Stellar Mass")
        plt.ylabel("log(dN(M)/dM)")
        plt.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
        plt.show()
    return(cluster,massfunc)

def interpND(*args,smooth=0,method='nearest',x_label='x',y_label='y',z_label=None,color_labels=None,showplot=False,radial=True,fx=1450,fy=500,w_pad=3,h_pad=1,pad=3,npoints=50,nrows=1,surface=True):
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
    if np.all(z_label==None): z_label=['z']*len(args[0][-1])
    if surface:
        meshed_coords = np.meshgrid(*[np.linspace(np.min(args[0][i]),np.max(args[0][i]),npoints) for i in range(len(args[0][:-1]))])
        new_coords=[meshed_coords[i].ravel() for i in range(len(meshed_coords))]
    else:
        new_coords=np.array([np.linspace(np.min(args[0][i]),np.max(args[0][i]),npoints) for i in range(len(args[0][:-1]))])
    node_list=args[0][-1]
    if showplot: 
        
        ncols= int(round_up(len(node_list)/nrows))
        fig = make_subplots(rows=nrows, cols=ncols,
                            specs= [[{"type": "surface"} for i in range(ncols)] for j in range(nrows)],
                            horizontal_spacing = 0.01, vertical_spacing = 0.01)

    interpolations_list=[]
    row=1
    col=1
    Dict={}
    for elno in range(len(node_list)):
        args_reshaped=[args[0][i] for i in range(len(args[0][:-1]))]
        if method =='nearest': 
            args2interp=[list(zip(*args_reshaped)), node_list[elno]]
            interpolation=LinearNDInterpolator(*args2interp)
        else:
            args_reshaped.append(args[0][-1][elno])
            interpolation=Rbf(*args_reshaped,smooth=smooth,function=method)
        interpolations_list.append(interpolation)
        Dict[z_label[elno]]=interpolation
        if showplot:
            if len(*args)-1==3:
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
                marker_color1=interpolation(*new_coords)
                marker_color2=node_list[elno].ravel()
                z_label_ND=z_label
                label_ND2=color_labels[elno]+'_o'
                label_ND1=color_labels[elno]+'_i'

            elif len(*args)-1==2:
                thisdict = {
                              "x": new_coords[0],
                              "y": new_coords[1],
                              "z": interpolation(*new_coords)}
                thisdict2 = {
                              "x": args_reshaped[0].ravel(),
                              "y": args_reshaped[1].ravel(),
                              "z": node_list[elno].ravel()}
                
                marker_color1='lightskyblue'
                marker_color2='black'
                z_label_ND=z_label[elno]
                label_ND2=z_label[elno]+'_o'
                label_ND1=z_label[elno]+'_i'
            elif len(*args)-1==1:pass
            
                
            plot_3d(thisdict2,row=row,col=col,showplot=False,marker_color=marker_color2,fig=fig,name_label=label_ND2)
            plot_3d(thisdict,showplot=False,row=row,col=col,marker_color=marker_color1,fx=fx,fy=fy,fig=fig,x_label=x_label,y_label=y_label,z_label=z_label_ND,name_label=label_ND1)
            col+=1
            if col >ncols:
                row+=1
                col=1
    if showplot:
        fig.show()
    # return(np.array(interpolations_list),Dict)
    return(Dict)

def plot_3d(args,xerror=None,yerror=None,zerror=None,x_label='x',y_label='y',z_label='z',name_label='var',elno=0,fig=None,color='k',pad=1,w_pad=1,h_pad=1,fx=1000,fy=750,size=3,width=1,row=1,col=1,showplot=False,subplot_titles=['Plot1'],marker_color='black',aspectmode='cube'):
    if showplot: 
        fig = make_subplots(
            rows=1, cols=1,
            specs= [[{"type": "scatter3d"}]],
            # specs=[[{'is_3d': True}]],
            subplot_titles=subplot_titles)

    error_x=dict(type='data', array=xerror,visible=True)
    error_y=dict(type='data', array=yerror,visible=True)
    error_z=dict(type='data', array=zerror,visible=True)
    fig.add_trace(go.Scatter3d(args, error_x=error_x, error_y=error_y, error_z=error_z,
                               mode='markers',
                               marker=dict(size=size,line=dict(width=width),
                               color=marker_color),
                               name=name_label),
                               # text=["t: %.3f"%x for x in marker_color ]),
                               row=row, col=col)

    fig.update_layout(autosize=False,width=fx,height=fy, margin=dict(l=10,r=10,b=10,t=22,pad=4),paper_bgcolor="LightSteelBlue")
    fig.update_scenes(xaxis=dict(title_text=x_label),yaxis=dict(title_text=y_label),zaxis=dict(title_text=z_label),row=row,col=col,aspectmode=aspectmode)
    if showplot: fig.show()
    
def sample_posteriors(interp_star_properties,MCMC_sim_df,ID,mlabel,ndim,show_samples=False,truths=[None,None,None],bins=20,pranges=None,figsize=(10,10)):
    samples=np.array(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'samples'].values[0])
    flat_samples=np.array(MCMC_sim_df.loc[MCMC_sim_df.ID==ID,'flat_samples'].values[0])
    display(MCMC_sim_df.loc[MCMC_sim_df.ID==ID])
    mass,emass_u,emass_d,Av,eAv_u,eAv_d,age,eage_u,eage_d,T,eT_u,eT_d,L,eL_u,eL_d=mcmc_utils.star_properties(flat_samples,ndim,interp_star_properties,mlabel)
   
    txt = "\mathrm{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{{2:.4f}}}"
    txt = txt.format(mass, emass_d, emass_u, 'mass')
    display(Math(txt))

    txt = "\mathrm{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{{2:.4f}}}"
    txt = txt.format(Av, eAv_d, eAv_u, 'Av')
    display(Math(txt))

    txt = "\mathrm{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{{2:.4f}}}"
    txt = txt.format(age, eage_d, eage_u, 'A')
    display(Math(txt))

    txt = "\mathrm{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{{2:.4f}}}"
    txt = txt.format(T, eT_d, eT_u, 'T')
    display(Math(txt))

    txt = "\mathrm{{{3}}} = {0:.4f}_{{-{1:.4f}}}^{{{2:.4f}}}"
    txt = txt.format(L, eL_d, eL_u, 'L')
    display(Math(txt))

    if mlabel =='4': labels=['Teff','Av','Age']
    else: labels=['mass','Av','Age']
    if show_samples:
        fig, axes = plt.subplots(ndim, figsize=(8, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            if truths[i]!= None:ax.axhline(truths[i])
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
    
        axes[-1].set_xlabel("step number")
        plt.tight_layout()
        plt.show()

    fig=plt.figure(figsize=figsize)
    fig.suptitle('Star ID %i'%ID)
    fig = corner.corner(flat_samples,truths=truths,range=pranges,labels=labels,bins=bins,quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 15},plot_contours=True,fig=fig)

    plt.tight_layout()
    plt.show()
    
def sample_blobs(MCMC_df,ID,mag_label_list,color_label_list,show_samples=False,bins=20,pranges=None,figsize=(10,10)):
    display(MCMC_df.loc[MCMC_df.ID==ID])
    blobs=np.array(MCMC_df.loc[MCMC_df.ID==ID,'blobs'].values[0])
    flat_blobs=np.array(MCMC_df.loc[MCMC_df.ID==ID,'flat_blobs'].values[0])
    labels=[]
    truths=[]
    for label in MCMC_df.variables.values[0]:
        labels.append(label)
        if label in mag_label_list: truths.append(MCMC_df.loc[MCMC_df.ID==ID,'mags'].values[0][label==mag_label_list]) 
        elif label in color_label_list: truths.append(MCMC_df.loc[MCMC_df.ID==ID,'cols'].values[0][label==color_label_list]) 
        
    
    ndim=len(truths)
    if show_samples:

        fig, axes = plt.subplots(ndim, figsize=figsize, sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(blobs[:, :, i], "k", alpha=0.3)
            ax.axhline(truths[i])
            ax.set_xlim(0, len(blobs))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        axes[-1].set_xlabel("step number")
        plt.tight_layout()
    
    import corner
    fig=plt.figure(figsize=figsize)
    fig = corner.corner(flat_blobs,truths=truths,range=pranges,labels=labels,bins=bins,quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 15},plot_contours=True,fig=fig)
    
    plt.tight_layout()
    plt.show()
    
