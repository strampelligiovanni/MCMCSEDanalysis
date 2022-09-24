# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 17:42:02 2022

@author: stram
"""

import os,sys,glob
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('../../../../../Projects/MCMC_analysis/')
from config import path2projects,path2data
sys.path.append(path2projects)
sys.path.append(path2projects+'/MCMC_analysis')
import mcmc_utils
import pandas as pd
import pickle
pd.set_option('display.max_columns', 50)
pmean=2.4957678361522033 
pM=0.31352512471278526 
pm=0.31352512471278526

with open(path2data+"/Giovanni/Synthetic Photometry/interpolated_isochrones.pck", 'rb') as file_handle:
        interp_btsettl = pickle.load(file_handle)


ONC_combined_df=pd.read_hdf(path2data+'/Giovanni/MCMC_analysis/ONC_combined_final_df.hdf','df')
file_list=sorted(glob.glob(path2data+"/Giovanni/MCMC_analysis/samplers/*"))[1:]
path2savedir=path2data+'/Giovanni/MCMC_analysis/SED fitting'
ONC_combined_df=mcmc_utils.update_dataframe(ONC_combined_df,file_list,interp_btsettl,kde_fit=True,pmin=pmean-pm*3,pmax=pmean+pM*3,path2savedir=path2savedir)
ONC_combined_df.to_hdf(path2data+'/Giovanni/MCMC_analysis/ONC_combined_final_df.hdf','df')