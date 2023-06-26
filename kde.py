#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 14:06:30 2022

@author: giovanni
"""
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np

class KDE():
    def __init__(self,x,x_grid,bandwidth=None,bandwidth2fit=np.linspace(0.1, 1, 100),cv=20,n_jobs=10,kernel='gaussian'):
        self.x=x
        self.x_grid=x_grid
        self.bandwidth=bandwidth
        self.kernel=kernel
        self.bandwidth2fit=bandwidth2fit
        self.cv=cv
        self.n_jobs=n_jobs
        
    def kde_sklearn(self):
        """Kernel Density Estimation with Scikit-learn"""
        if self.bandwidth==None:
            grid=self.fine_tune_bw()
            self.bandwidth=grid['bandwidth']
        kde_skl = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel)
        self.kde_skl=kde_skl.fit(self.x[:, np.newaxis])
    
    def pdf(self,x):
        if not isinstance(x,(list,np.ndarray)):x=np.array([x])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = self.kde_skl.score_samples(x[:, np.newaxis])
        return(np.exp(log_pdf))
        
    def fine_tune_bw(self):
        grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': self.bandwidth2fit},cv=self.cv,n_jobs=self.n_jobs) # 20-fold cross-validation
        grid.fit(self.x[:, None])
        return(grid.best_params_)