################## DEFAULT SETTING ##################
import warnings, os
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"
import os
from mcmcsedanalysis.mcmc import MCMC,run
from mcmcsedanalysis import mcmc_utils, show_priors
import numpy as np
import argparse

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mcmcsedanalysis.synthetic_photometry import load_spectra_task,plot_SEDfit
from synphot import SourceSpectrum,SpectralElement,Empirical1D
import concurrent.futures
from glob import glob
from itertools import repeat
from astropy.time import Time
import stsynphot as stsyn
import pickle
import ruamel.yaml
yaml = ruamel.yaml.YAML()

pd.set_option('display.max_columns', 500)
def parse():
    # read in command line arguments
    parser = argparse.ArgumentParser(description='MCMC SED fit')
    parser.add_argument('-p', type=str, help='A pipeline config file', default='pipe.yaml', dest='pipe_cfg')
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    parser.add_argument('--make-dir', dest='make_paths', help='Create all needed directories', action='store_true')
    return parser.parse_args()

def pprint_all(self, max_rows=None, max_columns=None, expand_frame=False):
    with pd.option_context(
        'display.max_rows', max_rows if max_rows is not None else None,
        'display.max_columns', max_columns if max_columns is not None else None,
        'display.expand_frame_repr', expand_frame
    ):
        print(self)

# Patch it ONCE for all DataFrames
if not hasattr(pd.DataFrame, 'pprint_all'):
    pd.DataFrame.pprint_all = pprint_all

def load(file):
    if not isinstance(file, str):
        return file

    if file.lower().endswith(('yaml', 'yml')):
        with open(file, 'r') as f:
            ret = yaml.load(f)
        return ret

def assembling_spectra_dataframes(path2accr_spect,path2models,path2models_w_acc,acc_spec_filename = 'accretion_spectrum_2016.fits'):
    ### Accr Spectrum
    spAcc = SourceSpectrum.from_file(path2accr_spect + '/' + acc_spec_filename)

    ### Load Spectra without accretium from file
    file_list = sorted(glob(path2models + "/*.dat"))
    spectrum_list = []
    T_list = []
    logg_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for spectrum, temp, logg in tqdm(executor.map(load_spectra_task, file_list, repeat(False))):
            if temp / 100 % 1 == 0:
                spectrum_list.append(spectrum)
                T_list.append(temp)
                logg_list.append(logg)
    spectrum_without_acc_df = pd.DataFrame({'Teff': T_list, 'logg': logg_list, 'Spectrum': spectrum_list}).set_index(
        ['Teff', 'logg']).sort_values(['Teff', 'logg'])

    ### Load Spectra with accretium from file
    file_list = sorted(glob(path2models_w_acc + "/*.dat"))
    spectrum_list = []
    T_list = []
    logg_list = []
    logAcc_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for spectrum, temp, logg, logacc in tqdm(executor.map(load_spectra_task, file_list, repeat(True))):
            if temp / 100 % 1 == 0:
                spectrum_list.append(spectrum)
                T_list.append(temp)
                logg_list.append(logg)
                logAcc_list.append(logacc)

    spectrum_with_acc_df = pd.DataFrame(
        {'Teff': T_list, 'logg': logg_list, 'logAcc': logAcc_list, 'Spectrum': spectrum_list}).set_index(
        ['Teff', 'logg', 'logAcc']).sort_values(['Teff', 'logg', 'logAcc'])

    ### Vega Spectrum
    vega_spectrum = SourceSpectrum.from_vega()
    # vega_spectrum.plot(left=2000, right=20000, flux_unit='flam', title=vega_spectrum.meta['expr'])
    return(spAcc,spectrum_with_acc_df,spectrum_without_acc_df,vega_spectrum)

def interpolating_isochrones(path2iso, mag_label_list, interpfile="interpolated_isochrones.pck", redo=False, smooth=0.001, method='linear', showplot=False):
    if redo:
        ## Isochrones ((skip if already saved))
        iso_df=pd.read_hdf(path2iso+"bt_settl_AGSS2009_isochrones_new_acc_final.h5",'df')
        iso_df=iso_df.loc[np.isfinite(iso_df.logLacc.values)]

        node_label_list= list(mag_label_list)+['teff','logL','logg','logLacc','logMacc','R']
        node_list=[iso_df[label].values.ravel() for label in node_label_list]
        x=np.log10(iso_df['mass'].values).ravel()
        y=np.log10(iso_df.index.get_level_values('Age').values).ravel()
        z=iso_df.index.get_level_values('logAcc').values.ravel()

        interp_btsettl= mcmc_utils.interpND([x, y, z, node_list], method=method, showplot=showplot, smooth=smooth, z_label=node_label_list, workers=5)

        with open(path2iso+interpfile, 'wb') as file_handle:
            pickle.dump(interp_btsettl , file_handle)
    else:
        ## Load interpolated isochrones
        with open(path2iso + interpfile, 'rb') as file_handle:
            interp_btsettl = pickle.load(file_handle)
    return (interp_btsettl)

def assembling_dictionaries(filter_list,mag_label_list,sat_list, Rv):
    ## Extinction
    Av_dict = mcmc_utils.get_Av_list(filter_list, mag_label_list, verbose=False, Rv=Rv)

    ## Saturation
    sat_dict = dict(zip(mag_label_list, sat_list))

    bp_list = []
    ## BP dictionary
    for filter in filter_list:
        obsdate = Time('2005-01-1').mjd
        if filter.lower() == 'f336w':
            bp = stsyn.band('acs,wfpc2,f336w')
        elif filter.lower() == 'f439w':
            bp = stsyn.band('acs,wfpc2,f439w')
        elif filter.lower() == 'f656n':
            bp = stsyn.band('acs,wfpc2,f656n')
        elif filter.lower() == 'f814w':
            bp = stsyn.band('acs,wfpc2,f814w')

        elif filter.lower() == 'f435w':
            bp = stsyn.band(f'acs,wfc1,f435w,mjd#{obsdate}')
        elif filter.lower() == 'f555w':
            bp = stsyn.band(f'acs,wfc1,f555w,mjd#{obsdate}')
        elif filter.lower() == 'f658n':
            bp = stsyn.band(f'acs,wfc1,f658n,mjd#{obsdate}')
        elif filter.lower() == 'f775w':
            bp = stsyn.band(f'acs,wfc1,f775w,mjd#{obsdate}')
        elif filter.lower() == 'f850lp':
            bp = stsyn.band(f'acs,wfc1,f850lp,mjd#{obsdate}')

        elif filter.lower() == 'f110w':
            bp = stsyn.band('nicmos,3,f110w')
        elif filter.lower() == 'f160w':
            bp = stsyn.band('nicmos,3,f160w')

        elif filter.lower() == 'f130n':
            bp = stsyn.band('wfc3,ir,f130n')
        elif filter.lower() == 'f139m':
            bp = stsyn.band('wfc3,ir,f139m')

        bp_list.append(bp)

    # bp_list = [bp336, bp439, bp656, bp814, bp435, bp555, bp658, bp775, bp850, bp110, bp160, bp130, bp139]

    bp_flattened_list = []
    for bp in bp_list:
        bp_flattened = SpectralElement(Empirical1D, points=bp.waveset,
                                       lookup_table=bp(bp.waveset))
        bp_flattened_list.append(bp_flattened)
    bp_dict = {'%s' % mag_label_list[elno]: bp_flattened_list[elno] for elno in range(len(mag_label_list))}
    return(Av_dict,sat_dict,bp_dict)

def assembling_priors(path2data,showplot=False):
    Besançon_plus_GAIA_ONCFOV_df = pd.read_csv(path2data + '/Besançon_plus_GAIA_ONCFOV.csv')
    parallax_KDE0 = show_priors.kernel_prior(Besançon_plus_GAIA_ONCFOV_df.parallax, xlabel='parallax', nbins=50,
                                             bw_method=0.1, bandwidth2fit=np.linspace(0.01, 0.1, 1000),
                                             xlogscale=False, showplot=showplot,
                                             density=True, kernel='gaussian', return_prior=True)

    Av_df = pd.read_csv(path2data + '/Av.csv')
    Av_KDE0 = show_priors.kernel_prior(Av_df.Av, xlabel='Av', nbins=10, bw_method=0.2,
                                       bandwidth2fit=np.linspace(0.1, 3, 1000), xlogscale=False, density=True,
                                       kernel='gaussian', return_prior=True, showplot=showplot, )

    Age_df = pd.read_csv(path2data + '/Age.csv')
    Age_KDE0 = show_priors.kernel_prior(Age_df.Age, xlabel='A', nbins=20, bw_method=0.3,
                                        bandwidth2fit=np.linspace(0.1, 3, 1000), xlogscale=False, density=True,
                                        kernel='gaussian', return_prior=True, showplot=showplot, )
    Mass_df = pd.read_csv(path2data + '/Mass.csv')

    mass_KDE0 = show_priors.kernel_prior(Mass_df.Mass, xlabel='mass', nbins=10, bw_method=0.08,
                                         bandwidth2fit=np.linspace(0.1, 1, 1000), xlogscale=False, density=True,
                                         kernel='gaussian', return_prior=True, showplot=showplot, )
    return(parallax_KDE0, Av_KDE0, Age_df, Age_KDE0, mass_KDE0)

if __name__ == '__main__':
    args = parse()
    config = load(args.pipe_cfg)
    path2data = config['paths']['data']
    path2out = config['paths']['out']
    path2priors = config['paths']['priors']
    path2accr_spect = config['paths']['accr_spect']
    path2models = config['paths']['models']
    path2models_w_acc = config['paths']['models_w_acc']
    path2iso = config['paths']['iso']
    catalogue = config['catalogue']['name']

    if args.make_paths:
        ################################################################################################################
        # Making output dirs                                                                                            #
        ################################################################################################################
        os.makedirs(path2out+'/samplers', exist_ok=True)
        os.makedirs(path2out+'/corners', exist_ok=True)
        os.makedirs(path2out+'/fits', exist_ok=True)

    ################################################################################################################
    # Importing Catalog                                                                                            #
    ################################################################################################################
    input_df = pd.read_csv(path2data + catalogue)
    ################################################################################################################
    # Setting the stage for the MCMC run                                                                           #
    ################################################################################################################
    # Selecting the filters and saturation limits and magnitudes to work with from the catalog
    filter_list = config['filter_list']
    sat_list = config['sat_list']
    Rv = config['Rv']
    ID_label = config['ID_label']
    mag_label_list =  config['mag_list']
    emag_label_list =  config['emag_list']
    interpfile = config['interpfile']

    #Interpolating Isochrones on the selected magnitudes (or load an existing interpolated isochrone)
    interp_btsettl = interpolating_isochrones(path2iso,mag_label_list,interpfile)
    # Creating a filter dependent dictionary for the extinction, saturation and bandpass
    Av_dict, sat_dict, bp_dict = assembling_dictionaries(filter_list,mag_label_list,sat_list,Rv)

    # Loading KDE for priors to use as general priors in case specific prior for the target is missing.
    # Note: If you have the prior for Teff, it will skip the KDE prior on the mass.
    parallax_KDE0, Av_KDE0, Age_df, Age_KDE0, mass_KDE0 = assembling_priors(path2priors)

    ID_list = config['ID_list']
    #To add prior knowledge for the star's parameters we need to add the following entry to the catalog.
    # If any of these values are nans, or absent, then the pipeline will use the KDEs if provided.
    # If they are None, then no prior will be used for that parameter.
    input_df.loc[input_df[ID_label].isin(ID_list), 'Parallax'] = 1.514106
    input_df.loc[input_df[ID_label].isin(ID_list), 'eParallax'] = 0.13
    input_df.loc[input_df[ID_label].isin(ID_list), 'Teff'] = 3402
    input_df.loc[input_df[ID_label].isin(ID_list), 'eTeff'] = 150
    input_df.loc[input_df[ID_label].isin(ID_list), 'Av'] = 3.5
    input_df.loc[input_df[ID_label].isin(ID_list), 'eAv'] = 0.5
    input_df.loc[input_df[ID_label].isin(ID_list), 'Age'] = 1.2
    input_df.loc[input_df[ID_label].isin(ID_list), 'eAge'] = 0.5
    input_df.loc[input_df[ID_label].isin(ID_list), 'SpAcc'] = 0.18
    input_df.loc[input_df[ID_label].isin(ID_list), 'eSpAcc'] = 0.1

    ################################################################################################################
    # This is the start of the MCMC run                                                                            #
    ################################################################################################################
    print(input_df.loc[input_df[ID_label].isin(ID_list)].pprint_all())

    mcmc=MCMC(interp_btsettl,
              mag_label_list,
              sat_dict,
              Av_dict,
              spaccfit=config['MCMC']['spaccfit'],
              emag_label_list= emag_label_list,
              ID_label=ID_label,
              Teff_prior=config['MCMC']['Teff_prior'],
              eTeff_prior=config['MCMC']['eTeff_prior'],
              parallax_prior=config['MCMC']['parallax_prior'],
              eparallax_prior=config['MCMC']['eparallax_prior'],
              Av_prior=config['MCMC']['Av_prior'],
              eAv_prior=config['MCMC']['eAv_prior'],
              Age_prior=config['MCMC']['Age_prior'],
              eAge_prior=config['MCMC']['eAge_prior'],
              SpAcc_prior=config['MCMC']['SpAcc_prior'],
              eSpAcc_prior=config['MCMC']['eSpAcc_prior'],
              workers=config['MCMC']['workers'],
              conv_thr=config['MCMC']['conv_thr'],
              ndesired=config['MCMC']['ndesired'],
              err_max=config['MCMC']['err_max'],
              err_min=config['MCMC']['err_min'],
              logMass_range=config['MCMC']['logMass_range'],
              logAge_range=config['MCMC']['logAge_range'],
              logSPacc_range=config['MCMC']['logSPacc_range'],
              logAv_range=config['MCMC']['logAv_range'],
              Parallax_range=config['MCMC']['parallax_range'],
              nwalkers_ndim_niters=config['MCMC']['nwalkers_ndim_niters'],
              parallelize_sampler=config['MCMC']['parallelize_sampler'],
              show_test=config['MCMC']['show_test'],
              progress=config['MCMC']['progress'],
              blobs=config['MCMC']['blobs'],
              parallax_KDE=parallax_KDE0,
              Av_KDE=Av_KDE0,
              Age_KDE=Age_KDE0,
              mass_KDE=mass_KDE0,
              path2data=path2data,
              savedir=path2out+'/samplers') # ----> This will setup the MCMC. There are many options hidden in there!

    run(mcmc, input_df, ID_list,forced=True) # ---> This will run the MCMC and save the sampler

    ################################################################################################################
    # This part is very specific for my ONC Work.                                                                  #
    # I used these routines to update the DF with the generated new values and draw the summary plots.             #
    # You can always read the sampler in the samplers dir and sample the posterior and generate your own plots     #
    # skipping entirely this section.                                                                              #
    ################################################################################################################

    # Loading the posterior distributions saved in the sampler
    file_list=[]
    for ID in tqdm(ID_list):
        file_list.append(path2out + f"/samplers/samplerID_{ID}")

    ################################################################################################################
    # I use these values to sample the sampler, plot corner plots and trace plots,                                 #
    # and evaluate if the distance for my source is compatible with Orion                                          #
    ################################################################################################################
    pmean = config['MCMC']['pmean']
    pM = config['MCMC']['pM']
    pm = config['MCMC']['pm']

    if config['MCMC']['spaccfit']:
        label_list = ['logMass', 'logAv', 'logAge', 'logSPacc', 'Parallax']
    else:
        label_list = ['logMass', 'logAv', 'logAge', 'Parallax']

    # Sampling posteriors
    input_df = mcmc_utils.update_dataframe(input_df, file_list, interp_btsettl, kde_fit=True, ID_label=ID_label,
                                           pmin=pmean - pm * 3, pmax=pmean + pM * 3, path2loaddir=path2out+'/samplers',
                                           path2savedir=path2out+'/corners', parallel_runs=False, verbose=True,
                                           label_list=label_list,spaccfit=config['MCMC']['spaccfit'])

    ################################################################################################################
    # Saving summary plot                                                                                          #
    ################################################################################################################

    # Assembling Spectra dataframes for final plots
    spAcc, spectrum_with_acc_df, spectrum_without_acc_df, vega_spectrum = assembling_spectra_dataframes(path2accr_spect, path2models, path2models_w_acc)
    for ID in tqdm(ID_list):
        file = path2out+'/corners/cornerID%i.png' % int(ID)
        img = mpimg.imread(file)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig, ax[0], Ndict = plot_SEDfit(spAcc, spectrum_without_acc_df, vega_spectrum, bp_dict,
                                        sat_dict, interp_btsettl, input_df.loc[input_df[ID_label] == ID],
                                        mag_label_list, Rv=Rv, ms=2, showplot=False, fig=fig, ax=ax[0],
                                        spaccfit = config['MCMC']['spaccfit'])
        ax[1].imshow(img)
        ax[1].axis('off')
        fig.suptitle(int(ID))
        plt.tight_layout()
        plt.savefig(path2out+'/fits/ID%i.png' % int(ID))
        plt.close()

    pass

