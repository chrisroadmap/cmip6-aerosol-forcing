#!/usr/bin/env python
# coding: utf-8

"""Get ERFari and ERFaci from climate model output."""

import glob
import os
import warnings

from climateforcing.aprp import aprp
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import iris
import iris.coord_categorisation
import iris.analysis.cartography
from iris.util import equalise_attributes, unify_time_units
from tqdm.auto import tqdm

warnings.simplefilter('ignore')

print("Calculating APRP breakdown...")

datadir = '../data/yearly/'

models = []
for filepath in glob.glob(datadir + '*'):
    models.append(filepath.split('/')[-1])

runs_piclim_control = {}
runs_piclim_histaer = {}
runs_histsst_piaer = {}
runs_histsst = {}
for model in models:
    runs_piclim_control[model] = {}
    runs_piclim_histaer[model] = {}
    runs_histsst_piaer[model] = {}
    runs_histsst[model] = {}
    for filepath in glob.glob(f'{datadir}{model}/*/*'):
        expt = filepath.split('/')[-1]
        runid = filepath.split('/')[-2]
        if expt=='piClim-control':
            runs_piclim_control[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:-3]))
            runs_piclim_control[model][runid]['first'] = min(years)
            runs_piclim_control[model][runid]['last'] = max(years)
        elif expt=='piClim-histaer':
            runs_piclim_histaer[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:-3]))
            runs_piclim_histaer[model][runid]['first'] = min(years)
            runs_piclim_histaer[model][runid]['last'] = max(years)
        elif expt=='histSST-piAer':
            runs_histsst_piaer[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:-3]))
            runs_histsst_piaer[model][runid]['first'] = min(years)
            runs_histsst_piaer[model][runid]['last'] = max(years)
        elif expt=='histSST':
            runs_histsst[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:-3]))
            runs_histsst[model][runid]['first'] = min(years)
            runs_histsst[model][runid]['last'] = max(years)

# from email correspondence with Ron Miller 19 May 2020: use r?i1p1f2 from GISS.
# from email correspondence with Dirk Olivie 19 October 2020: use r?i1p2f1 from NorESM.

varlist = [
    "rsdt",
    "rsus",
    "rsds",
    "clt",
    "rsdscs",
    "rsuscs",
    "rsut",
    "rsutcs",
    "rlut",
    "rlutcs",
]

longnames = {
    'rsdt': 'toa_incoming_shortwave_flux',
    'rlut': 'toa_outgoing_longwave_flux',
    'rsut': 'toa_outgoing_shortwave_flux',
    'rlutcs': 'toa_outgoing_longwave_flux_assuming_clear_sky',
    'rsds': 'surface_downwelling_shortwave_flux_in_air',
    'rsus': 'surface_upwelling_shortwave_flux_in_air',
    'rsutcs': 'toa_outgoing_shortwave_flux_assuming_clear_sky',
    'clt': 'cloud_area_fraction',
    'rsdscs': 'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky',
    'rsuscs': 'surface_upwelling_shortwave_flux_in_air_assuming_clear_sky'
}

component_longnames = {
    'ERF' : 'Effective radiative forcing',
    'ERFariSW': 'Shortwave effective radiative forcing due to aerosol-radiation interactions',
    'ERFaciSW': 'Shortwave effective radiative forcing due to aerosol-cloud interactions',
    'albedo'  : 'Shortwave effective radiative forcing due to surface albedo',
    'ERFariLW': 'Longwave effective radiative forcing due to aerosol-radiation interactions',
    'ERFaciLW': 'Longwave effective radiative forcing due to aerosol-cloud interactions'
}

outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']

# move away from dicts because of mutability issues - even though this is terrible
# coding
def rfmip():
    base_run = list(runs_piclim_control[model])[0]
    first_base = runs_piclim_control[model][base_run]['first']
    last_base = runs_piclim_control[model][base_run]['last']
    # calculate aprp for each ensemble member
    for pert_run in tqdm(runs_piclim_histaer[model], desc='run', leave=False):
        outdir = f'../output/{model}/{pert_run}'
        os.makedirs(f'{outdir}/gridded', exist_ok=True)
        os.makedirs(f'{outdir}/mean', exist_ok=True)
        first_pert = runs_piclim_histaer[model][pert_run]['first']
        last_pert = runs_piclim_histaer[model][pert_run]['last']
        for pert_year in tqdm(range(first_pert, last_pert+1), desc='Model years', leave=False):
            clt_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/clt_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsdt_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsdt_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsus_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsus_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsds_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsds_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsdscs_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsdscs_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsut_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsut_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsutcs_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsutcs_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rlut_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rlut_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rlutcs_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rlutcs_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")
            rsuscs_pert = iris.load_cube(f"{datadir}{model}/{pert_run}/piClim-histaer/rsuscs_Amon_{model}_piClim-histaer_{pert_run}_{pert_year}.nc")

            nlat = rsdt_pert.shape[1]
            nlon = rsdt_pert.shape[2]

            results = {}
            for var in outvars:
                results[var] = np.ones((12, nlat, nlon)) * np.nan
            results['ERF'] = np.ones((12, nlat, nlon)) * np.nan

            # Every scenario year is compared to every piControl year in turn.
            interim = {}
            for var in outvars:
                interim[var] = np.ones((last_base-first_base+1, 12, nlat, nlon)) * np.nan
            interim['ERF'] = np.ones((last_base-first_base+1, 12, nlat, nlon)) * np.nan

            for j, base_year in tqdm(enumerate(range(first_base, last_base+1), desc='Base years')):
                clt_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/clt_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsdt_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsdt_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsus_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsus_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsds_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsds_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsdscs_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsdscs_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsut_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsut_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsutcs_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsutcs_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rlut_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rlut_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rlutcs_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rlutcs_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")
                rsuscs_base = iris.load_cube(f"{datadir}{model}/{base_run}/piClim-control/rsuscs_Amon_{model}_piClim-control_{base_run}_{base_year}.nc")

                base_slice = {
                    "clt": clt_base.data,
                    "rsdt": rsdt_base.data,
                    "rsus": rsus_base.data,
                    "rsds": rsds_base.data,
                    "rsdscs": rsdscs_base.data,
                    "rsut": rsut_base.data,
                    "rsutcs": rsutcs_base.data,
                    "rlut": rlut_base.data,
                    "rlutcs": rlutcs_base.data,
                    "rsuscs": rsuscs_base.data,
                }
                pert_slice = {
                    "clt": clt_pert.data,
                    "rsdt": rsdt_pert.data,
                    "rsus": rsus_pert.data,
                    "rsds": rsds_pert.data,
                    "rsdscs": rsdscs_pert.data,
                    "rsut": rsut_pert.data,
                    "rsutcs": rsutcs_pert.data,
                    "rlut": rlut_pert.data,
                    "rlutcs": rlutcs_pert.data,
                    "rsuscs": rsuscs_pert.data,
                }
                interim['ERF'][j, ...] = (
                    (
                        pert_slice['rsdt'] - pert_slice['rsut'] - pert_slice['rlut']
                    ) - (
                        base_slice['rsdt'] - base_slice['rsut'] - base_slice['rlut']
                    )
                )
                aprp_output = aprp(base_slice, pert_slice, longwave=True)
                for var in outvars:
                    interim[var][j, ...] = aprp_output[var]
            for var in outvars + ['ERF']:
                results[var] = np.mean(interim[var], axis=0)

            for component in results:
                cube = iris.cube.Cube(
                    results[component],
                    var_name = component,
                    long_name = component_longnames[component],
                    units = 'W m-2',
                    dim_coords_and_dims=[(rsdt_pert.coord('time'), 0), (rsdt_pert.coord('latitude'), 1), (rsdt_pert.coord('longitude'), 2)]
                )
                iris.save(cube, f'{outdir}/gridded/{component}_{pert_year}.nc')


def aerchemmip():
    run = list(runs_histsst_piaer[model])[0]
    first = runs_histsst_piaer[model][run]['first']
    last = runs_histsst_piaer[model][run]['last']
    outdir = f'../output/{model}/{run}'
    os.makedirs(f'{outdir}/gridded', exist_ok=True)
    os.makedirs(f'{outdir}/mean', exist_ok=True)
    for year in tqdm(range(first, last+1), desc='Model years', leave=False):
        clt_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/clt_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rsdt_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsdt_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rsus_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsus_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rsds_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsds_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rsdscs_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsdscs_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rsut_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsut_Amon_{model}_histSST-piAer_{run}_{year}.nc")    
        rsutcs_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsutcs_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rlut_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rlut_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rlutcs_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rlutcs_Amon_{model}_histSST-piAer_{run}_{year}.nc")
        rsuscs_base = iris.load_cube(f"{datadir}/{model}/{run}/histSST-piAer/rsuscs_Amon_{model}_histSST-piAer_{run}_{year}.nc")

        clt_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/clt_Amon_{model}_histSST_{run}_{year}.nc")
        rsdt_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsdt_Amon_{model}_histSST_{run}_{year}.nc")
        rsus_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsus_Amon_{model}_histSST_{run}_{year}.nc")
        rsds_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsds_Amon_{model}_histSST_{run}_{year}.nc")
        rsdscs_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsdscs_Amon_{model}_histSST_{run}_{year}.nc")
        rsut_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsut_Amon_{model}_histSST_{run}_{year}.nc")    
        rsutcs_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsutcs_Amon_{model}_histSST_{run}_{year}.nc")
        rlut_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rlut_Amon_{model}_histSST_{run}_{year}.nc")
        rlutcs_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rlutcs_Amon_{model}_histSST_{run}_{year}.nc")
        rsuscs_pert = iris.load_cube(f"{datadir}/{model}/{run}/histSST/rsuscs_Amon_{model}_histSST_{run}_{year}.nc")

        nlat = rsdt_base.shape[1]
        nlon = rsdt_base.shape[2]

        outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']
        results = {}

        base_slice = {
            "clt": clt_base.data,
            "rsdt": rsdt_base.data,
            "rsus": rsus_base.data,
            "rsds": rsds_base.data,
            "rsdscs": rsdscs_base.data,
            "rsut": rsut_base.data,
            "rsutcs": rsutcs_base.data,
            "rlut": rlut_base.data,
            "rlutcs": rlutcs_base.data,
            "rsuscs": rsuscs_base.data,
        }
        pert_slice = {
            "clt": clt_pert.data,
            "rsdt": rsdt_pert.data,
            "rsus": rsus_pert.data,
            "rsds": rsds_pert.data,
            "rsdscs": rsdscs_pert.data,
            "rsut": rsut_pert.data,
            "rsutcs": rsutcs_pert.data,
            "rlut": rlut_pert.data,
            "rlutcs": rlutcs_pert.data,
            "rsuscs": rsuscs_pert.data,
        }

        results['ERF'] = (
            (
                pert_slice['rsdt'] - pert_slice['rsut'] - pert_slice['rlut']
            ) - (
                base_slice['rsdt'] - base_slice['rsut'] - base_slice['rlut']
            )
        )

        aprp_output = aprp(base_slice, pert_slice, longwave=True)
        for var in outvars:
            results[var] = aprp_output[var]

        for component in results:
            cube = iris.cube.Cube(
                results[component],
                var_name = component,
                long_name = component_longnames[component],
                units = 'W m-2',
                dim_coords_and_dims=[(rsdt_pert.coord('time'), 0), (rsdt_pert.coord('latitude'), 1), (rsdt_pert.coord('longitude'), 2)]
            )
            iris.save(cube, f"{outdir}/gridded/{component}_{year}.nc")


for model in tqdm(models, desc='Models'):
    if model in ['CanESM5', 'GFDL-CM4', 'IPSL-CM6A-LR', 'MPI-ESM-1-2-HAM', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'MIROC6', 'MRI-ESM2-0']:
        continue
    if len(runs_piclim_control[model])>0:
        rfmip()
    else:
        aerchemmip()
