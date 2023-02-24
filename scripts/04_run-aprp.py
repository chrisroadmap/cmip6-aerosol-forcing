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
                years.append(int(files.split('_')[-1][:4]))
            runs_piclim_control[model][runid]['first'] = min(years)
            runs_piclim_control[model][runid]['last'] = max(years)
        elif expt=='piClim-histaer':
            runs_piclim_histaer[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:4]))
            runs_piclim_histaer[model][runid]['first'] = min(years)
            runs_piclim_histaer[model][runid]['last'] = max(years)
        elif expt=='histSST-piAer':
            runs_histsst_piaer[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:4]))
            runs_histsst_piaer[model][runid]['first'] = min(years)
            runs_histsst_piaer[model][runid]['last'] = max(years)
        elif expt=='histSST':
            runs_histsst[model][runid] = {}
            years = []
            for files in glob.glob(f'{datadir}{model}/{runid}/{expt}/*.nc'):
                years.append(int(files.split('_')[-1][:4]))
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

            for j, base_year in enumerate(tqdm(range(first_base, last_base+1), desc='Base years')):
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

                # iris.coord_categorisation.add_year(cube, 'time')
                # cube_year = cube.aggregated_by('year', iris.analysis.MEAN)
                # if not cube_year.coord('latitude').has_bounds():
                #     cube_year.coord('latitude').guess_bounds()
                # if not cube_year.coord('longitude').has_bounds():
                #     cube_year.coord('longitude').guess_bounds()
                # grid_areas = iris.analysis.cartography.area_weights(cube_year)
                # cube_gmym = cube_year.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
                # iris.save(cube_gmym, f"{outdir}/mean/{component}.nc")



def aerchemmip():
    run = runs_histsst_piaer[model][0]
    clt_base = iris.load(f"{datadir}/clt_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(clt_base)
    unify_time_units(clt_base)
    clt_base = clt_base.concatenate_cube()
    rsdt_base = iris.load(f"{datadir}/rsdt_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsdt_base)
    unify_time_units(rsdt_base)
    rsdt_base = rsdt_base.concatenate_cube()
    rsus_base = iris.load(f"{datadir}/rsus_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsus_base)
    unify_time_units(rsus_base)
    rsus_base = rsus_base.concatenate_cube()
    rsds_base = iris.load(f"{datadir}/rsds_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsds_base)
    unify_time_units(rsds_base)
    rsds_base = rsds_base.concatenate_cube()
    rsdscs_base = iris.load(f"{datadir}/rsdscs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsdscs_base)
    unify_time_units(rsdscs_base)
    rsdscs_base = rsdscs_base.concatenate_cube()
    rsut_base = iris.load(f"{datadir}/rsut_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsut_base)
    unify_time_units(rsut_base)
    rsut_base = rsut_base.concatenate_cube()
    rsutcs_base = iris.load(f"{datadir}/rsutcs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsutcs_base)
    unify_time_units(rsutcs_base)
    rsutcs_base = rsutcs_base.concatenate_cube()
    rlut_base = iris.load(f"{datadir}/rlut_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rlut_base)
    unify_time_units(rlut_base)
    rlut_base = rlut_base.concatenate_cube()
    rlutcs_base = iris.load(f"{datadir}/rlutcs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rlutcs_base)
    unify_time_units(rlutcs_base)
    rlutcs_base = rlutcs_base.concatenate_cube()
    rsuscs_base = iris.load(f"{datadir}/rsuscs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsuscs_base)
    unify_time_units(rsuscs_base)
    rsuscs_base = rsuscs_base.concatenate_cube()

    # calculate aprp for each ensemble member
    for run in tqdm(runs_histsst[model], desc='run', leave=False):
        outdir = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/aerosol/{model}/{run}'
        os.makedirs(outdir, exist_ok=True)
        clt_pert = iris.load(f"{datadir}/clt_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(clt_pert)
        unify_time_units(clt_pert)
        clt_pert = clt_pert.concatenate_cube()
        rsdt_pert = iris.load(f"{datadir}/rsdt_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsdt_pert)
        unify_time_units(rsdt_pert)
        rsdt_pert = rsdt_pert.concatenate_cube()
        rsus_pert = iris.load(f"{datadir}/rsus_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsus_pert)
        unify_time_units(rsus_pert)
        rsus_pert = rsus_pert.concatenate_cube()
        rsds_pert = iris.load(f"{datadir}/rsds_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsds_pert)
        unify_time_units(rsds_pert)
        rsds_pert = rsds_pert.concatenate_cube()
        rsdscs_pert = iris.load(f"{datadir}/rsdscs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsdscs_pert)
        unify_time_units(rsdscs_pert)
        rsdscs_pert = rsdscs_pert.concatenate_cube()
        rsut_pert = iris.load(f"{datadir}/rsut_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsut_pert)
        unify_time_units(rsut_pert)
        rsut_pert = rsut_pert.concatenate_cube()
        rsutcs_pert = iris.load(f"{datadir}/rsutcs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsutcs_pert)
        unify_time_units(rsutcs_pert)
        rsutcs_pert = rsutcs_pert.concatenate_cube()
        rlut_pert = iris.load(f"{datadir}/rlut_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rlut_pert)
        unify_time_units(rlut_pert)
        rlut_pert = rlut_pert.concatenate_cube()
        rlutcs_pert = iris.load(f"{datadir}/rlutcs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rlutcs_pert)
        unify_time_units(rlutcs_pert)
        rlutcs_pert = rlutcs_pert.concatenate_cube()
        rsuscs_pert = iris.load(f"{datadir}/rsuscs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsuscs_pert)
        unify_time_units(rsuscs_pert)
        rsuscs_pert = rsuscs_pert.concatenate_cube()

        pert_nmonths = rsdt_pert.shape[0]
        pert_nyears = pert_nmonths//12
        nlat = rsdt_base.shape[1]
        nlon = rsdt_base.shape[2]

        outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']
        results = {}

        for var in outvars:
            results[var] = np.ones((pert_nmonths, nlat, nlon)) * np.nan

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

            iris.coord_categorisation.add_year(cube, 'time')
            cube_year = cube.aggregated_by('year', iris.analysis.MEAN)
            if not cube_year.coord('latitude').has_bounds():
                cube_year.coord('latitude').guess_bounds()
            if not cube_year.coord('longitude').has_bounds():
                cube_year.coord('longitude').guess_bounds()
            grid_areas = iris.analysis.cartography.area_weights(cube_year)
            cube_gmym = cube_year.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
            iris.save(cube_gmym, f"{outdir}/{component}.nc")


for model in tqdm(["GISS-E2-1-G"], desc='Models'):
    # what to do about ec-earth, which is a huge model?
    # GFDL-ESM4 seemed to struggle
    # I think MRI-ESM would too
    if len(runs_piclim_control[model])>0:
        rfmip()
    else:
        aerchemmip()
