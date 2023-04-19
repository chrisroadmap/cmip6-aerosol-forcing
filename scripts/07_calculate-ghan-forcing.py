#!/usr/bin/env python
# coding: utf-8

"""Calculate Ghan breakdown."""

import glob
import os
import warnings

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import iris
import iris.coord_categorisation
import iris.analysis.cartography
from iris.util import equalise_attributes, unify_time_units

warnings.simplefilter('ignore')

print("Calculating Ghan breakdown...")

datadir = '../data/esgf_ghan2013/'

varlist = [
    "rsdt",
    "rsut",
    "rsutcs",
    "rlut",
    "rlutcs",
    "rsutaf",
    "rsutcsaf",
    "rlutaf",
    "rlutcsaf"
]

pert = {}
base = {}
for var in varlist:
    cubes = iris.load(f"{datadir}/{var}_A*mon_UKESM1-0-LL_histSST_r1i1p1f2_*.nc")
    equalise_attributes(cubes)
    unify_time_units(cubes)
    cubes = cubes.concatenate_cube()
    pert[var] = cubes

    cubes = iris.load(f"{datadir}/{var}_A*mon_UKESM1-0-LL_histSST-piAer_r1i1p1f2_*.nc")
    equalise_attributes(cubes)
    unify_time_units(cubes)
    cubes = cubes.concatenate_cube()
    base[var] = cubes

results = {}
results['ERF'] = (
    (pert['rsdt'] - pert['rsut'] - pert['rlut']) - (base['rsdt'] - base['rsut'] - base['rlut'])
)

results['ERFari'] = (
    (
        (pert['rsdt'] - pert['rsut'] - pert['rlut']) - (pert['rsdt'] - pert['rsutaf'] - pert['rlutaf'])
    ) - (
        (base['rsdt'] - base['rsut'] - base['rlut']) - (base['rsdt'] - base['rsutaf'] - base['rlutaf'])
    )
)

results['ERFaci'] = (
    (
        (pert['rsdt'] - pert['rsutaf'] - pert['rlutaf']) - (pert['rsdt'] - pert['rsutcsaf'] - pert['rlutcsaf'])
    ) - (
        (base['rsdt'] - base['rsutaf'] - base['rlutaf']) - (base['rsdt'] - base['rsutcsaf'] - base['rlutcsaf'])
    )
)

results['albedo'] = (
    (
        (pert['rsdt'] - pert['rsutcsaf'] - pert['rlutcsaf'])
    ) - (
        (base['rsdt'] - base['rsutcsaf'] - base['rlutcsaf'])
    )
)

data = {}
for var in ['ERF', 'ERFari', 'ERFaci', 'albedo']:
    iris.coord_categorisation.add_year(results[var], 'time')
    cube_year = results[var].aggregated_by('year', iris.analysis.MEAN)
    if not cube_year.coord('latitude').has_bounds():
        cube_year.coord('latitude').guess_bounds()
    if not cube_year.coord('longitude').has_bounds():
        cube_year.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube_year)
    cube_gmym = cube_year.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

    data[var] = cube_gmym.data
    time = cube_gmym.coord('year').points

os.makedirs(f'../output/UKESM1-0-LL/r1i1p1f2/ghan2013/', exist_ok=True)
pd.DataFrame(data, index=time).to_csv(f'../output/UKESM1-0-LL/r1i1p1f2/ghan2013/UKESM1-0-LL_r1i1p1f2_aerosol_forcing.csv')
