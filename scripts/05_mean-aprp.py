#!/usr/bin/env python
# coding: utf-8

"""Make global mean time series."""

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

vars = ['ERF', 'ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']

for outdir in glob.glob('../output/*/*'):
    model = outdir.split('/')[-2]
    run = outdir.split('/')[-1]
    print(model, run)

    data = {}
    for var in vars:
        cubes = iris.load(f'../output/{model}/{run}/gridded/{var}_*.nc')
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cubes = cubes.concatenate_cube()

        iris.coord_categorisation.add_year(cubes, 'time')
        cube_year = cubes.aggregated_by('year', iris.analysis.MEAN)
        if not cube_year.coord('latitude').has_bounds():
            cube_year.coord('latitude').guess_bounds()
        if not cube_year.coord('longitude').has_bounds():
            cube_year.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube_year)
        cube_gmym = cube_year.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

        data[var] = cube_gmym.data
        time = cube_gmym.coord('year').points

    data['ERFari'] = data['ERFariSW'] + data['ERFariLW']
    data['ERFaci'] = data['ERFaciSW'] + data['ERFaciLW']
    data['residual'] = data['ERF'] - data['ERFari'] - data['ERFaci'] - data['albedo']
    pd.DataFrame(data, index=time).to_csv(f'../output/{model}/{run}/mean/{model}_{run}_aerosol_forcing.csv')
