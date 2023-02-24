#!/usr/bin/env bash

# This script deletes the original downloaded CMIP6 data. It is not required, but will
# free up disk space as exactly the same data is output in single yearly files.

# Caution! ensure that your yearly data looks good first! It would be a good idea to
# make execution of this script conditional on exit code zero from script 2.

rm -rf ../data/esgf/*.nc
