#!/usr/bin/env bash

# This script downloads CMIP6 model data from ESGF.

# Note: sometimes, things fail silently. CCCma for instance seems to only
# allow a few file downloads per access. You may need to run the script
# a few times to allow for this.

script_dir=$PWD

cd ../data/esgf
for file in *.sh
do
    bash $file -s
done

cd ${script_dir}
