#!/usr/bin/env bash

# Create conda environment
conda env create -f diffrl_conda_upgrade.yml

# Activate conda environment
source activate diffrl_warp

# Install the package
cd externals/rl_games
pip install -e .

# go to examples
cd ../../examples