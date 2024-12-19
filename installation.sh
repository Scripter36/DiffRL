#!/usr/bin/env bash

# Create conda environment
conda env create -f diffrl_conda_upgrade.yml

# Activate conda environment
conda activate shac

# Install the package
cd dflex
pip install -e .
cd ../externals/rl_games
pip install -e .

# Link cuda
ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64

# go to examples
cd ../../examples