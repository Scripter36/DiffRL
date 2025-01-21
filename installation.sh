#!/usr/bin/env bash

# Create conda environment
conda env create -f diffrl_conda.yml

# Activate conda environment
conda activate diffrl

# Link cuda
ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64

# Finish
echo "Installation finished. Please run 'conda activate diffrl' to activate the environment, and 'python run.py' to start the program."
