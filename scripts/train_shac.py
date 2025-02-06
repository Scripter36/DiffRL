# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# gradient-based policy optimization by actor critic method
import sys, os

import mlflow.artifacts

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import argparse

import envs
import algorithms.shac as shac
import os
import sys
import yaml
import torch
import mlflow
from utils.mlflow_utils import flatten_dict, mlflow_manager, set_experiment_name_from_env, unflatten_dict

import numpy as np
import copy

from utils.common import *

def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    
    args = parser.parse_args()
    
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args

def get_args(): # TODO: delve into the arguments
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--cfg", "type": str, "default": "./cfg/shac/ant.yaml",
            "help": "Configuration file for training/playing"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test"},
        {"name": "--run", "type": str, "default": None,
            "help": "run id to play"},
        {"name": "--checkpoint", "type": str, "default": None,
            "help": "Path to the saved weights"},
        {"name": "--logdir", "type": str, "default": "logs/tmp/shac/"},
        {"name": "--save-interval", "type": int, "default": 0},
        {"name": "--no-time-stamp", "action": "store_true", "default": False,
            "help": "whether not add time stamp at the log path"},
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": None, "help": "Random seed"},
        {"name": "--render", "action": "store_true", "default": False,
            "help": "whether generate rendering file."}]
    
    # parse arguments
    args = parse_arguments(
        description="SHAC",
        custom_parameters=custom_parameters)
    
    return args

if __name__ == '__main__':
    args = get_args()
    vargs = vars(args)

    # load base config
    run_id = vargs["run"]
    checkpoint_name = vargs["checkpoint"]
    if checkpoint_name is not None and run_id is not None:
        # if checkpoint is provided, load the run and get the parameters
        checkpoint_path = f'runs:/{run_id}/{checkpoint_name}'
        loaded_run = mlflow.get_run(run_id)
        experiment_name = mlflow.get_experiment(loaded_run.info.experiment_id).name
        cfg_train = mlflow.artifacts.load_dict(loaded_run.info.artifact_uri + "/cfg_train.json")
        print('loaded parameters:', cfg_train)
    else:
        checkpoint_path = None
        # else, load the config file
        with open(args.cfg, 'r') as f:
            cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    # override the parameters with the command line arguments
    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        if vargs[key] is not None: 
            cfg_train["params"]["general"][key] = vargs[key]
        else:
            cfg_train["params"]["general"][key] = cfg_train["params"]["general"].get(key, None)

    # seed default value
    if cfg_train["params"]["general"]["seed"] is None:
        cfg_train["params"]["general"]["seed"] = random.randint(0, 1000000)

    # for playing, set the number of actors to 1 default
    if args.play or args.test:  
        cfg_train["params"]["config"]["num_actors"] = cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())
    
    args.device = torch.device(args.device)

    if args.train:
        # ----------- MLFlow integration starts here -----------
        set_experiment_name_from_env(cfg_train["params"]["config"].get("name", "default_experiment"))

        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow_manager.active_run = run
            mlflow.log_params(flatten_dict(cfg_train))
            # also store original cfg_train for reproducibility
            mlflow.log_dict(cfg_train, "cfg_train.json")
            traj_optimizer = shac.SHAC(cfg_train)
            if checkpoint_path is not None:
                traj_optimizer.load(checkpoint_path)
            traj_optimizer.train()
    else:
        traj_optimizer = shac.SHAC(cfg_train, render_name=experiment_name)
        if checkpoint_path is not None:
            traj_optimizer.load(checkpoint_path)
        traj_optimizer.run(cfg_train['params']['config']['player']['games_num'])