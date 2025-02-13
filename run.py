import json
import os
import subprocess
import sys
import re
from pathlib import Path
import time

import mlflow
import mlflow.entities
import mlflow.experiments

from dotenv import load_dotenv

load_dotenv()

# ANSI color codes
YELLOW = "\033[93m"
RESET = "\033[0m"

# Function to list files or directories
def list_items(path, filter_ext=None):
    items = sorted(Path(path).iterdir())
    if filter_ext:
        items = [item for item in items if item.suffix == filter_ext]
    return items

# Prompt user for input with a list of choices
def prompt_choice(options, message, name_key = lambda option: option.name):
    while True:
        for i, option in enumerate(options, start=1):
            print(f"{i}) {name_key(option)}")  # White (default) text for options
        try:
            choice = int(input(f"{YELLOW}{message} {RESET}")) - 1  # Yellow text for input prompt
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print(f"{YELLOW}Invalid choice. Please try again.{RESET}")  # Yellow text for errors
        except ValueError:
            print(f"{YELLOW}Invalid input. Please enter a number.{RESET}")  # Yellow text for errors

def prompt_choice_dataframe(options, message, print_keys):
    while True:
        reduced_options = options[print_keys]
        key_max_widths = {key: max(len(str(value)) for value in reduced_options[key]) for key in print_keys}
        id_width = max(len(str(len(options))), 2)
        # print header
        print_keys_str = ' | '.join(f'{key.ljust(max_width)}' for key, max_width in key_max_widths.items())
        print(f'| {"id".ljust(id_width)} | {print_keys_str} |')
        # print rows
        for i, row in reduced_options.iterrows():
            row_str = ' | '.join(f'{str(row[key]).ljust(max_width)}' for key, max_width in key_max_widths.items())
            print(f'| {str(i + 1).ljust(id_width)} | {row_str} |')
        try:
            choice = int(input(f"{YELLOW}{message} {RESET}")) - 1
            if 0 <= choice < len(options):
                return options.iloc[choice]
            else:
                print(f"{YELLOW}Invalid choice. Please try again.{RESET}")  # Yellow text for errors
        except ValueError:
            print(f"{YELLOW}Invalid input. Please enter a number.{RESET}")  # Yellow text for errors

def prompt_choice_not_continuous(options, indices, names, message):
    while True:
        for i, name in zip(indices, names):
            print(f"{i}) {name}")
        try:
            choice = input(f"{YELLOW}{message} {RESET}")
            if choice not in indices:
                print(f"{YELLOW}Invalid choice. Please try again.{RESET}")  # Yellow text for errors
            else:
                return options[indices.index(choice)]
        except ValueError:
            print(f"{YELLOW}Invalid input. Please enter a number.{RESET}")  # Yellow text for errors


config_path = Path("cfg/shac")
logs_path = Path("logs")

def main():
    while True:
        print(f"{YELLOW}Select an option:{RESET}")  # Yellow text for headers
        print("1) Train")  # White (default) text for options
        print("2) Inference")  # White (default) text for options
        choice = input(f"{YELLOW}Enter your choice (1/2/3): {RESET}").strip()  # Yellow text for input prompt

        if choice == "1":
            # Training mode
            print(f"{YELLOW}Training mode selected.{RESET}")  # Yellow text for status messages
            
            env_files = list_items(config_path, ".yaml")
            if not env_files:
                print(f"{YELLOW}No environments found.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_env = prompt_choice(env_files, "Select an environment by number")
            selected_env_name = selected_env.stem
            log_dir = logs_path / selected_env_name / "shac"

            print(f"{YELLOW}Running training for environment: {selected_env_name}{RESET}")  # Yellow text for status messages
            # Print command in white (default)
            print(f"python scripts/train_shac.py --cfg {selected_env} --logdir {log_dir}")
            num_tries = 1
            while True:
                try:
                    result = subprocess.run([sys.executable, "scripts/train_shac.py", "--cfg", selected_env, "--logdir", log_dir])
                    if result.returncode == 0:
                        print(f"{YELLOW}Training completed with {num_tries} tries.{RESET}")  # Yellow text for status messages
                        break
                    else:
                        raise Exception(f"Training failed with {num_tries} tries.")
                except Exception as e:
                    # if interrupted by user, exit
                    if isinstance(e, KeyboardInterrupt):
                        print(f"{YELLOW}Training interrupted by user.{RESET}")
                        sys.exit(0)
                    print(f"{YELLOW}Error: {e}{RESET}")  # Yellow text for errors
                    print(f"{YELLOW}Restarting training... ({num_tries} tries){RESET}")  # Yellow text for errors
                    num_tries += 1
                    time.sleep(3)
            break

        elif choice == "2":
            # Inference mode
            print(f"{YELLOW}Inference mode selected.{RESET}")  # Yellow text for status messages
            
            # from mlflow, get the experiment list
            experiment_list = mlflow.search_experiments(mlflow.entities.ViewType.ACTIVE_ONLY)
            experiment_list.sort(key=lambda x: int(x.experiment_id))
            selected_experiment = prompt_choice_not_continuous(experiment_list, [experiment.experiment_id for experiment in experiment_list], [experiment.name for experiment in experiment_list], "Select an experiment by number")
            print(f"Selected experiment: {selected_experiment.name}")

            # get the run list
            run_list = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

            keys = ['tags.mlflow.runName', 'run_id', 'status', 'start_time', 'end_time']
            selected_run = prompt_choice_dataframe(run_list, "Select a run by number", keys)
            print(f"Selected run: {selected_run['run_id']}")

            # get the checkpoint list
            checkpoint_list = json.loads(selected_run['tags.mlflow.log-model.history'])
            selected_checkpoint = prompt_choice(checkpoint_list, "Select a checkpoint by number", lambda checkpoint : checkpoint['artifact_path'])
            run_id = selected_run['run_id']
            checkpoint_path = selected_checkpoint["artifact_path"]

            # ask for render, default to false
            render = input(f"{YELLOW}Render? (y/N): {RESET}").strip()
            if render == "y":
                render = True
            else:
                render = False

            # subprocess to run inference
            subprocess.run([sys.executable, "scripts/train_shac.py", "--checkpoint", checkpoint_path, "--run", run_id, "--play"] + (["--render"] if render else []))
            break
        else:
            print(f"{YELLOW}Invalid choice. Please try again.{RESET}")  # Yellow text for errors


if __name__ == "__main__":
    main()