import os
import subprocess
import sys
import re
from pathlib import Path

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
def prompt_choice(options, message):
    while True:
        for i, option in enumerate(options, start=1):
            print(f"{i}) {option.name}")  # White (default) text for options
        try:
            choice = int(input(f"{YELLOW}{message} {RESET}")) - 1  # Yellow text for input prompt
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print(f"{YELLOW}Invalid choice. Please try again.{RESET}")  # Yellow text for errors
        except ValueError:
            print(f"{YELLOW}Invalid input. Please enter a number.{RESET}")  # Yellow text for errors

config_path = Path("examples/cfg/shac")
logs_path = Path("examples/logs")

def main():
    while True:
        print(f"{YELLOW}Select an option:{RESET}")  # Yellow text for headers
        print("1) Train")  # White (default) text for options
        print("2) Render")  # White (default) text for options
        print("3) Play")  # White (default) text for options
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
            print(f"python examples/train_shac.py --cfg {selected_env} --logdir {log_dir}")
            num_tries = 1
            while True:
                try:
                    result = subprocess.run([sys.executable, "examples/train_shac.py", "--cfg", selected_env, "--logdir", log_dir])
                    if result.returncode == 0:
                        print(f"{YELLOW}Training completed with {num_tries} tries.{RESET}")  # Yellow text for status messages
                        break
                    else:
                        raise Exception(f"Training failed with {num_tries} tries.")
                except Exception as e:
                    print(f"{YELLOW}Error: {e}{RESET}")  # Yellow text for errors
                    print(f"{YELLOW}Restarting training... ({num_tries} tries){RESET}")  # Yellow text for errors
                    num_tries += 1
            break

        elif choice == "2":
            # Rendering mode
            print(f"{YELLOW}Rendering mode selected.{RESET}")  # Yellow text for status messages
            
            env_dirs = list_items(logs_path, None)
            if not env_dirs:
                print(f"{YELLOW}No environments found.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_env = prompt_choice(env_dirs, "Select an environment by number")
            selected_env_name = selected_env.name

            # Move into the "shac" subdirectory
            shac_path = selected_env / "shac"
            if not shac_path.exists():
                print(f"{YELLOW}No 'shac' directory found for {selected_env_name}.{RESET}")  # Yellow text for errors
                sys.exit(1)

            date_dirs = list_items(shac_path, None)
            if not date_dirs:
                print(f"{YELLOW}No dates found for the selected environment.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_date = prompt_choice(date_dirs, "Select a date by number")

            policy_files = list_items(selected_date, ".pt")
            if not policy_files:
                print(f"{YELLOW}No policy files (*.pt) found for the selected date.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_policy = prompt_choice(policy_files, "Select a policy by number")

            print(f"{YELLOW}Running rendering for environment: {selected_env_name}{RESET}")  # Yellow text for status messages
            # Print command in white (default)
            print(f"python examples/train_shac.py --cfg {config_path}/{selected_env_name}.yaml --checkpoint {selected_policy} --play --render")
            subprocess.run([sys.executable, "examples/train_shac.py", "--cfg", config_path / f"{selected_env_name}.yaml", "--checkpoint", selected_policy, "--play", "--render"])
            break

        elif choice == "3":
            # Play mode
            print(f"{YELLOW}Play mode selected.{RESET}")  # Yellow text for status messages
            
            env_dirs = list_items(logs_path, None)
            if not env_dirs:
                print(f"{YELLOW}No environments found.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_env = prompt_choice(env_dirs, "Select an environment by number")
            selected_env_name = selected_env.name

            # Move into the "shac" subdirectory
            shac_path = selected_env / "shac"
            if not shac_path.exists():
                print(f"{YELLOW}No 'shac' directory found for {selected_env_name}.{RESET}")  # Yellow text for errors
                sys.exit(1)

            date_dirs = list_items(shac_path, None)
            if not date_dirs:
                print(f"{YELLOW}No dates found for the selected environment.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_date = prompt_choice(date_dirs, "Select a date by number")

            policy_files = list_items(selected_date, ".pt")
            if not policy_files:
                print(f"{YELLOW}No policy files (*.pt) found for the selected date.{RESET}")  # Yellow text for errors
                sys.exit(1)

            selected_policy = prompt_choice(policy_files, "Select a policy by number")

            print(f"{YELLOW}Running play for environment: {selected_env_name}{RESET}")  # Yellow text for status messages
            # Print command in white (default)
            print(f"python examples/train_shac.py --cfg {config_path}/{selected_env_name}.yaml --checkpoint {selected_policy} --play")
            subprocess.run([sys.executable, "examples/train_shac.py", "--cfg", config_path / f"{selected_env_name}.yaml", "--checkpoint", selected_policy, "--play"])
            break

        else:
            print(f"{YELLOW}Invalid choice. Please try again.{RESET}")  # Yellow text for errors


if __name__ == "__main__":
    main()