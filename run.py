import os
import sys
import re
from pathlib import Path


# Function to convert snake_case to PascalCase
def snake_to_pascal(snake_str):
    return ''.join(word.capitalize() for word in snake_str.split('_'))


# Function to convert PascalCase to snake_case
# append _ between upper and lower
# ex - PascalCase -> pascal_case
# ex2 - PascalCASE -> pascal_case
def pascal_to_snake(pascal_str):
    final_str = ''
    for i, char in enumerate(pascal_str):
        if i == len(pascal_str) - 1:
            final_str += char.lower()
        else:
            next_char = pascal_str[i + 1]
            if next_char.islower() and char.isupper():
                final_str += f"{char.lower()}_"
            else:
                final_str += char.lower()
    return final_str

# Function to list files or directories
def list_items(path, filter_ext=None):
    items = sorted(Path(path).iterdir())
    if filter_ext:
        items = [item for item in items if item.suffix == filter_ext]
    return items


# Prompt user for input with a list of choices
def prompt_choice(options, message):
    for i, option in enumerate(options, start=1):
        print(f"{i}) {option.name}")  # White (default) text for options
    choice = int(input(f"\033[93m{message} \033[0m")) - 1  # Yellow text for input prompt
    if 0 <= choice < len(options):
        return options[choice]
    else:
        print("\033[93mInvalid choice. Exiting.\033[0m")  # Yellow text for errors
        sys.exit(1)


def main():
    print("\033[93mSelect an option:\033[0m")  # Yellow text for headers
    print("1) Train")  # White (default) text for options
    print("2) Render")  # White (default) text for options
    choice = input("\033[93mEnter your choice (1/2): \033[0m").strip()  # Yellow text for input prompt

    if choice == "1":
        # Training mode
        print("\033[93mTraining mode selected.\033[0m")  # Yellow text for status messages
        config_path = Path("examples/cfg/shac")
        env_files = list_items(config_path, ".yaml")
        if not env_files:
            print("\033[93mNo environments found.\033[0m")  # Yellow text for errors
            sys.exit(1)

        selected_env = prompt_choice(env_files, "Select an environment by number")
        selected_env_name = selected_env.stem
        log_dir = f"examples/logs/{selected_env_name}/shac"

        print(f"\033[93mRunning training for environment: {selected_env_name}\033[0m")  # Yellow text for status messages
        # Print command in white (default)
        print(f"python examples/train_shac.py --cfg {selected_env} --logdir {log_dir}")
        os.system(f"python examples/train_shac.py --cfg {selected_env} --logdir {log_dir}")

    elif choice == "2":
        # Rendering mode
        print("\033[93mRendering mode selected.\033[0m")  # Yellow text for status messages
        logs_path = Path("examples/logs")
        env_dirs = list_items(logs_path, None)
        if not env_dirs:
            print("\033[93mNo environments found.\033[0m")  # Yellow text for errors
            sys.exit(1)

        selected_env = prompt_choice(env_dirs, "Select an environment by number")
        selected_env_name = selected_env.name

        # Move into the "shac" subdirectory
        shac_path = selected_env / "shac"
        if not shac_path.exists():
            print(f"\033[93mNo 'shac' directory found for {selected_env_name}.\033[0m")  # Yellow text for errors
            sys.exit(1)

        date_dirs = list_items(shac_path, None)
        if not date_dirs:
            print("\033[93mNo dates found for the selected environment.\033[0m")  # Yellow text for errors
            sys.exit(1)

        selected_date = prompt_choice(date_dirs, "Select a date by number")

        policy_files = list_items(selected_date, ".pt")
        if not policy_files:
            print("\033[93mNo policy files (*.pt) found for the selected date.\033[0m")  # Yellow text for errors
            sys.exit(1)

        selected_policy = prompt_choice(policy_files, "Select a policy by number")

        print(f"\033[93mRunning rendering for environment: {selected_env_name}\033[0m")  # Yellow text for status messages
        # Print command in white (default)
        print(f"python examples/train_shac.py --cfg examples/cfg/shac/{selected_env_name}.yaml --checkpoint {selected_policy} --play --render")
        os.system(
            f"python examples/train_shac.py --cfg examples/cfg/shac/{selected_env_name}.yaml --checkpoint {selected_policy} --play --render")

    else:
        print("\033[93mInvalid choice. Exiting.\033[0m")  # Yellow text for errors
        sys.exit(1)


if __name__ == "__main__":
    main()