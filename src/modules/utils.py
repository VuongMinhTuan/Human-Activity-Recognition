import os, yaml, torch
from rich import print
from typing import Dict, List, Tuple, Union




# Load the configuration
def load_yaml(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileExistsError("File is not found!!!")
    
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg



# Save the configuration
def save_yaml(path: str, **config):
    if os.path.exists(path):
        print("\n[bold][yellow]File is already existed![/][/]")

        while True:
            answer = input("\nDo you want to overwrite this file? ([Y]/n) ")

            if answer == "Y" or answer == "n" or answer == "\n":
                break
            
            print("\n[bold][yellow]Wrong syntax!!!")
        
        if answer == "\n":
            return
        

    print("\n[bold][yellow]Saving configuration...[/][/]")

    with open(path, 'w') as file:
        file.write(
            "### This folder contains configuration###"
            "\n"
        )

        yaml.dump(config, file)


    print("\n[bold][yellow]Successfully saved configuration!!!")



# Handles the specification of device choice
def device_handler(value: str = "auto") -> str:
    # Check type
    if not isinstance(value, str):
        raise TypeError(
            f"The 'value' parameter must be a string. Got {type(value)} instead."
        )
    
    # Prepare
    value = value.strip().lower()

    # Check value
    if not (value in ["auto", "gpu", "cpu"] or value.startswith("cuda")):
        raise ValueError(
            f'Device options: ["auto", "cpu", "cuda"]. Got {value} instead.'
        )
    
    # Check auto option
    if value == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = value

    return device


# Calculate the number of workers based on an input value
def workers_handler(value: Union[int, float]) -> int:
    max_workers = os.cpu_count()

    match value:
        case int():
            workers = value
        case float():
            workers = int(max_workers * value)
        case _:
            workers = 0

    if not (-1 < workers < max_workers):
        raise ValueError(
            f"Number of workers is out of bounds. Min: 0 | Max: {max_workers}"
        )
    
    return workers


# Create a tuple with specified dimensions and values
def tuple_handler(value: Union[int, List[int], Tuple[int]], max_dim: int) -> Tuple:
    # Check max_dim
    if not isinstance(max_dim, int) and max_dim > 1:
        raise TypeError(
            f"The 'max_dim' parameter must be an int. Got {type(max_dim)} instead."
        )
    
    # Check value
    if isinstance(value, int):
        output = tuple([value] * max_dim)
    else:
        try:
            output = tuple(value)
        except:
            raise TypeError(
                f"The 'value' parameter must be an int or tuple or list. Got {type(value)} instead."
            )
        
    if len(output) != max_dim:
        raise ValueError(
            f"The lenght of 'value' parameter must be equal to {max_dim}. Got {len(output)} instead."
        )
    
    return output