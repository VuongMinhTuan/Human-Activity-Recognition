import os, yaml
from rich import print
from typing import Dict




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