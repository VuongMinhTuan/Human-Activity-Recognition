import os
import torch
from typing import Union, Tuple, List


__all__ = ["device_handler", "workers_handler", "tuple_handler"]


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