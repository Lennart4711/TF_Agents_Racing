import numpy as np
import ast


def load_boarders(file_name):
    """Load the list of points and convert to numpy array"""
    with open(file_name, "r") as file:
        # Read the file and convert to list of tuples
        out = ast.literal_eval(file.read())
        # Convert to numpy array
        return np.array(out)
