# Python standard Library
import os
import pickle

# Third party libraries
from datetime import datetime, date
import numpy as np


def save_data_checkpoint(filename, path):
    """
    Save picklable object to destination path.

    Parameters
    ----------
    filename : pickable obj
        name of the picklable objetc to save.

    path : str
        full path of destination directory.

    Returns
    -------
        Print confirmation message.
    """

    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump(filename, f, protocol=pickle.HIGHEST_PROTOCOL)
    return print(
        f"Object saved successfully in {path} with {np.round(os.path.getsize(path) / 1024 / 1024, 2)}MB."
    )


def load_data_checkpoint(path):
    """
    Load picklable object from destination path.

    Parameters
    ----------
    filename : str
        name of the picklable objetc to save.

    path : str
        full path of destination directory.

    Returns
    -------
        Confirmation message.
    """

    if os.path.exists(path):
        with open(path, "rb") as f:
            filename = pickle.load(f)
            print(f"Object loaded successfully from {path}.")
            return filename
    else:
        return print(f"Object or {path} does not exist.")


def timer(start_time=None):
    """
    This function calculate the time between two points: Start Time and Timer.

    Parameters
    ----------
        Start time: given datetime or None (now as default).

    Returns
    -------
        Time taken by function in hours (integer), mins(integer) and secs (integer).
    """

    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        hour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600
        )
        min, sec = divmod(temp_sec, 60)
        print(
            f"Time taken by function: {int(hour)} hours , {int(min)} mins and {(sec)} secs"
        )
