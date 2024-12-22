import os, sys, signal, sys

def other_suppress_out():
    """
    Redirect stdout to null device to suppress print statements.

    Parameters: None
    Returns: None
    """
    sys.stdout = open(os.devnull, 'w')

def other_restore_out():
    """
    Restore the standard output.

    Parameters: None
    Returns: None
    """
    sys.stdout = sys.__stdout__

