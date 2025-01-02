import os, sys, shutil, sys

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

def other_delete_directory(path: str) -> None:
    """
    Deletes the directory at the given path.
    """
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        raise Exception(f"clean: error: {e}")

def other_delete_file(path: str) -> None:
    """
    Deletes the file at the given path.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        raise Exception(f"clean: error: {e}")