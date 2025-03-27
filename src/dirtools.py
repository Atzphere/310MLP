'''
Useful directory-manipulation functions

Functions:

    get_parent_dir(directory, depth)
    gets the (depth)th parent directory of (directory) -> os.path

    get_subdirs(directory, fullpath)
    gets a list of child folder names by default in (directory)
    Or their entire paths if (fullpath=True) -> list(str) or list(os.path)

VERSION 1.1

'''
import os


def get_parent_dir(directory, depth=1):
    path = directory
    for i in range(0, depth):
        path = os.path.dirname(path)
    return path


def get_subdirs(directory, fullpath=False, dir_filter=None):
    '''
    Gets the folder in a folder. Not recursive.
    Only returns folder names by default.

    Parameters
        directory (path str) : The parent folder to get the children of

        fullpath (bool) : Whether or not to return full folder paths.
                          Default value: False
    '''

    if fullpath:
        res = [os.path.join(directory, dI).replace("\\", "/")
               for dI in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, dI))]
    else:
        res = [dI.replace("\\", "/") for dI in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, dI))]
    if dir_filter:
        filtered = filter(lambda i: any([filt in os.path.basename(i) for filt in dir_filter]), res)
        return filtered

    return res


def get_files(directory, fullpath=False, file_filter=None):
    '''
    Gets the folder in a folder. Not recursive.
    Only returns folder names by default.

    Parameters
        directory (path str) : The parent folder to get the files of.

        fullpath (bool) : Whether or not to return full file paths.
                          Default value: False
    '''
    if fullpath:
        res = [os.path.join(directory, f) for f in os.listdir(
            directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        res = [f for f in os.listdir(
            directory) if os.path.isfile(os.path.join(directory, f))]

    if file_filter:
        filtered = filter(lambda i: any([filt in os.path.basename(i) for filt in file_filter]), res)
        return filtered

    return res
