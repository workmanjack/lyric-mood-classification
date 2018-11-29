import datetime
import logging
import pickle
import time
import json
import os


logger = logging.getLogger(__name__)


def configure_logging(logname, verbosity=1):

    global logger

    level = logging.INFO
    if verbosity is not None and int(verbosity) > 0:
        level = logging.DEBUG

    logger.setLevel(logging.DEBUG)  # we adjust on console and file later
    # create file handler which logs even debug messages
    fh = logging.FileHandler(datetime.datetime.now().strftime('logs/{0}__%Y-%m-%d_%H-%M.log'.format(logname)), 'w', 'utf-8')
    fh.setLevel(logging.DEBUG)  # always log everything to file
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)  # only log to console what the user wants
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return


def elapsed_time(start):
    return (time.time() - start) / 60


def elapsed_time_str(start):
    return '{0:.02f} minutes'.format(elapsed_time(start))


def full_elapsed_time_str(start):
    return 'Elapsed Time: {0}'.format(elapsed_time_str(start))


def picklify(data, dest):
    """
    Helper function to save variables as pickle files to be reloaded by python later
    
    Args:
        data: any type, variable to save
        dest: str, where to save the pickle
        
    Returns: None
    """
    with open(dest, 'wb') as outfile:
        pickle.dump(data ,outfile)
    logger.debug('pickled {0} to {1}'.format(type(data), dest))
    return


def unpicklify(src):
    """
    Helper function to load previously pickled variables
    
    Args:
        src: str, pickle file
        
    Returns: loaded pickle data if successful; else, None
    """
    data = None
    with open(src, 'rb') as infile:
         data = pickle.load(infile)
    logger.debug('unpickled {0} from {1}'.format(type(data), src))
    return data


def read_file_contents(path, read_json=False):
    contents = None
    encoding = None
    try:
        with open(path, 'r') as f:
            encoding = 'default'
            if read_json:
                contents = json.load(f)
            else:
                contents = f.read()
    except:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                encoding = 'utf-8'
                if read_json:
                    contents = json.load(f)
                else:
                    contents = f.read()
        except:
            try:
                with open(path, 'r', encoding='utf-16') as f:
                    encoding = 'utf-16'
                    if read_json:
                        contents = json.load(f)
                    else:
                        contents = f.read()
            except:
                pass
    finally:
        if not contents:
            logger.error('could not read file contents for {0}'.format(path))
        return contents, encoding

