from os.path import expanduser, join, isfile, isdir, dirname, abspath
from os import makedirs
import json
from pathlib import Path

parent_path = dirname(abspath(__file__))
parent_path = abspath(dirname(abspath(__file__)) + "/../")

JUNN_DIR = join(parent_path, 'checkpoint')
JUNN_SETTINGS = join(JUNN_DIR, 'settings.json')
if not isdir(JUNN_DIR):
    makedirs(JUNN_DIR)


def get_data_loc():
    if not isfile(JUNN_SETTINGS):
        data_dir = join(JUNN_DIR, 'data')
        if not isdir(data_dir):
            makedirs(data_dir)
        return data_dir
    else:
        with open(JUNN_SETTINGS) as f:
            settings = json.load(f)
        return settings['data_loc']


def set_data_loc(data_path):
    if not isfile(JUNN_SETTINGS):
        settings = {}
    else:
        with open(JUNN_SETTINGS) as f:
            settings = json.load(f)
    
    settings['data_loc'] = data_path
    
    print(settings)

    with open(JUNN_SETTINGS, 'w') as f:
        json.dump(settings, f)

print(get_data_loc())