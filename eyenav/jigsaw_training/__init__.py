from .train import train
from .utils import get_kwargs_as_dict, EXPERIMENT_TAG

def get_default_configs():
    ''' Return a dict of configurations for training/deploy stages

    The parameters and their default values are automatically aggregated from function's 
    arguments.
    '''
    configs = {}
    for fn in [train]:
        configs[fn.__name__] = get_kwargs_as_dict(fn)
    configs[EXPERIMENT_TAG] = 'Enter your experiment name'
    return configs