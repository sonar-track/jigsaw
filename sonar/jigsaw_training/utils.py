import os, logging, sys, inspect

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    green = "\x1b[1;32m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
# create logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(filename='logging', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(CustomFormatter())
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(CustomFormatter())
logger.addHandler(stdout_handler)


EXPERIMENT_TAG  = 'experiment_unique_id'
TRAIN_TAG       = 'train'
DATASET_TAG     = 'dataset'
CHECKPOINT_TAG  = 'checkpoint'
LOG_TAG         = 'log'
EVALUATION_TAG  = 'evaluate'
INFERENCE_TAG   = 'inference'


def get_kwargs_as_dict(func):
    """
    This function takes a function as input and returns a dictionary
    containing key-value pairs from the function's arguments.

    Args:
        func: The function to inspect.

    Returns:
        A dictionary containing key-value pairs from the function's arguments.
    """
    args = {}
    sig = inspect.signature(func)
    for kw_name, param in sig.parameters.items():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            if param.default != param.empty:
                args[kw_name] = param.default
        elif param.kind == param.KEYWORD_ONLY:
            args[kw_name] = param.default
    return args


def get_kwargtypes_as_dict(func):
    """
    This function takes a function as input and returns a dictionary
    containing key-value pairs from the function's arguments.

    Args:
        func: The function to inspect.

    Returns:
        A dictionary containing key-value pairs from the function's arguments.
    """
    args = {}
    sig = inspect.signature(func)
    for kw_name, param in sig.parameters.items():
        if param.kind == param.KEYWORD_ONLY:
            args[kw_name] = param.annotation
    return args


def scaffold_directory_structure(
        home_dir: str, 
        experiment_unique_id: str,
        with_train_dir: bool=False,
        with_evaluation_dir: bool=False,
        with_inference_dir: bool=False):
    '''
    '''
    output_dir = os.path.join(home_dir, 'generated')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    expt_dir = os.path.join(output_dir, experiment_unique_id)
    if not os.path.isdir(expt_dir):
        os.mkdir(expt_dir)
    
    if with_train_dir:
        train_dir = os.path.join(expt_dir, TRAIN_TAG)
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)

        dataset_dir = os.path.join(train_dir, DATASET_TAG)
        if not os.path.isdir(dataset_dir):
            os.mkdir(dataset_dir)

        checkpoint_dir = os.path.join(train_dir, CHECKPOINT_TAG)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        log_dir = os.path.join(train_dir, LOG_TAG)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    if with_evaluation_dir:
        eval_dir = os.path.join(expt_dir, EVALUATION_TAG)
        if not os.path.isdir(eval_dir):
            os.mkdir(eval_dir)

    if with_inference_dir:
        inference_dir = os.path.join(expt_dir, INFERENCE_TAG)
        if not os.path.isdir(inference_dir):
            os.mkdir(inference_dir)
    
    return {
        DATASET_TAG: dataset_dir,
        TRAIN_TAG: train_dir,
        CHECKPOINT_TAG: checkpoint_dir,
        LOG_TAG: log_dir,
        EVALUATION_TAG: eval_dir,
        INFERENCE_TAG: inference_dir
    }
    