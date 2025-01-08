#!/usr/bin/env python3
import argparse, yaml, os, json
from eyenav.jigsaw_training import get_default_configs
from eyenav.jigsaw_training.utils import scaffold_directory_structure, logger
from eyenav.jigsaw_training.utils import EXPERIMENT_TAG, TRAIN_TAG, DATASET_TAG, EVALUATION_TAG, INFERENCE_TAG, CHECKPOINT_TAG, LOG_TAG
from eyenav.jigsaw_training import train


def add_arguments_from_dict(
        parser: argparse.ArgumentParser, 
        args_dict: dict
    ):
    ''' Add key:value tuples from input dict as arguments to argparse object
    '''
    for argument, default_value in args_dict.items():
        if default_value is not None:
            parser.add_argument(f'--{argument}', type=type(default_value))
        else:
            parser.add_argument(f'--{argument}')


def prepare_kwargs_for_train(directories, configs):
    ''' Prepare auto-arguments for train stage
    '''
    config = configs[TRAIN_TAG]
    if config['train_image_dirpath'] == 'auto':
        raise NotImplementedError
    if config['val_image_dirpath'] == 'auto':
        raise NotImplementedError
    if config['log_dirpath'] == 'auto':
        log_dirpath = os.path.join(directories[TRAIN_TAG], 'log')
    else:
        log_dirpath = config['log_dirpath']
    if not os.path.isdir(log_dirpath):
        os.makedirs(log_dirpath)
    
    if config['checkpoint_dirpath'] == 'auto':
        checkpoint_dirpath = os.path.join(directories[TRAIN_TAG], 'checkpoint')
    else:
        checkpoint_dirpath = config['checkpoint_dirpath']
    if not os.path.isdir(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    if config['trained_classifier_filepath'] == 'auto':
        trained_classifier_filepath = os.path.join(directories[TRAIN_TAG], 'classifier')
    else:
        trained_classifier_filepath = config['trained_classifier_filepath']
    
    if config['trained_backbone_filepath'] == 'auto':
        trained_backbone_filepath = os.path.join(directories[TRAIN_TAG], 'backbone')
    else:
        trained_backbone_filepath = config['trained_backbone_filepath']

    return {
        'train_image_dirpath': config['train_image_dirpath'],
        'val_image_dirpath': config['val_image_dirpath'],
        'log_dirpath': log_dirpath,
        'checkpoint_dirpath': checkpoint_dirpath,
        'trained_classifier_filepath': trained_classifier_filepath,
        'trained_backbone_filepath': trained_backbone_filepath
    }        


def main():

    default_configs = get_default_configs()
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', default=None)
    subparsers = parser.add_subparsers(help='wrangle | train | eval | inference')

    parser_train = subparsers.add_parser('train', help='Model training stage')
    add_arguments_from_dict(parser_train, default_configs['train'])
    parser_train.set_defaults(func=train, pre_func=prepare_kwargs_for_train)

    args = parser.parse_args()

    if args.config_file is None:
        logger.warning('Config file not found. Create default config and terminate program...')
        cfg_file = './default-config.yml'
        if os.path.isfile(cfg_file):
            while True:
                answer = input(f'Default config file {cfg_file} have existed. Overwrite ? [Y/n] ')
                if answer == 'Y':
                    with open(cfg_file, 'wt') as dst:
                        yaml.dump(default_configs, dst, default_flow_style=False)
                    logger.info('Config file {cfg_file} created. Edit parameters to your need then come back.')
                    return
                elif answer == 'n':
                    logger.info('Not proceeding anything. Terminate program.')
                    return
        else:
            with open(cfg_file, 'wt') as dst:
                yaml.dump(default_configs, dst)
            logger.info(f'Default config is saved at {cfg_file}. Edit and run the program.')
    
    elif not os.path.isfile(args.config_file):
        raise ValueError(f'Config file {args.config_file} not found')
    else:
        with open(args.config_file, 'rt') as src:
            configs = yaml.safe_load(src)

        logger.info('Scaffolding directory structure')
        home_dir = os.path.dirname(os.path.abspath(args.config_file))
        directories = scaffold_directory_structure(
            home_dir, 
            experiment_unique_id=configs[EXPERIMENT_TAG],
            with_train_dir=True, 
            with_evaluation_dir=True,
            with_inference_dir=True)

        if not hasattr(args, 'pre_func'):
            logger.info('Directories created. Prepare necessary data and run again')

        fn_kwargs = args.pre_func(directories, configs)
        configs[args.func.__name__].update(fn_kwargs)

        logger.info('The following keyword arguments are finalized:')
        logger.info(json.dumps(configs[args.func.__name__], indent=2))

        args.func(**configs[args.func.__name__])

if __name__ == '__main__':
    main()