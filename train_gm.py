import importlib
import argparse
import torch

from utils.config import load_config
from utils.helper import get_logger
from utils.data_handler import get_data_loaders

CONFIG_PATH = "./configs/gm_01.yaml"

def _get_model(module_path, config):
    def _model_class(module_path, class_name):
        m = importlib.import_module(module_path)
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(module_path, model_config['name'])
    return model_class(**model_config)

def main():
    # Create main logger
    logger = get_logger('Graph matching requester')

    parser = argparse.ArgumentParser(description='Graph matching')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default = CONFIG_PATH)
    args = parser.parse_args()

    # Load and log experiment configuration
    config = load_config(args.config)
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    module_path = "models.model"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(_get_model(module_path, config))
    else:
        model = _get_model(module_path, config)

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}', using {torch.cuda.device_count()} GPUs...")
    model = model.to(config['device'])

    # Create data loaders
    loaders = get_data_loaders(config)

if __name__ == '__main__':
    main()