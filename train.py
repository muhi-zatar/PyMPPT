import argparse
import yaml

from src.model import train

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", type=str, default='config.yml',
                             help="Experiment Configurations ")
    run_args = args_parser.parse_args()
    config_path = run_args.config

    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    train(hparams)
