import yaml
import logging.config


def configure_logging():
    with open('logging_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())
    return config

