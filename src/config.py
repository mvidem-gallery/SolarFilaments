import yaml
from omegaconf import OmegaConf

config = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)

config = OmegaConf.create(config)