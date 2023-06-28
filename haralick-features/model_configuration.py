import yaml
import pprint

with open('model_arch.yaml', 'r') as config_file:
    data = yaml.safe_load(config_file)
    pprint.pp(data)
