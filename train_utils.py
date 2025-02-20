from os.path import dirname, abspath
import yaml
import yaml_include
from typing import Dict

def recursive_update(d: Dict, u: Dict) -> Dict:
    """
    Merge two dictionaries recursively.

    Args:
      d (Dict): The base dictionary that contents from u is added to
      u (Dict): The dictionary with the contents to add to d

    Returns:
      d (Dict): The updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    
    return d


def load_yaml(yaml_path):
    yaml.add_constructor(tag = "!include",
                         constructor = yaml_include.Constructor(base_dir=dirname(abspath(yaml_path)))
                        )
    
    with open(yaml_path, "r") as file:
        config = yaml.full_load(file)
    
    while "__base__" in config.keys():
        base = config["__base__"]
        del config["__base__"]
        config = recursive_update(base, config)
    
    return config