from os.path import dirname, abspath, exists
import yaml
import yaml_include
from typing import Dict
import torch

def load_config(config_name: str) -> Dict:
  """
  Returns the configuration dictionary for the given experiment

  Args:
    - `config_name` (`str`): The name of the experiment configuration file

  Returns:
    - `config` (`dict`): The configuration dictionary for the given experiment
  """
  config_file = load_yaml(config_name)

  formatted_config = change_yaml_dtype(config_file)

  return formatted_config

def recursive_update(d: Dict, u: Dict) -> Dict:
  """
  Merge two dictionaries recursively.

  Args:
    - `d` (`Dict`): The base dictionary that contents from `u` is added to
    - `u` (`Dict`): The dictionary with the contents to add to `d`

  Returns:
    - `d` (`Dict`): The updated dictionary
  """
  for k, v in u.items():
    if isinstance(v, dict):
      d[k] = recursive_update(d.get(k, {}), v)
    else:
      d[k] = v
    
  return d

def load_yaml(config_name: str) -> Dict:
  """
  Load the configuration YAML file for the given experiment

  Args:
    - `config_name` (`str`): The name of the experiment configuration file
  
  Returns:
    - `config_file` (`Dict`): The configuration dictionary for the given experiment
  """
  # Check that config file exists
  if not exists(f"configs/train_config/{config_name}.yaml"):
    raise ValueError(f"Config file {config_name}.yaml does not exist")
  else:
    print(f"Using config file {config_name}.yaml")
    config_file_path = f"configs/train_config/{config_name}.yaml"

  # Load config file
  yaml.add_constructor(tag = "!include",
                       constructor = yaml_include.Constructor(base_dir=dirname(abspath(config_file_path)))
                      )
  
  with open(config_file_path, "r") as file:
    config = yaml.full_load(file)
  
  while "__base__" in config.keys():
    base = config["__base__"]
    del config["__base__"]
    config = recursive_update(base, config)
  
  return config

def change_yaml_dtype(config: Dict) -> Dict:
  """
  Change the datatypes of the config dictionary to the appropriate types. 
  NOTE: This is necessary because the YAML file is loaded as a string, and we need to convert it to the appropriate (data)types for the config dictionary. 

  Args:
    - `config` (`Dict`): The configuration dictionary for the given experiment

  Returns:
    - `config` (`Dict`): The configuration dictionary for the given experiment
  """
  #config["train_epochs"] = int(config["train_epochs"])
  #config["eval_freq"] = int(config["eval_freq"])
  config["learning_rate"] = float(config["learning_rate"])
  config["min_learning_rate"] = float(config["min_learning_rate"])
  config["weight_decay"] = float(config["weight_decay"])
  config["dtype"] = dtype_converter(config["dtype"])

  return config

def dtype_converter(dtype_str: str) -> torch.dtype:
  """
  Convert a string for a dtype from the config yaml file into a torch.dtype

  Args:
    - `dtype_str` (`str`): The datatype in string form from the config yaml file

  Returns:
    - `model_dtype` (`torch.dtype`): The datatype in torch.dtype form
  """
  if dtype_str == "fp16":
    model_dtype = torch.float16

  elif dtype_str == "bf16":
    model_dtype = torch.bfloat16

  elif dtype_str == "fp32":
    model_dtype = torch.float32

  else:
    raise NotImplementedError(f"The datatype {dtype_str} is not supported")

  return model_dtype