from os.path import dirname, abspath
import yaml
import yaml_include
from typing import Dict
import torch

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

def dtype_converter(dtype_str: str) -> torch.dtype:
  """
  Convert a string for a dtype from the config yaml file into a torch.dtype

  Args:
    dtype_str (str): The datatype in string form from the config yaml file

  Returns:
    model_dtype (torch.dtype): The datatype in torch.dtype form
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