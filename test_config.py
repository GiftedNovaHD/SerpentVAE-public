from train_utils.config_utils import load_config
from serpentvae.modules.module_utils.layer_parser import layer_parser, get_aliases


config = load_config("debug_config")

print(f"Encoder config:{config["encoder"]}")

aliases = get_aliases(config["encoder"])

print(f"Aliases: {aliases}")

layer_config = layer_parser(config["encoder"]["layer_config"], aliases)

print(f"Layer config: {layer_config}")