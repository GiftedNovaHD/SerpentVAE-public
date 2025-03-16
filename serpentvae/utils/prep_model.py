from typing import Dict

from serpentvae.modules.SerpentVAE import SerpentVAE

def prep_model(config: Dict) -> SerpentVAE:
  """
  Prepares and returns a (SerpentVAE) model based on config parameters. 

  Args: 
    config (dict): The (SerpentVAE) model's hyperparameters 

  Returns 
    model (SerpentVAE): The (SerpentVAE) neural network Module
  """

  model = SerpentVAE(hidden_dim = config["hidden_dim"],
                     concept_dim = config["concept_dim"],
                     vocab_size = config["vocab_size"],
                     distribution_config = config["distribution"],
                     encoder_config = config["encoder"],
                     decoder_config = config["decoder"],
                     confidence_module_config = config["confidence_module"],
                     enable_qnet = config["enable_qnet"],
                     qnet_config = config["qnet"],
                     share_input_embeddings = config["share_input_embeddings"],
                     tie_embeddings = config["tie_embeddings"],
                     residual_in_fp32 = config["residual_in_fp32"],
                     device = config["device"],
                     dtype = config["dtype"]
                    )

  model.to(config["device"])

  return model