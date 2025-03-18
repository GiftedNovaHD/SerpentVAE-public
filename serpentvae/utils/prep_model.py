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
                     distribution_config = config["distribution"],
                     encoder_config = config["encoder"],
                     decoder_config = config["decoder"],
                     recon_loss_name = config["recon_loss_name"],
                     recon_loss_reduction = config["recon_loss_reduction"],
                     vocab_size = config["vocab_size"],
                     use_odds_ratio = config["use_odds_ratio"],
                     alpha = config["alpha"],
                     beta = config["beta"],
                     ema_decay_factor = config["ema_decay_factor"],
                     enable_confidence_module = config["enable_confidence_module"],
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