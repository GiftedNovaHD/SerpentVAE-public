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
                     distribution_desired_std = config["dist_desired_std"],
                     num_encoder_layers = config["num_encoder_layers"],
                     num_decoder_layers = config["num_decoder_layers"],
                     state_dim = config["mamba_state_dim"],
                     conv_length = config["mamba_conv_length"],
                     mamba_expand = config["mamba_expand"],
                     mamba_head_dim = config["mamba_head_dim"],
                     mlp_inner_dim = config["mlp_inner_dim"],
                     confidence_module_inner_dim = config["confidence_inner_dim"],
                     segment_predictor_inner_dim = config["segment_pred_inner_dim"],
                     enable_qnet = config["enable_qnet"],
                     num_qnet_layers = config["num_qnet_layers"],
                     qnet_conv_length = config["qnet_conv_length"],
                     qnet_mamba_expand = config["qnet_mamba_expand"],
                     qnet_mamba_head_dim = config["qnet_mamba_head_dim"],
                     qnet_mlp_inner_dim = config["qnet_mlp_inner_dim"],
                     qnet_mamba_state_dim = config["qnet_mamba_state_dim"],
                     share_input_embeddings = config["share_input_embeddings"],
                     tie_embeddings = config["tie_embeddings"],
                     residual_in_fp32 = config["residual_in_fp32"],
                     device = config["device"],
                     dtype = config["dtype"]
                    )

  model.to(config["device"])

  return model