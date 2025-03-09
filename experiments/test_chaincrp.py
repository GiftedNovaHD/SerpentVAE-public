"""
Script to test torchscripted ChainCRP module and ensure that ChainCRP can be backpropagated through.
"""
import torch 

from serpentvae.ops.segment.boundary.ChainCRP_grad import ChainCRP 

def test_torchscript(): 
  batch_size, seq_len, concept_dim = 2, 10, 16

  dummy_encoder_predictions = torch.rand(batch_size, seq_len, 1)

  # Arbitrary value for previous reconstruction loss error 
  dummy_recon_loss = torch.tensor([0.5])

  model = ChainCRP(use_odds_ratio=True, dtype=torch.bfloat16)

  try: 
    compiled_model = torch.compile(model)
    out = compiled_model(dummy_encoder_predictions, dummy_recon_loss)
    print("TorchScript compilation successful") 
    print("Output segmentation shape: ", out.shape)
    print("Output segmentation")
    print(out) 
  except Exception as e: 
    print("failed")
    print(e)

if __name__ == "__main__":
  test_torchscript()