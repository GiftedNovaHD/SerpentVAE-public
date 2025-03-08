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
    scripted_model = torch.jit.script(model)
    out = scripted_model(dummy_encoder_predictions, dummy_recon_loss)
    print("TorchScript compilation successful") 
    print("Output segmentation shape: ", out.shape)
    print("Output segmentation")
    print(out) 
  except Exception as e: 
    print("failed")
    print(e)

def test_backpropagation():
  batch_size, seq_len, concept_dim = 2, 10, 16

  dummy_encoder_predictions = torch.rand(batch_size, seq_len, 1)

  # Arbitrary value for previous reconstruction loss error 
  dummy_recon_loss = torch.tensor([0.5])

  model = ChainCRP(use_odds_ratio=True, dtype=torch.bfloat16)

  out = model(dummy_encoder_predictions, dummy_recon_loss)
  loss = out.mean()
  loss.backward()

  if model.log_theta.grad is not None:
    print("Backpropagation successful")
    print("Gradients: ", model.log_theta.grad)
  else:
    print("Differentiability test failed. No gradients for log theta")

if __name__ == "__main__":
  test_torchscript()
  test_backpropagation()