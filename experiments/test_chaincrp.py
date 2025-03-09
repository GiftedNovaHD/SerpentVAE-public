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
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    out = model(dummy_encoder_predictions, dummy_recon_loss)
    
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)

    compiled_model = torch.compile(model)
    
    # To get it to compile
    _ = compiled_model(dummy_encoder_predictions, dummy_recon_loss)

    compiled_start_time = torch.cuda.Event(enable_timing=True)
    compiled_end_time = torch.cuda.Event(enable_timing=True)
    
    compiled_start_time.record()
    
    out = compiled_model(dummy_encoder_predictions, dummy_recon_loss)
    
    compiled_end_time.record()
    
    torch.cuda.synchronize()
    compiled_elapsed_time = compiled_start_time.elapsed_time(compiled_end_time)
    
    print("Torch compilation successful") 
    print("Output segmentation shape: ", out.shape)
    print("Output segmentation")
    print(out)
    print(f"Elapsed time: {elapsed_time:.2f} ms")
    print(f"Compiled Elapsed time: {compiled_elapsed_time:.2f} ms")
  except Exception as e: 
    print("Compilation failed")
    print(e)
if __name__ == "__main__":
  test_torchscript()