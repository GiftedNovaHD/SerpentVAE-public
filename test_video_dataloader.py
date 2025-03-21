import multiprocessing
import torch

# Ensure proper multiprocessing method for CUDA
if __name__ == "__main__":
    # Set spawn method for multiprocessing to work with CUDA
    multiprocessing.set_start_method('spawn', force=True)
    
    from train_utils.dataloaders.prep_video_dataloader import prep_video_dataset
    from train_utils.config_utils import load_config

    # Pre-initialize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dummy = torch.zeros(1).cuda()
        print(f"CUDA initialized on device: {torch.cuda.get_device_name(0)}")
    
    config = load_config("video_debug_config")
    
    # Ensure config has device property
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
    train_dataloader, test_dataloader, val_dataloader = prep_video_dataset(config=config)

    print("Successfully created dataloaders")
    
    for batch_idx, batch in enumerate(train_dataloader):
        print(f"Batch {batch_idx} shape: {batch.shape}")
        # Only process one batch
        break
        
    print("Test completed successfully!")