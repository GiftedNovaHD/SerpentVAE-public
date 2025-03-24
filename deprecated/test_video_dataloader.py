import multiprocessing
import torch
import sys
import traceback
import os

def init_worker():
    """Initialize worker process with CUDA"""
    if torch.cuda.is_available():
        # Get the worker's rank
        worker_id = multiprocessing.current_process().name
        # Set CUDA device based on worker ID
        device_id = int(worker_id.split('-')[-1]) % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        print(f"Worker {worker_id} initialized on CUDA device {device_id}")

# Ensure proper multiprocessing method for CUDA
if __name__ == "__main__":
    try:
        # Set spawn method for multiprocessing to work with CUDA
        multiprocessing.set_start_method('spawn', force=True)
        
        from train_utils.dataloaders.fastvit_video_dataloader import prep_video_dataset
        from train_utils.config_utils import load_config

        # Pre-initialize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            dummy = torch.zeros(1).cuda()
            print(f"CUDA initialized on device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        else:
            print("CUDA is not available, using CPU")
        
        config = load_config("video_debug_config")
        
        # Ensure config has device property
        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Set device to: {config['device']}")
        
        print("Creating dataloaders...")
        train_dataloader, test_dataloader, val_dataloader = prep_video_dataset(config=config)
        print("Successfully created dataloaders")
        
        print("Starting batch processing...")
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                print(f"Processing batch {batch_idx}")
                print(f"Batch shape: {batch.shape}")
                print(f"Batch dtype: {batch.dtype}")
                print(f"Batch device: {batch.device}")

                for batch_feature in batch:
                    print(batch_feature)
                # Only process one batch
                #break
            except Exception as e:
                print(f"Error processing batch {batch_idx}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                traceback.print_exc()
                #break
            
        print("Test completed successfully!")
        
    except Exception as e:
        print("Fatal error in main:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)