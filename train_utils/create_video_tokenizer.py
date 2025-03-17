from transformers import VideoMAEFeatureExtractor

def create_video_feature_extractor(): 
  """
  Create and returns the VideoMAE feature extractor. 

  See https://huggingface.co/docs/transformers/en/model_doc/videomae for more details. 

  Returns:
    feature_extractor (AutoFeatureExtractor): The VideoMAE feature extractor
  """
  
  # Default checkpoints
  feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
  return feature_extractor