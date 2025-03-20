from transformers import AutoFeatureExtractor

def create_video_tokenizer(): 
  """
  Creates and returns the VideoMAE video tokenizer (i.e. feature extractor) for processing video frames. 

  Returns:
    tokenizer (AutoFeatureExtractor): The VideoMAE video tokenizer to process video frames.
  """

  tokenizer = AutoFeatureExtractor.from_pretrained("MCG-NJU/VideoMAE-base")
  return tokenizer
