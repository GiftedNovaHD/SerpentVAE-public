from transformers import AutoTokenizer

def create_text_tokenizer():
  """
  Create and returns the DeepSeek-V3 tokenizer

  Returns:
    tokenizer (AutoTokenizer): The DeepSeek-V3 tokenizer
  """
  # Create tokenizer - This is basically the DeepSeek-V3 tokeniser
  # NOTE: Vocab size is 129280
  tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer_config")

  return tokenizer