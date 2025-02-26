from transformers import AutoTokenizer
from datasets import load_dataset_builder, load_dataset
from train_utils import load_yaml, dtype_converter # For loading configs


config = load_yaml("configs/train_config/debug_config.yaml")

# NOTE: Using smallest possible version for testing
dataset_builder = load_dataset_builder(path = config["dataset_path"], name = config["dataset_name"])

# Load datasets
train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train")
test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test")
val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation")

# Create tokenizer - This is basically the DeepSeek-V3 tokeniser
# NOTE: Vocab size is 129280
tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer_config")

# Create sequences for training and validation
train_texts = train_dataset["text"][1:]
test_texts = test_dataset["text"][1:]
val_texts = val_dataset["text"][1:]

tokenized_train_texts = tokenizer(train_texts, padding = True, truncation = True, max_length = 128, return_tensors = "pt")

print(tokenized_train_texts['input_ids'][0])