from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, latest_weights_file_path
from train import get_model, get_ds, run_validation, greedy_decode
# from translate import translate
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)
# def translate_text(model, tokenizer_src, tokenizer_tgt, max_len, device):
#     model.eval()

#     try:
#         # get the console window width
#         with os.popen('stty size', 'r') as console:
#             _, console_width = console.read().split()
#             console_width = int(console_width)
#     except:
#         # If we can't get the console width, use 80 as default
#         console_width = 80

#     with torch.no_grad():
#         while True:
#             # Get input from the user
#             source_text = input("Enter a sentence in English (or 'exit' to quit): ")
#             if source_text.lower() == 'exit':
#                 break

#             # Tokenize the input text
#             encoder_input = tokenizer_src(source_text, return_tensors="pt").input_ids.to(device)
#             encoder_mask = (encoder_input != tokenizer_src.pad_token_id).unsqueeze(1).unsqueeze(2).to(device)

#             # Perform the translation using the model
#             model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

#             # Decode the model output to text
#             model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)

#             # Print the source and translated text
#             print('-'*console_width)
#             print(f"{f'SOURCE: ':>12}{source_text}")
#             print(f"{f'PREDICTED: ':>12}{model_out_text}")
#             print('-'*console_width)

# # Example usage:
# translate_text(model, tokenizer_src, tokenizer_tgt, max_len=50, device='cuda')
