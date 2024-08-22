
import sys
sys.path.append('/root/datav/nlp/NLP-Basis/4.Transformer/src')

from model import build_transformer, Transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import xml.etree.ElementTree as ET
from tokenizers.pre_tokenizers import PreTokenizer



import warnings
from tqdm import tqdm
import os
from pathlib import Path
import jieba

# Huggingface datasets and tokenizers
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers

from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, StripAccents

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def load_from_xml():
    # Function to parse XML and extract sentences
    def parse_xml(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        sentences = []
        for seg in root.findall(".//seg"):
            sentences.append(seg.text)
        return sentences

    # Load English and Chinese sentences
    english_sentences = parse_xml('data/translation/zh-en/IWSLT15.TED.dev2010.zh-en.en.xml')
    chinese_sentences = parse_xml('data/translation/zh-en/IWSLT15.TED.dev2010.zh-en.zh.xml')
    # Create a dictionary for the dataset
    data_dict = {
        "id": list(range(len(english_sentences))),  # Create an id for each sentence pair
        "translation": [{"en": en, "zh": zh} for en, zh in zip(english_sentences, chinese_sentences)]
    }
    # Create the dataset
    train_dataset = Dataset.from_dict(data_dict)
    # Wrap the dataset in a DatasetDict
    ds_raw = DatasetDict({
        "train": train_dataset
    })
    return ds_raw


def load_from_files(num_sample=15000):
    
    with open('data/translation/zh-en/train.tags.zh-en.en', 'r', encoding='utf-8') as en_file:
        english_sentences = en_file.readlines()[6:][:num_sample]
    with open('data/translation/zh-en/train.tags.zh-en.zh', 'r', encoding='utf-8') as zh_file:
        chinese_sentences = zh_file.readlines()[6:][:num_sample]
        
    # Create a dictionary for the dataset
    data_dict = {
        "id": list(range(len(english_sentences))),  # Create an id for each sentence pair
        "translation": [{"en": en, "zh": zh} for en, zh in zip(english_sentences, chinese_sentences)]
    }
    # Create the dataset
    train_dataset = Dataset.from_dict(data_dict)
    # Wrap the dataset in a DatasetDict
    ds_raw = DatasetDict({
        "train": train_dataset
    })
    return ds_raw


def greedy_decode(model: Transformer, source, source_mask, tokenizer_src, tokenizer_tgt: Tokenizer, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())


            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        print(f"Validation Character Error Rate (CER): {cer:.4f}")

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        print(f"Validation Word Error Rate (WER): {wer:.4f}")

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        print(f"Validation BLEU Score: {bleu:.4f}")

def get_all_sentences(ds, lang):
    for item in ds['train']:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    # tokenizer_path = Path(config['tokenizer_file'].format(lang))
    tokenizer_path = Path("output") / config['tokenizer_file'].format(lang)
    if not Path.exists(tokenizer_path):
        if lang != "zh":
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        else:
            # 初始化一个 WordPiece 分词器
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            # 设置文本的正则化处理，这里使用NFD和去除重音符号作为例子
            tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
            # 使用空白字符进行预分词
            tokenizer.pre_tokenizer = Whitespace()
            # 使用 WordPieceTrainer 进行训练
            trainer = WordPieceTrainer(vocab_size=30000, min_frequency=2, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print("Loading exists tokenizer...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # 因为只有训练集, 所以我们自己分比例
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # ds_raw = load_dataset('parquet', data_files=['data/opus_book/train-00000-of-00001.parquet'])


    # ds_raw = load_from_xml()
    ds_raw = load_from_files()
    
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw['train']))
    val_ds_size = len(ds_raw['train']) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw['train'], [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw['train']:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"output/{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 200 == 0:
                run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
  
        # if epoch % 5 == 0 and epoch !=0:
            
        # Save the model at the end of every epoch
        model_folder = f"output/{config['datasource']}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) >1:
            weights_files.sort()
            os.remove(str(weights_files[0]))
            print(f"{str(weights_files[0])} is deleted.")
            
        model_filename = get_weights_file_path(config, f"{epoch:02d}")   
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        print(f"Save Model to {model_filename}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
