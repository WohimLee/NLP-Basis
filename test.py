from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from datasets import Dataset, load_dataset

import sys
sys.path.append('/root/datav/nlp/NLP-Basis/4.Transformer/src')

from config import get_config

def load_vocab_from_file(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            vocab[token] = idx
    return vocab

def get_or_build_tokenizer(config, ds, lang):
    # 根据语言选择词典文件
    vocab_file = "src_vocab.txt" if lang == "src" else "trg_vocab.txt"
    
    tokenizer_path = Path("output") / config['tokenizer_file'].format(lang)
    
    if not Path.exists(tokenizer_path):
        # 加载词典
        vocab = load_vocab_from_file(vocab_file)
        
        # 使用词典初始化Tokenizer
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # 训练器不再需要，因为我们已经有了词典
        # trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # 保存Tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

if __name__ == '__main__':
    # config = get_config()
    
    # tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    # tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 加载自定义数据集
    # 读取两个文件
    with open("data/translation_en2zh/train.en", "r", encoding="utf-8") as f_en:
        en_sentences = f_en.readlines()

    with open("data/translation_en2zh/train.zh", "r", encoding="utf-8") as f_zh:
        zh_sentences = f_zh.readlines()

    # 确保两个文件的行数相同
    assert len(en_sentences) == len(zh_sentences), "两个文件的行数不匹配！"
    
    # 创建数据集字典
    data_dict = {"en": en_sentences, "zh": zh_sentences}

    # 将字典转换为 Huggingface Dataset
    dataset = Dataset.from_dict(data_dict)
    
    # data_files = {"train": {"en": "data/translation_en2zh/train.en", "zh": "data/translation_en2zh/train.zh"}}
    # dataset = load_dataset("text", data_files=data_files, split="train")

    # 查看前几条数据
    for example in dataset.select(range(3)):
        print("English:", example["en"].strip())
        print("Chinese:", example["zh"].strip())
        print("\n")
