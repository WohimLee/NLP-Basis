from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from pathlib import Path
import jieba

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path("output") / config['tokenizer_file'].format(lang)
    
    if not Path.exists(tokenizer_path):
        # Use a Byte-Pair Encoding tokenizer (or any other suitable for Chinese)
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # For Chinese, we'll use jieba to segment the text before tokenizing
        def jieba_pre_tokenizer(sentence):
            return list(jieba.cut(sentence))
        
        # Add normalizer for handling Chinese-specific normalization, if necessary
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(jieba_pre_tokenizer)
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer