
def load_from_files(num_sample=15000):
    
    with open('data/translation/zh-en/train.tags.zh-en.en', 'r', encoding='utf-8') as en_file:
        english_sentences = en_file.readlines()[6:]
        print(len(english_sentences))
        pass
    
load_from_files()