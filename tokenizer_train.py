from tokenizers import BertWordPieceTokenizer
import os
from transformers import BertTokenizer

def train_bert_tokenizer(txt_files, save_name):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train( 
        files=txt_files,
        vocab_size=30_000, 
        min_frequency=5,
        limit_alphabet=1000, 
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )
    save_path =  './bert_tokenizer'
    os.mkdir(save_path)
    tokenizer.save_model(save_path, save_name)
    tokenizer = BertTokenizer.from_pretrained(save_path +'/' + save_name + '-vocab.txt', local_files_only=True)

    return tokenizer