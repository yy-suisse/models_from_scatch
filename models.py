import torch 
import math
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len,d_model).float()
        pe.requires_grad = False #just give the positional info, no need to train on it, save time too
        for pos in range(max_len):   
            for i in range(0, d_model, 2):
            # for each dimension of the each position
                pe[pos,i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(2*i/d_model))) # (max_len, d_model)

        #  include the batch size as the first dimension
        self.pe = pe.unsqueeze(0)

    def forward(self):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, vocab_size, embed_size, seq_len, dropout = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0) # 0 because pad token idx = 0, can be found with tokenizer.pad_token_id
        self.segment = torch.nn.Embedding(3,embed_size,padding_idx=0) # 3 possible numbers for segment
        self.position = PositionalEmbedding(embed_size, seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.segment(segment_label) + self.position()
        return self.dropout(x)
    

class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """