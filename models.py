import torch 
import math
import torch.nn.functional as F

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
    def __init__(self, vocab_size, embed_size, seq_len = 64, dropout = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0) # 0 because pad token idx = 0, can be found with tokenizer.pad_token_id
        self.segment = torch.nn.Embedding(3,embed_size,padding_idx=0) # 3 possible numbers for segment
        self.position = PositionalEmbedding(embed_size, seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.segment(segment_label) + self.position()
        return self.dropout(x)
    
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, d_model, heads,dropout = 0.1):
        super().__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, mask, value):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, max_len, max_len)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads,self.d_k).permute(0,2,1,3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        # (batch_size, h, max_len, d_k) * (batch_size, h, d_k, max_len) == (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0,1,3,2))/math.sqrt(query.size(-1))
        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9) 
        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)
        # (batch_size, h, max_len, max_len) * (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights,value)
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0,2,1,3).contiguous().view(context.shape[0],-1,self.heads * self.d_k)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model,middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim,d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out # torch.Size([batch size, max len, d_model])

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model=768,heads = 12, feed_forward_hidden = 4*768, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(d_model, heads)
        self.feedforward = FeedForward(d_model,middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(query = embeddings, key = embeddings, value = embeddings, mask = mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feedforward(interacted))
        encoded = self.layernorm(interacted + feed_forward_out)
        return encoded



class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4
        self.embedding = BERTEmbedding(vocab_size, d_model)
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])
        
    def forward(self, x,segment_info):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) # this is only for masking paddings
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
class NextSentencePrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """
    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0]))
    
class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)    

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)