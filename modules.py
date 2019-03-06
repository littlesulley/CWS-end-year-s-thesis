"""
This file implements several common modules.
@ author: Qinghong Han
@ date: Feb 22nd, 2019
@ contact: qinghong_han@shannonai.com
"""

import numpy as np  
import pandas as pd 
import os 
import sys
import itertools

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



class CRFLayer(nn.Module):
    """This module implements a simple CRF layer."""
    def __init__(self,
                 output_size=5,
                 start_tag=0,
                 end_tag=6):
        super(CRFLayer, self).__init__()
        self.output_size = output_size
        self.start_tag = start_tag
        self.end_tag = end_tag

        self.transition = nn.Parameter(torch.zeros(output_size + 2, output_size + 2))
        self.transition.data[start_tag, :] = -100000.
        self.transition.data[:, end_tag] = -100000.

    def forward(self,
                inputs,
                mask,
                lengths):
        """
        args:
            inputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
            mask: a 2D tensor, shape of (batch_size, seq_len)
            lengths: a 1D tensor, shape of (batch_size, )
        returns:
            outputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
        """
        # TODO 
        pass


    def decode(self, inputs, mask):
        '''Use Viterbi algorithm to calculate the scores.'''
        #TODO
        pass

        


    def log_sum_exp(self, x):
        x_max = torch.max(x, dim=-1)
        return x_max + torch.log(torch.sum(torch.exp(x - x_max), dim=-1))

class Attention(nn.Module):
    """This module implements attention."""
    def __init__(self, 
                 hidden_dim, 
                 attention_type='dot', 
                 is_scaled=False,
                 **kwargs):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        self.is_scaled = is_scaled

        if attention_type not in ['dot', 'general', 'concat', 'self']:
            raise ValueError('Attention type %s is not appropriate.' % attention_type)
        
        if attention_type == 'dot' or attention_type == 'self':
            pass
        elif attention_type == 'general':
            self.attn = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.attn = nn.Linear(2 * hidden_dim, hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_dim))
    
    def forward(self,
                queries,
                keys,
                values, 
                key_mask=None):
        """
        args:
            queries: 3D tensor, shape of (batch_size, query_len, hidden_dim)
            keys: 3D tensor, shape of (batch_size, key_len, hidden_dim)
            values: 3D tensor, shape of (batch_size, value_len, hidden_dim)
            key_mask: 2D tensor, shape of (batch_size, key_len)
        returns:
            a 3D tensor, shape of (batch_size, query_len, hidden_dim)
        """
        query_len = queries.size(1)
        if key_mask is None:
            key_mask = torch.ones(keys.size()[:2], dtype=torch.uint8)
        mask = key_mask.unsqueeze(1).expand(-1, query_len, -1)  # shape of (batch_size, query_len, key_len)

        if self.attention_type == 'dot' or self.attention_type == 'self':
            return self.dot_score(queries, keys, values, mask)
        elif self.attention_type == 'general':
            return self.general_score(queries, keys, values, mask)
        else: 
            return self.concat_score(queries, keys, values, mask)

    def dot_score(self, queries, keys, values, mask):
        attended = torch.bmm(queries, keys.transpose(1, 2))   # shape of (batch_size, query_len, key_len)
        attended = attended.masked_fill(1 - mask, -np.inf)   # shape of (batch_size, query_len, key_len)
        if self.is_scaled:
            attended = attended / np.sqrt(self.hidden_dim)
        attended = F.softmax(attended, dim=-1)                # shape of (batch_size, query_len, key_len)
        scores = torch.bmm(attended, values)                  # shape of (batch_size, query_len, hidden_dim)
        return scores

    def general_score(self, queries, keys, values, mask):
        attended = torch.bmm(self.attn(queries), keys.transpose(1, 2))   # shape of (batch_size, query_len, key_len)
        attended = attended.masked_fill(1 - mask, -np.inf)               # shape of (batch_size, query_len, key_len)
        if self.is_scaled:
            attended = attended / np.sqrt(self.hidden_dim)
        attended = F.softmax(attended, dim=-1)                # shape of (batch_size, query_len, key_len)
        scores = torch.bmm(attended, values)                  # shape of (batch_size, query_len, hidden_dim)
        return scores

    def concat_score(self, queries, keys, values, mask):
        NotImplementedError

class MultiHeadBlock(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 heads=8):
        super(MultiHeadBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_per_head = hidden_dim // heads

        self.transition = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attention = Attention(self.dim_per_head)
        self.layernorm_layer = nn.LayerNorm(hidden_dim)

    def forward(self,   
                inputs, 
                mask):
        '''
        args:
            inputs: a 3D tensor, shape of (batch_size, seq_len, hidden_dim)
            mask: a 2D tensor, shape of (batch_size, seq_len)
        returns:
            outputs: a 3D tensor, shape of (batch_size, seq_len, hidden_dim)
        '''
        assert inputs.size()[:2] == mask.size(), ("Whoops, the dimensions don't match.")

        batch_size = inputs.size(0)

        transitted = self.transition(inputs) # shape of (batch_size, seq_len, 3 * hidden_dim)
        Q = transitted[:, :, : self.hidden_dim]
        K = transitted[:, :, self.hidden_dim: 2 * self.hidden_dim]
        V = transitted[:, :, 2 * self.hidden_dim:]

        Q = Q.contiguous().view(batch_size * self.heads, -1, self.dim_per_head) # shape of (batch_size*heads, seq_len, hidden_dim/heads)
        K = K.contiguous().view(batch_size * self.heads, -1, self.dim_per_head) # shape of (batch_size*heads, seq_len, hidden_dim/heads)
        V = V.contiguous().view(batch_size * self.heads, -1, self.dim_per_head) # shape of (batch_size*heads, seq_len, hidden_dim/heads)

        # NOTE: at the next step, Q/K/V is shape of (batch_size * heads, seq_len, hidden_dim / heads),
        #       while mask is shape of (batch_size, seq_len)
        #       so we need to expand the shape to (batch_size * heads, seq_len)
        mask = mask.repeat(self.heads, 1)                          # shape of (batch_size * heads, seq_len)
        attended = self.attention(Q, K, V, mask)                   # shape of (batch_size*heads, seq_len, hidden_dim/heads)
        attended = attended.view(batch_size, -1, self.hidden_dim)  # shape of (batch_size, seq_len, hidden_dim)

        outputs = self.layernorm_layer(inputs + attended)          # shape of (batch_size, seq_len, hidden_dim)
        return outputs

class MultiHeadFFBlock(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 heads=8,
                 dropout=0.2,
                 ff_dim=[1024, 512]):
        super(MultiHeadFFBlock, self).__init__()
        
        self.dropout_layer = nn.Dropout(dropout)
        self.multihead_layer = MultiHeadBlock(hidden_dim=hidden_dim, heads=heads)
        self.layernorm_layer = nn.LayerNorm(hidden_dim)
        self.ff_layer_1 = nn.Linear(hidden_dim, ff_dim[0])
        self.ff_layer_2 = nn.Linear(ff_dim[0], ff_dim[1])

    def forward(self,
                inputs, 
                mask):
        '''
        args:
            inputs: a 3D tensor, shape of (batch_size, seq_len, hidden_dim)
            mask: a 2D tensor, shape of (batch_size, seq_len)
        returns:
            outputs: a 3D tensor, shape of (batch_size, seq_len, hidden_dim)
        '''
        multiheaded = self.multihead_layer(inputs, mask) # shape of (batch_size, seq_len, hidden_dim)
        multiheaded = self.dropout_layer(multiheaded)    # shape of (batch_size, seq_len, hidden_dim)
        outputs = self.ff_layer_2(self.dropout_layer(F.relu(self.ff_layer_1(multiheaded))))
                                                         # shape of (batch_size, seq_len, hidden_dim)
        outputs = self.layernorm_layer(multiheaded + outputs)   # shape of (batch_size, seq_len, hidden_dim)
        return outputs

class LanguageModel(nn.Module):
    """This module implements a basic language model."""
    def __init__(self,
                 vocab_size,
                 embed_dim=300,
                 hidden_dim=256,
                 layers=2,
                 max_len=35,
                 dropout=0.2,
                 **kwargs):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers 
        self.max_len = max_len

        self.dropout_layer = nn.Dropout(dropout)
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.lstm_layer = nn.LSTM(embed_dim, hidden_dim, layers=layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(2 * hidden_dim, vocab_size)
        self.init_orthogonal(self.lstm_layer.weight)

    def forward(self, 
                inputs, 
                lengths):
        """
        args:
            inputs: shape of (batch_size, seq_len)
            lengths: shape of (batch_size, )
        returns:
            outputs: shape of (batch_size, seq_len, vocab_size)
        """
        embeded = self.embedding_layer(inputs)
        embeded = self.dropout_layer(embeded)
        packed = pack_padded_sequence(embeded, lengths, batch_first=True)
        hiddens, _ = self.lstm_layer(packed)  
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)  # shape of (batch_size, seq_len, 2*hidden_dim)
        outputs = self.output_layer(hiddens)
        return outputs               # shape of (batch_size, seq_len, vocab_size)

    def init_orthogonal(self, tensor):
        nn.init.orthogonal_(tensor)

class BiLstmClassifier(nn.Module):
    """This module implements a simple BiLSTM->Pooling model for classification."""
    def __init__(self, 
                 output_size,
                 vocab_size=None,
                 embed_dim=300,
                 pretrained_embedding=None,
                 layers=2,
                 hidden_dim=256,
                 dropout=0.2,
                 transition_type='concat',
                 pooling_type='max',
                 **kwargs):
        super(BiLstmClassifier, self).__init__()
        if vocab_size is None:
            if pretrained_embedding is None:
                raise ValueError('You should either provide `vocab_size` or `pretrained_embedding`.')
            else:
                self.embedding = pretrained_embedding
                self.vocab_size = pretrained_embedding.size(0)
                self.embed_dim = pretrained_embedding.size(1)      
        else:
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.embedding = nn.Linear(vocab_size, embed_dim)
        
        if transition_type not in ['concat', 'sum']:
            raise ValueError('You should choose either `concat` or `sum` for trainsition_type.')

        if pooling_type not in ['max', 'avg']:
            raise ValueError('You should choose either `max` or `avg` for pooling_type')

        self.layers = layers 
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.transition_type = transition_type
        self.pooling_type = pooling_type
        self.lstm_layer = nn.LSTM(self.embed_dim, hidden_dim, bidirectional=True, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(2 * hidden_dim if transition_type == 'concat' else hidden_dim, output_size)


    def forward(self,
                inputs,
                lengths):
        """
        args:
            inputs: a 2D tensor, shape of (batch_size, seq_len)
            length: a 1D tensor, shape of (batch_size,)
        returns:
            outputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
        """
        seq_len = inputs.size(1)
        embeded = self.embedding(inputs)
        packed = pack_padded_sequence(embeded, lengths, batch_first=True)
        hiddens, _ = self.lstm_layer(packed)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)   # shape of (batch_size, seq_len, 2*hidden_dim)
        
        if self.transition_type == 'sum':
            hiddens = hiddens[:, :, :self.hidden_dim] + hiddens[:, :, self.hidden_dim]
        
        hiddens = hiddens.transpose(1, 2)                # shape of (batch_size, (2*)hidden_dim, seq_len)
        
        if self.pooling_type == 'max':
            hiddens = F.max_pool1d(hiddens, seq_len).squeeze(2)
        else:
            hiddens = F.avg_pool1d(hiddens, seq_len).squeeze(2)
        
        # now, hidden is shape of (batch_size, (2*)hidden_dim)
        outputs = self.output_layer(hiddens)  # shape of (batch_size, output_size)
        return outputs

class CWSLstm(nn.Module):
    def __init__(self,
                layers=2,
                hidden_dim=256,
                output_size=4,
                embed_dim=300,
                vocab_size=None,
                pretrained_embedding=None,
                dropout=0.2,
                use_CRF=False,
                oov_window=0,
                predict_word=False,
                use_attention=False,
               **kwargs):
        super(CWSLstm, self).__init__()
        if vocab_size is None:
            if pretrained_embedding is None:
                raise ValueError('You should either provide `vocab_size` or `pretrained_embedding`.')
            else:
                self.embedding = pretrained_embedding
                self.vocab_size = pretrained_embedding.size(0)
                self.embed_dim = pretrained_embedding.size(1)      
        else:
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.layers = layers 
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.use_CRF = use_CRF
        self.oov_window = oov_window
        self.predict_word = predict_word
        self.use_attention = use_attention

        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(2 * hidden_dim, output_size)

        if predict_word:
            self.predict_layer_1 = nn.Linear(2 * hidden_dim, 100)
            self.predict_layer_2 = nn.Linear(100, 1)
        
        if use_attention is False:
            self.lstm_layer = nn.LSTM(self.embed_dim, hidden_dim, layers, batch_first=True, bidirectional=True, dropout=dropout)
        else:
            self.lstm_layer = nn.ModuleList(nn.LSTM(self.embed_dim if _ == 0 else 2 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True) for _ in range(layers))
            self.attentions = nn.ModuleList(MultiHeadFFBlock(hidden_dim=2*hidden_dim, heads=2, dropout=dropout, ff_dim=[4*hidden_dim, 2*hidden_dim]) for _ in range(layers))

        self.init_orthogonal(self.lstm_layer)

        if use_CRF:
            #TODO: implement CRF layer module.
            pass 

    def forward(self,
                inputs,
                lengths, 
                mask):
        """
        args:
            inputs: a 2D tensor, shape of (batch_size, seq_len)
            lengths: a 1D tensor, shape of (batch_size,)
        returns:
            outputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
        """
        embeded = self.embedding(inputs)   # shape of (batch_size, seq_len, embedding_dim)
        if self.oov_window != 0:
            embeded = self.predict_unk(inputs, lengths, embeded, self.oov_window)  # shape of (batch_size, seq_len, embedding_dim)
        prev_layer = self.dropout_layer(embeded)

        if self.use_attention is False:
            packed = pack_padded_sequence(prev_layer, lengths, batch_first=True)
            hiddens, _ = self.lstm_layer(packed)
            padded, _ = pad_packed_sequence(hiddens, batch_first=True) # shape of (batch_size, seq_len, 2*hidden_dim)
            outputs = self.output_layer(padded)                        # shape of (batch_size, seq_len, output_size)
        else:
            for i in range(self.layers):
                packed = pack_padded_sequence(prev_layer, lengths, batch_first=True)
                hiddens, _ = self.lstm_layer[i](packed)
                padded, _ = pad_packed_sequence(hiddens, batch_first=True) # shape of (batch_size, seq_len, 2*hidden_dim)
                prev_layer = self.attentions[i](padded, mask)              # shape of (batch_size, seq_len, 2*hidden_dim)
                if i != self.layers - 1:
                    prev_layer = self.dropout_layer(prev_layer)            # shape of (batch_size, seq_len, 2*hidden_dim)
            
            outputs = self.output_layer(prev_layer)                        # shape of (batch_size, seq_len, output_size)

        if self.predict_word:
            outputs_word = self.predict_layer_2(F.relu(self.dropout_layer(self.predict_layer_1(padded)))).squeeze(2) # shape of (batch_size, seq_len)
            return outputs, outputs_word
        else: return outputs

    def init_orthogonal(self, lstm_layer):
        if isinstance(lstm_layer, nn.ModuleList):
            for layer in lstm_layer:
                for params in layer.all_weights:
                    for param in params:
                        if param.ndimension() >= 2:
                            nn.init.orthogonal_(param)
        else:
            for params in lstm_layer.all_weights:
                for param in params:
                    if param.ndimension() >= 2:
                        nn.init.orthogonal_(param)

    def predict_unk(self, inputs, lengths, embeded, n_window=2):
        batch_size = inputs.size(0)

        for current_batch in range(batch_size):
            current_batch_len = lengths[current_batch]
            for current_char in range(current_batch_len):
                if inputs[current_batch][current_char] != 1: # <UNK> is 1
                    continue  
                if current_char == 0:
                    left = 0
                    right = (current_batch_len - 1 if current_char + n_window >= current_batch_len else current_char + n_window)
                    n_neighbors = right - left
                elif current_batch == current_batch_len - 1:
                    right = current_batch_len - 1
                    left = (0 if current_char - n_window < 0 else current_char - n_window)
                    n_neighbors = right - left
                else:
                    left = (0 if current_char - n_window < 0 else current_char - n_window)
                    right = (current_batch_len - 1 if current_char + n_window >= current_batch_len else current_char + n_window)
                    n_neighbors = right - left
                
                if n_neighbors != 0:
                    if current_char != current_batch_len - 1 and current_char != 0:
                        embeded[current_batch][current_char] = 1. / n_neighbors * \
                                (torch.sum(embeded[current_batch][left: current_char], dim=0) + torch.sum(embeded[current_batch][current_char+1: right+1], dim=0))
                    elif current_char == current_batch_len - 1:
                        embeded[current_batch][current_char] = 1. / n_neighbors * torch.sum(embeded[current_batch][left: current_char], dim=0)
                    else:
                        embeded[current_batch][current_char] = 1. / n_neighbors * torch.sum(embeded[current_batch][current_char+1: right+1], dim=0)
        
        return embeded