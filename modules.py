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
                values):
        """
        args:
            queries: 3D tensor, shape of (batch_size, query_len, hidden_dim)
            keys: 3D tensor, shape of (batch_size, key_len, hidden_dim)
            values: 3D tensor, shape of (batch_size, value_len, hidden_dim)
            In fact, `key_len` == `value_len`
        returns:
            a 3D tensor, shape of (batch_size, query_len, hidden_dim)
        """
        if self.attention_type == 'dot' or self.attention_type == 'self':
            return self.dot_score(queries, keys, values)
        elif self.attention_type == 'general':
            return self.general_score(queries, keys, values)
        else: 
            return self.concat_score(queries, keys, values)

    def dot_score(self, queries, keys, values):
        attended = torch.bmm(queries, keys)                   # shape of (batch_size, query_len, key_len)
        if self.is_scaled:
            attended = attended / np.sqrt(self.hidden_dim)
        attended = F.softmax(attended, dim=-1)                # shape of (batch_size, query_len, key_len)
        scores = torch.bmm(attended, values)                  # shape of (batch_size, query_len, hidden_dim)
        return scores

    def general_score(self, queries, keys, values):
        attended = torch.bmm(self.attn(queries), keys)        # shape of (batch_size, query_len, key_len)
        scores = torch.bmm(attended, values)
        return scores

    def concat_score(self, queries, keys, values):
        raise NotImplementedError

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
        batch_size = inputs.size(0)
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

        self.dropout_layer = nn.Dropout(dropout)
        self.lstm_layer = nn.LSTM(self.embed_dim, hidden_dim, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.output_layer = nn.Linear(2 * hidden_dim, output_size)
        if use_CRF:
            #TODO: implement CRF layer module.
            pass 

    def forward(self,
                inputs,
                lengths):
        """
        args:
            inputs: a 2D tensor, shape of (batch_size, seq_len)
            lengths: a 1D tensor, shape of (batch_size,)
        returns:
            outputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        embeded = self.embedding(inputs)   # shape of (batch_size, seq_len, embedding_dim)
        embeded = self.dropout_layer(embeded)
        packed = pack_padded_sequence(embeded, lengths, batch_first=True)
        hiddens, _ = self.lstm_layer(packed)
        padded, _ = pad_packed_sequence(hiddens, batch_first=True) # shape of (batch_size, seq_len, 2*hidden_dim)
        outputs = self.output_layer(padded)  # shape of (batch_size, seq_len, output_size)
        if self.use_CRF:
            #TODO
            pass

        return outputs