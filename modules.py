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

    def __repr__(self):
        return 'Attention(dimension=%d)' % (self.hidden_dim)

class MultiHeadBlock(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 heads=8,
                 dropout=None):
        super(MultiHeadBlock, self).__init__()
        assert hidden_dim % heads == 0, ("Whoops, `hidden_dim` should be exactly divided by `heads` T_T")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dim_per_head = hidden_dim // heads
        self.dropout = dropout

        self.transition = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attention = Attention(self.dim_per_head)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
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
        assert inputs.size()[:2] == mask.size(), ("Whoops, dimensions between `inputs` and `mask` don't match.")

        batch_size = inputs.size(0)

        transitted = self.transition(inputs)                                   # shape of (batch_size, seq_len, 3 * hidden_dim)
        Q = transitted[:, :, : self.hidden_dim]                                # shape of (batch_size, seq_len, hidden_dim)
        K = transitted[:, :, self.hidden_dim: 2 * self.hidden_dim]             # shape of (batch_size, seq_len, hidden_dim)
        V = transitted[:, :, 2 * self.hidden_dim:]                             # shape of (batch_size, seq_len, hidden_dim)

        # NOTE: at the next step, you should be careful that, you CAN'T don `Q.view(batch_size * self.heads, -1, self.dim_per_head)`
        #       This is very subtle, so you need to be advertent.

        # First, resize to (batch_size, seq_len, heads, hidden_dim/heads)
        Q = Q.contiguous().view(batch_size, -1, self.heads, self.dim_per_head) # shape of (batch_size, seq_len, heads, hidden_dim/heads)
        K = K.contiguous().view(batch_size, -1, self.heads, self.dim_per_head) # shape of (batch_size, seq_len, heads, hidden_dim/heads)
        V = V.contiguous().view(batch_size, -1, self.heads, self.dim_per_head) # shape of (batch_size, seq_len, heads, hidden_dim/heads)

        # Second, transpose to (batch_size, heads, seq_len, hidden_dim/heads)
        Q = Q.transpose(1, 2)                                                  # shape of (batch_size, heads, seq_len, hidden_dim/heads)
        K = K.transpose(1, 2)                                                  # shape of (batch_size, heads, seq_len, hidden_dim/heads)
        V = V.transpose(1, 2)                                                  # shape of (batch_size, heads, seq_len, hidden_dim/heads)

        # Third, merge to (batch_size*heads, seq_len, hidden_dim/heads)
        Q = Q.contiguous().view(batch_size * self.heads, -1, self.dim_per_head) # shape of (batch_size*heads, seq_len, hidden_dim/heads)
        K = K.contiguous().view(batch_size * self.heads, -1, self.dim_per_head) # shape of (batch_size*heads, seq_len, hidden_dim/heads)
        V = V.contiguous().view(batch_size * self.heads, -1, self.dim_per_head) # shape of (batch_size*heads, seq_len, hidden_dim/heads)

        # Last, go through attenton layer
  
        # NOTE: at the next step, Q/K/V is shape of (batch_size * heads, seq_len, hidden_dim / heads),
        #       while mask is shape of (batch_size, seq_len)
        #       so we need to expand the shape to (batch_size * heads, seq_len)
        mask = mask.repeat(self.heads, 1)                                      # shape of (batch_size * heads, seq_len)
        attended = self.attention(Q, K, V, mask)                               # shape of (batch_size*heads, seq_len, hidden_dim/heads)

        # Now, we reverse the procedure that we did above
        attended = attended.contiguous().view(batch_size, self.heads, -1, self.dim_per_head).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
                                                                               # shape of (batch_size, seq_len, hidden_dim)

        if self.dropout:
            outputs = F.dropout(attended, 0.2)
        outputs = self.linear(outputs)
        outputs = self.layernorm_layer(outputs + attended)                     # shape of (batch_size, seq_len, hidden_dim)
        return outputs

class MultiHeadFFBlock(nn.Module):
    def __init__(self,
                 hidden_dim=512,
                 heads=8,
                 dropout=0.2,
                 ff_dim=[1024, 512]):
        super(MultiHeadFFBlock, self).__init__()
        
        self.dropout_layer = nn.Dropout(dropout)
        self.multihead_layer = MultiHeadBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
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
        outputs = self.dropout_layer(multiheaded)        # shape of (batch_size, seq_len, hidden_dim)
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
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
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
                use_cnn=False,
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
        self.use_cnn = use_cnn
        self.oov_window = oov_window
        self.predict_word = predict_word
        self.use_attention = use_attention

        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(2 * hidden_dim, output_size)

        if use_cnn:
            self._2gram_cnn_layer = nn.Conv1d(self.embed_dim, 20, 2)
            self._3gram_cnn_layer = nn.Conv1d(self.embed_dim, 20, 3)
            self._4gram_cnn_layer = nn.Conv1d(self.embed_dim, 20, 4)
            self._5gram_cnn_layer = nn.Conv1d(self.embed_dim, 20, 5)

            self.embed_dim += 80

        if predict_word:
            self.predict_layer_1 = nn.Linear(2 * hidden_dim, 100)
            self.predict_layer_2 = nn.Linear(100, 1)
        
        self.lstm_layer = nn.LSTM(self.embed_dim, hidden_dim, layers, batch_first=True, bidirectional=True, dropout=dropout)

        if use_attention is True:
            self.transition_1 = nn.Linear(10 * hidden_dim, 4 * hidden_dim)
            self.transition_2 = nn.Linear(4 * hidden_dim, 2 * hidden_dim)

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
        embeded = self.dropout_layer(embeded)

        if self.use_cnn:   # transpose --> (batch_size, embedding_dim, seq_len) --> *
            embeded_2gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=1, right_size=0) 
            embeded_3gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=1, right_size=1)
            embeded_4gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=2, right_size=1)
            embeded_5gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=2, right_size=2)

            embeded_2gram = self._2gram_cnn_layer(embeded_2gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            embeded_3gram = self._3gram_cnn_layer(embeded_3gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            embeded_4gram = self._4gram_cnn_layer(embeded_4gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            embeded_5gram = self._5gram_cnn_layer(embeded_5gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)

            embeded = torch.cat((embeded, embeded_2gram, embeded_3gram, embeded_4gram, embeded_5gram), dim=-1)

        packed = pack_padded_sequence(embeded, lengths, batch_first=True)
        hiddens, _ = self.lstm_layer(packed)
        padded, _ = pad_packed_sequence(hiddens, batch_first=True) # shape of (batch_size, seq_len, 2*hidden_dim)
        prev_layer = padded
        
        if self.use_attention is True:
            padded_left = self.shift_tensor(padded, -1)      # shape of (batch_size, seq_len, 2*hidden_dim)
            padded_right = self.shift_tensor(padded, 1)      # shape of (batch_size, seq_len, 2*hidden_dim)

            minus_left = torch.abs(padded - padded_left)     # shape of (batch_size, seq_len, 2*hidden_dim)
            minus_right = torch.abs(padded_right - padded)   # shape of (batch_size, seq_len, 2*hidden_dim)
            multi_left = torch.mul(padded, padded_left)      # shape of (batch_size, seq_len, 2*hidden_dim)
            multi_right = torch.mul(padded, padded_right)    # shape of (batch_size, seq_len, 2*hidden_dim)

            features = torch.cat((padded, minus_left, minus_right, multi_left, multi_right), dim=-1) # shape of (batch_size, seq_len, 10*hidden_dim)
            transitted_1 = torch.tanh(self.transition_1(self.dropout_layer(features)))               # shape of (batch_size, seq_len, 4*hidden_dim)
            transitted_2 = self.transition_2(self.dropout_layer(transitted_1))                       # shape of (batch_size, seq_len, 2*hidden_dim)

            prev_layer = transitted_2
            
        outputs = self.output_layer(prev_layer)

        if self.predict_word:
            outputs_word = self.predict_layer_2(F.relu(self.dropout_layer(self.predict_layer_1(prev_layer)))).squeeze(2) # shape of (batch_size, seq_len)
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

    def shift_tensor(self, tensor, bias=1):
        '''`tensor` is a 3D tensor, shape of (batch_size, seq_len, hidden_dim)'''
        batch_size = tensor.size(0)
        hidden_dim = tensor.size(2)
        append_tensor = torch.zeros(batch_size, abs(bias), hidden_dim).cuda()
        if bias > 0:
            return torch.cat((tensor[:, bias:, :], append_tensor), dim=1)
        else:
            return torch.cat((append_tensor, tensor[:, :bias, :]), dim=1)

    def cnn_tensor(self, tensor, left_size=1, right_size=1):
        '''`tensor` is of shape (batch_size, hidden_dim, seq_len)'''
        batch_size = tensor.size(0)
        hidden_dim = tensor.size(1)
        left_added_tensor = torch.zeros(batch_size, hidden_dim, left_size).cuda()   # shape of (batch_size, hidden_dim, left_size)
        right_added_tensor = torch.zeros(batch_size, hidden_dim, right_size).cuda() # shape of (batch_size, hidden_dim, right_size)
        
        return torch.cat((left_added_tensor, tensor, right_added_tensor), dim=-1)   # shape of (batch_size, hidden_dim, left_size + seq_len + right_size)