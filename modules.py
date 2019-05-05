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
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CRFLayer(nn.Module):
    """This module implements a simple CRF layer, modified from https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py"""
    def __init__(self,
                 output_size=5):
        super(CRFLayer, self).__init__()
        self.output_size = output_size

        # In https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py, there're two extra tokens <START> and <END>, 
        # whereas here only the original tokens are used.
        self.transition = nn.Parameter(torch.zeros(output_size, output_size))

    def forward(self, inputs, mask, lengths, labels=None):
        """
        args: 
            inputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
        returns:
            best_path: a 2D tensor, shape of (batch_size, seq_len)
        """
        path_score, best_path = self.decode(inputs, mask, lengths)
        if labels is not None:
            loss = self.NLLLoss(inputs, mask, lengths, labels)
            return path_score, best_path, loss
        else: path_score, best_path

    def calculate_PZ(self,
                     inputs,
                     mask):
        """
        args:
            inputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
            mask: a 2D tensor, shape of (batch_size, seq_len)
        returns:
            total_score: a scalar, the sum of scores of all samples in one batch
            scores: a 4D tensor, shape of (seq_len, batch_size, output_size, output_size)

        NOTE: To understand the code, you can refer to https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        output_size = inputs.size(2)
        n_tokens = seq_len * batch_size

        assert output_size == self.output_size, ("Whoops, there's something wrong with your label dimension.")

        mask = mask.transpose(0, 1).contiguous()                                    # shape of (seq_len, batch_size)
        inputs = inputs.transpose(0, 1).contiguous()                                # shape of (seq_len, batch_size, output_size)
        inputs = inputs.view(n_tokens, 1, output_size).expand(-1, output_size, -1)  # shape of (seq_len * batch_size, output_size, output_size)

        # next we add `inputs` with `transition` to get scores
        # you can do this at every step below, we do it here for efficiency
        # NOTE: at the third line, we subtract `transition` at the first step for every sample in the batch,
        #       this is because at the first step, there's no any transition to it.
        scores = inputs + self.transition.unsqueeze(0).expand(n_tokens, -1, -1)     # shape of (seq_len * batch_size, output_size, output_size)
        scores = scores.view(seq_len, batch_size, output_size, -1)                  # shape of (seq_len, batch_size, output_size, output_size)
        scores[0] = scores[0] - self.transition.unsqueeze(0).expand(batch_size, -1, -1)
                                                                                    # shape of (seq_len, batch_size, output_size, output_size)
        # next we build iterator
        iterator = enumerate(scores)
        partition = torch.zeros(batch_size, output_size, 1).cuda()                  # shape of (batch_size, output_size, 1)

        #
        for i, current_step in iterator:
            # partition: previous results log(sum(exp(*))), shape of (batch_size, output_size, 1)
            # current_step: shape of (batch_size, output_size, output_size)

            current_step = current_step + partition.expand(-1, -1, output_size)     # shape of (batch_size, output_size, output_size)
            current_partition = self.log_sum_exp(current_step)                      # shape of (batch_size, output_size)

            mask_i = mask[i, :].view(batch_size, 1).expand(-1, output_size)         # shape of (batch_size, output_size)
            masked_current_partition = current_partition.masked_select(mask_i)
            mask_i = mask_i.contiguous().unsqueeze(2)                               # shape of (batch_size, output_size, 1)

            partition.masked_scatter_(mask_i, masked_current_partition)             # shape of (batch_size, output_size, 1)

        # at last, we calculate the total scores. 
        total_score = torch.sum(self.log_sum_exp(partition))

        return total_score, scores

    def decode(self, inputs, mask, lengths):
        '''Use Viterbi algorithm to calculate the scores.
        args:
            inputs: a 3D tensor, shape of (batch_size, seq_len, output_size)
            mask: a 2D tensor, shape of (batch_size, seq_len)
            lengths: a 1D tensor, shape of (batch_size, )
        outputs:
            decode_index: the decoded sequence, shape of (batch_size, seq_len)
            path_score(TODO): the corresponding score for each sequence, shape of (batch_size, ) 
        '''
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        output_size = inputs.size(2)
        n_tokens = seq_len * batch_size

        assert output_size == self.output_size, ("Whoops, there's something wrong with your label dimension.")

        mask = mask.transpose(0, 1).contiguous()                                    # shape of (seq_len, batch_size)
        inputs = inputs.transpose(0, 1).contiguous()                                # shape of (seq_len, batch_size, output_size)
        inputs = inputs.view(n_tokens, 1, output_size).expand(-1, output_size, -1)  # shape of (seq_len * batch_size, output_size, output_size)

        # next we add `inputs` with `transition` to get scores
        # you can do this at every step below, we do it here for efficiency
        # NOTE: at the third line, we subtract `transition` at the first step for every sample in the batch,
        #       this is because at the first step, there's no any transition to it.
        scores = inputs + self.transition.unsqueeze(0).expand(n_tokens, -1, -1)    # shape of (seq_len * batch_size, output_size, output_size)
        scores = scores.view(seq_len, batch_size, output_size, -1)                 # shape of (seq_len, batch_size, output_size, output_size)
        scores[0] = scores[0] - self.transition.unsqueeze(0).expand(batch_size, -1, -1)
                                                                                   # shape of (seq_len, batch_size, output_size, output_size)
        # next we build iterator and back checkpoints for the best path
        iterator = enumerate(scores)
        back_points = list()
        partition_history = list()

        # the first step
        partition = torch.zeros(batch_size, output_size, 1).cuda()                  # shape of (batch_size, output_size, 1)

        #
        for i, current_step in iterator:
            current_step = current_step + partition.expand(-1, -1, output_size)     # shape of (batch_size, output_size, output_size)
            partition, current_argmax = torch.max(current_step, dim=1)              # shape of (batch_size, output_size)
            partition_history.append(partition)
            partition = partition.unsqueeze(2)                                      # shape of (batch_size, output_size, 1)

            back_points.append(current_argmax)                                      # shape of (batch_size, output_size)
        
        partition_history = torch.cat(partition_history, dim=0).view(seq_len, batch_size, -1).transpose(0, 1).contiguous()
                                                                                    # shape of (batch_size, seq_len, output_size)
        last_position = lengths.view(batch_size, 1, 1).expand(-1, -1, output_size) - 1 
                                                                                    # shape of (batch_size, 1, output_size)
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, output_size)
                                                                                    # shape of (batch_size, output_size)

        # then, we calculate the indexes of the maximum values
        _, pointer = torch.max(last_partition, dim=1)                               # shape of (batch_size, )
        back_points = torch.cat(back_points).view(seq_len, batch_size, -1)          # shape of (seq_len, batch_size, output_size)

        # now we decode from the last position of each sample
        decode_index = autograd.Variable(torch.LongTensor(seq_len, batch_size)).cuda()
        decode_index[-1] = pointer.detach()
        for i in range(len(back_points) - 1, 0, -1):
            current_mask = mask[i]
            current_pointer = torch.gather(back_points[i], 1, pointer.contiguous().view(batch_size, 1))
            pointer.masked_scatter_(current_mask, current_pointer)
            decode_index[i - 1] = pointer.detach().view(batch_size)

        path_score = None
        decode_index = decode_index.transpose(0, 1)                                 # shape of (batch_seq, seq_len)
        return path_score, decode_index

    def sentence_score(self, scores, mask, labels):
        """
        args:
            scores: a 4D tensor, shape of (seq_len, batch_size, output_size, output_size)
            mask: a 2D tensor, shape of (batch_size, seq_len)
            lengths: a 1D tensor, shape of (batch_size, )
            labels: a 2D tensor, shape of (batch_size, seq_len)
        outputs:
            score: sum of scores for gold sentences within each batch
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        output_size = scores.size(2)

        new_labels = autograd.Variable(torch.LongTensor(batch_size, seq_len)).cuda()

        for i in range(seq_len):
            if i == 0:
                new_labels[:, 0] = labels[:, 0]
            else:
                new_labels[:, i] = labels[:, i - 1] * output_size + labels[:, i]

        new_labels = new_labels.transpose(0, 1).contiguous().view(seq_len, batch_size, 1)                       # shape of (seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_labels).view(seq_len, batch_size) # shape of (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(0, 1))

        gold_score = tg_energy.sum()
        return gold_score
    
    def log_sum_exp(self, x):
        """calculate the log of exp sum
        args:
            x: a 3D tensor, shape of (batch_size, output_size, _output_size)
            NOTE: at most cases, `_output_size` == `output_size`, but not always.
        returns:
            outputs: a 2D tensor, shape of (batch_size, _output_size)
        """
        x_max, _ = torch.max(x, dim=1, keepdim=True)                                         # shape of (batch_size, 1, _output_size)
        logged_x = x_max + torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True))   # shape of (batch_size, 1, _output_size)
        return logged_x.squeeze(1)                                                           # shape of (batch_size, _output_size)

    def NLLLoss(self, inputs, mask, lengths, labels):
        batch_size = inputs.size(0)
        forward_score, scores = self.calculate_PZ(inputs, mask)
        gold_score = self.sentence_score(scores, mask, labels)
        return (forward_score - gold_score) / batch_size

    def __repr__(self):
        return 'CRF_layer(output_size=%d)' % (self.output_size)

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
            self.v = nn.Parameter(torch.zeros(self.hidden_dim))
    
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

            self._2gram_gated_leyer = nn.Conv1d(self.embed_dim, 20, 2)
            self._3gram_gated_leyer = nn.Conv1d(self.embed_dim, 20, 3)
            self._4gram_gated_leyer = nn.Conv1d(self.embed_dim, 20, 4)
            self._5gram_gated_leyer = nn.Conv1d(self.embed_dim, 20, 5)

            self.embed_dim += 40

        if predict_word:
            self.predict_layer_1 = nn.Linear(2 * hidden_dim, 100)
            self.predict_layer_2 = nn.Linear(100, 1)
        
        self.lstm_layer = nn.LSTM(self.embed_dim, hidden_dim, layers, batch_first=True, bidirectional=True, dropout=dropout)

        if use_attention is True:
            self.attention = Attention(2 * hidden_dim)

        self.init_orthogonal(self.lstm_layer)
        if use_CRF:
            self.crf_layer = CRFLayer(4)

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

        if self.use_cnn:   # transpose --> (batch_size, embedding_dim, seq_len) --> *
            # NOTE: if we want to use CNN as well as keeping the size after that, we should manually pad the sequence.
            # In this implementation, we use GLU (Gated Linear Unit)
            padded_2gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=1, right_size=0) 
            padded_3gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=1, right_size=1)
            padded_4gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=2, right_size=1)
            padded_5gram = self.cnn_tensor(embeded.transpose(1, 2), left_size=2, right_size=2)

            embeded_2gram = self._2gram_cnn_layer(padded_2gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            embeded_3gram = self._3gram_cnn_layer(padded_3gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            embeded_4gram = self._4gram_cnn_layer(padded_4gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            embeded_5gram = self._5gram_cnn_layer(padded_5gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)

            gated_2gram = self._2gram_gated_leyer(padded_2gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            gated_3gram = self._3gram_gated_leyer(padded_3gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            gated_4gram = self._4gram_gated_leyer(padded_4gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)
            gated_5gram = self._5gram_gated_leyer(padded_5gram).transpose(1, 2)  # shape of (batch_size, seq_len, 20)

            gated_2gram = torch.mul(torch.sigmoid(gated_2gram), embeded_2gram)   # shape of (batch_size, seq_len, 20)
            gated_3gram = torch.mul(torch.sigmoid(gated_3gram), embeded_3gram)   # shape of (batch_size, seq_len, 20)
            gated_4gram = torch.mul(torch.sigmoid(gated_4gram), embeded_4gram)   # shape of (batch_size, seq_len, 20)
            gated_5gram = torch.mul(torch.sigmoid(gated_5gram), embeded_5gram)   # shape of (batch_size, seq_len, 20)

            embeded = torch.cat((embeded, gated_2gram, gated_3gram), dim=-1)
            
        embeded = self.dropout_layer(embeded)

        packed = pack_padded_sequence(embeded, lengths, batch_first=True)
        hiddens, _ = self.lstm_layer(packed)
        padded, _ = pad_packed_sequence(hiddens, batch_first=True)         # shape of (batch_size, seq_len, 2*hidden_dim)
        prev_layer = padded
        
        if self.use_attention is True:
            prev_layer = self.attention(padded, padded, padded, mask)      # shape of (batch_size, seq_len, 2*hidden_dim)
            
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
