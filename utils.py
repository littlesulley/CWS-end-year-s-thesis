"""
This file implements some useful functions and classes.
@ author:        Qinghong Han
@ date:          Feb 22nd, 2019
@ contact:       qinghong_han@shannonai.com
@ modified date: Mar 2nd. 2019
"""

import math
import random
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import sys
import re
import csv
import itertools
from collections import Counter
import collections

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset 

plt.style.use('seaborn-whitegrid')
sns.set_style("white")

class Vocab:
    def __init__(self, 
                 vocab_name='default_vocab',
                 max_size=50000,
                 min_count=1,
                 level='char'):
        self.name = vocab_name
        self.max_size = max_size
        self.min_count = min_count
        self.level = level
        self.counter = Counter()

        self.id2word = {0:"<PAD>", 1:"<UNK>", 2:"<START>", 3:"<END>"}
        self.word2id = {"<PAD>":0, "<UNK>":1, "<START>":2, "<END>":3}
        self.vocab_size = 4

    def __repr__(self):
        print("Vocabulary %s, contains words at maximum of %d." % (self.name, self.max_size))

    def __len__(self):
        return self.vocab_size
    
    def add_word(self, word):
        if word not in self.counter:
            self.counter[word] = 1
        else:
            self.counter[word] += 1

    def add_sentence(self, sentence):
        sentence = sentence.strip().lower()
        if self.level == 'word':
            words = sentence.split()
        else:
            words = sentence
        for word in words:
            self.add_word(word)

    def add_file(self, in_file):
        with open(in_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                self.add_sentence(line)

    def word_to_index(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<UNK>']
    
    def save_vocab(self, path):
        words = list(self.word2id.keys())
        save_words = '\n'.join(words)
        with open(path, 'w', encoding='utf-8') as fout:
            fout.write(save_words)

    @classmethod
    def from_file(self, in_file):
        self.id2word = {0:"<PAD>", 1:"<UNK>", 2:"<START>", 3:"<END>"}
        self.word2id = {"<PAD>":0, "<UNK>":1, "<START>":2, "<END>":3}
        self.vocab_size = 4

        with open(in_file, 'r', encoding='utf-8') as f:
            for word in f:
                word = word.strip()
                if word not in self.word2id:
                    self.word2id[word] = self.vocab_size
                    self.id2word[self.vocab_size] = word
                    self.vocab_size += 1

    def trim(self):
        self.id2word = {0:"<PAD>", 1:"<UNK>", 2:"<START>", 3:"<END>"}
        self.word2id = {"<PAD>":0, "<UNK>":1, "<START>":2, "<END>":3}
        self.vocab_size = 4
        
        words = self.counter.most_common()
        words = list(itertools.takewhile(lambda x: x[1] >= self.min_count, words))
        
        if len(words) > self.max_size:
            words = words[:self.max_size]

        for word in words:
            self.word2id[word[0]] = self.vocab_size
            self.id2word[self.vocab_size] = word[0]
            self.vocab_size += 1

class CWSDataset(Dataset):
    def __init__(self, file, type='PKU', mode='pair', sort=True):
        super(CWSDataset, self).__init__()
        assert type in ['PKU', 'AS', 'City', 'MS'], ("Make sure the dataset type is one of ['PKU', 'AS', 'City', 'MS'].")
        self.type = type

        with open(file, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
            data_lines = [line.strip() for line in data_lines]
        if sort:
            data_lines = sorted(data_lines, key=len, reverse=True)

        if mode == 'pair':
            Y_lines = [self.get_sentence_tags(line) for line in data_lines if len(line) > 0]
            X_lines = [self.get_sentence_chars(line) for line in data_lines if len(line) > 0]
            self.lines = list(zip(X_lines, Y_lines))
            for sample in self.lines:
                assert len(sample[0]) == len(sample[1]), ('Whoops, something goes wrong.')
        
        else:
            self.lines = [line for line in data_lines if len(line) > 0]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def get_sentence_tags(self, sentence):
        sentence_len = len(sentence)
        tags = ""
        prev_space = 1  # whether the previous character is a space
        for i in range(sentence_len):
            if sentence[i] == (' ' if self.type != 'AS' else '\u3000'):
                prev_space = 1
            else:
                if prev_space == 1:
                    if i == sentence_len - 1 or sentence[i + 1] == (' ' if self.type != 'AS' else '\u3000'):
                        tags += 'S'
                    else:
                        tags += 'B'
                else:
                    if i == sentence_len - 1 or sentence[i + 1] == (' ' if self.type != 'AS' else '\u3000'):
                        tags += 'E'
                    else:
                        tags += 'I'
                prev_space = 0
        return tags

    def get_sentence_chars(self, sentence):
        if self.type != 'AS':
            chars = sentence.replace(' ', '')
        else:
            chars = sentence.replace('\u3000', '')
        return chars

    @classmethod
    def unsegmented_to_segmented(cls, unseg_text, labels, save_seg_text_file=None, rewrite=False, type='PKU'):
        '''This function is used to convert an unsegmented text to a segmented text with predicted labels.
            unseg_text: raw unsegmented text, a list of strings.
            labels: predicted labels, a list of labels.
            save_seg_text_file: save path.
            rewrite: if True, `save_seg_text_file` will cleared, or mode `a` is used.
        '''
        assert len(unseg_text) == len(labels), ('Make sure the lengths of `unseg_text` and `labels` match.')
        batch_size = len(unseg_text)

        for i in range(batch_size):
            assert len(unseg_text[i]) == len(labels[i]), ('Make sure the lengths of each sentence and its corresponding label match.')

        if type == 'AS':
            delimiter = '\u3000'
        elif type == 'City':
            delimiter = ' '
        else:
            delimiter = '  '
        
        seg_text = []
        for i in range(batch_size):
            unseg = unseg_text[i]
            label = labels[i]
            seg = ""

            sentence_len = len(unseg)

            for j in range(sentence_len):
                seg += unseg[j]
                if label[j] == 'B' or label[j] == 'I':
                    continue   # we don't need to do anything if label is 'B' or 'I'
                else:
                    if j == sentence_len - 1:
                        continue   # if it has reached EOS, we do nothing
                    else:
                        seg += delimiter # if it hasn't reached EOS, we need to add delimiter
            
            seg += '\n'
            seg_text.append(seg)
            
        if save_seg_text_file is not None:
            with open(save_seg_text_file, 'w' if rewrite == True else 'a', encoding='utf-8') as fout:
                fout.writelines(seg_text)

        return seg_text

    @classmethod
    def tensor_label_to_str(cls, tensor_label, mask, label_vocab):
        '''This function transfers a tensor of a batch of labels into a list of strings.
            tensor_label: a 2D tensor, shape of (batch_size, seq_len)
            mask: a 2D tensor, shape same as `tensor_label`
            label_vocab: a label vocabulary, where keys are labels and values are corresponding index
        '''
        index_vocab = {value: key for key, value in label_vocab.items()}

        batch_size = len(tensor_label)
        seq_len = tensor_label.size(1)

        labels = []
        for i in range(batch_size):
            label = ""
            for j in range(seq_len):
                if mask[i][j] == 1:
                    label += index_vocab[tensor_label[i][j].item()]
                else:
                    break
            labels.append(label)
        
        return labels  

# <========== This function is used to calculate OOV rate ==========>
def calculate_oov(test_corpus_file, token_type='word', train_vocab=None, train_vocab_file=None):
    '''Note: OOV rate = # sentence if exists word in sentence not in train_vocab / # total sentences in test_corpus
    '''
    if train_vocab is None:
        with open(train_vocab_file, 'r', encoding='utf-8') as f:
            train_vocab = f.readlines()
        for token in train_vocab:
            token = token.strip()
    train_vocab = set(train_vocab)
    train_vocab.add(' ')
    train_vocab.add('\n')
    train_vocab.add('\t')
    train_vocab.add('\r')
    train_vocab.add('\xa0')
    print(len(train_vocab))

    with open(test_corpus_file, 'r', encoding='utf-8') as f:
        test_corpus = f.readlines()

    n_all = len(test_corpus)
    n_oov = 0

    oov_sentences = []
    oov_words = []

    for sentence in test_corpus:
        sentence = sentence.strip()
        if token_type == 'word':
            sentence = sentence.split()
        for word in sentence:
            if word not in train_vocab:
                oov_sentences.append(sentence)
                oov_words.append(word)
                n_oov += 1
                break

    oov_rate = 1.0 * n_oov / n_all
    print('='*80)
    print('The oov rate is %.6f', oov_rate)
    print('*'*80)
    print('The last 10 oov sentences are:')
    print(oov_sentences[-10:])
    print('*'*80)
    print('The last 10 oov words are:')
    print(oov_words[-10:])
    print('='*80)

# <========== This function is used to get the data of a dataset ==========>
def statisctics(file_path):
    '''This function counts the number of sentences(lines, if one sentence per line), 
    and the number of unique tokens given the file, and prints the statistics to 
    standard output. What's more, this function gathers information of the sentences and 
    words.
    '''
    base_name = os.path.basename(file_path)
    line_length_count = {}
    word_length_count = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        words = []
        chars = []
        n_lines = 0
        for line in f:
            n_lines += 1
            line_len = len(line.strip())
            if line_len not in line_length_count.keys():
                line_length_count[line_len] = 1
            else: line_length_count[line_len] += 1

            words_line = line.strip().lower().split()
            for word in words_line:
                words.append(word)
                word_len = len(word)
                if word_len not in word_length_count.keys():
                    word_length_count[word_len] = 1
                else: 
                    word_length_count[word_len] += 1
                for char in word:
                    chars.append(char)
        n_words = len(words)
        n_chars = len(chars)
        n_unique_words = len(set(words))
        n_unique_chars = len(set(chars))

    print('='*80)
    print('There are %10d sentences/lines in file "%s".' % (n_lines, base_name))
    print('-'*80)
    print('There are %10d words in file "%s".' % (n_words, base_name))
    print('-'*80)
    print('There are %10d chars in file "%s".' % (n_chars, base_name))
    print('-'*80)
    print('There are %10d unique words in file "%s".' % (n_unique_words, base_name))
    print('-'*80)
    print('There are %10d unique chars in file "%s".' % (n_unique_chars, base_name))
    print('='*80)
    
    line_length_count = list(line_length_count.items())
    word_length_count = list(word_length_count.items())

    line_length_count = sorted(line_length_count, key=lambda x: x[0])
    word_length_count = sorted(word_length_count, key=lambda x: x[0])

    line_length_count = list(zip(*line_length_count))
    word_length_count = list(zip(*word_length_count))

    plt.figure(figsize=(20, 8), dpi=80)
    plt.subplot(1, 2, 1)
    plt.plot(line_length_count[0], line_length_count[1], 'b', linewidth=2)
    plt.xlabel('line length')
    plt.ylabel('frequency')
    plt.subplot(1, 2, 2)
    plt.plot(word_length_count[0], word_length_count[1], 'r', linewidth=2)
    plt.xlabel('word length')
    plt.ylabel('frequency')
    plt.show()
    return set(words), set(chars)

# <========== This function is used to split a dataset ==========>
def split_dataset(file_path, save_dir, save_file_name, parts=2, ratio=[0.9, 0.1]):
    '''You may need this function to split a large dataset without pre-splitting.
    You should choose to split into `train-dev` or `train-dev-test`, corresponding to `parts` and `ratio`.
    '''
    random.seed(2019)
    save_path = os.path.join(save_dir, save_file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)

    len_lines = len(lines)
    train_len_lines = int(len_lines * ratio[0])
    if parts == 2:
        train_lines = lines[:train_len_lines]
        dev_lines = lines[train_len_lines:]
        with open(save_path + '_train', 'w', encoding='utf-8') as fout:
            fout.writelines(train_lines)
        with open(save_path + '_dev', 'w', encoding='utf-8') as fout:
            fout.writelines(dev_lines)
    else:
        dev_len_lines = int(len_lines * (ratio[0] + ratio[1]))
        train_lines = lines[:train_len_lines]
        dev_lines = lines[train_len_lines:dev_len_lines]
        test_lines = lines[dev_len_lines:]
        with open(save_path + '_train', 'w', encoding='utf-8') as fout:
            fout.writelines(train_lines)
        with open(save_path + '_dev', 'w', encoding='utf-8') as fout:
            fout.writelines(dev_lines)
        with open(save_path + '_test', 'w', encoding='utf-8') as fout:
            fout.writelines(test_lines)

# <========== The next 3 functions is used to calculate BPE(byte pair encoding) ==========>
# Borrowed from paper `Neural Machine Translation of Rare Words with Subword Units`
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def BPE(vocab, n_merges=10):
    for _ in range(n_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

# <========== The next function is used to convert a batch of input into a batch tensor ==========>
def convert_to_tensor(batch, PAD=0, mode='pair', sort=True):
    '''Make sure `batch` is a list of tuples, where each tuple consists of an `x_sample` and a `y_sample`, if mode is `pair`.
    The `<PAD>` token is defaulted to `0`.
    '''
    if mode == 'pair':
        if sort:
            batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        X_batch = [sample[0] for sample in batch]
        Y_batch = [sample[1] for sample in batch]
        lengths = [len(x) for x in X_batch]
        max_len = lengths[0]
        assert max(lengths) == max_len, ('Something goes wrong, please check carefully.')
        length_tensor = torch.LongTensor(lengths)  # 1D tensor, shape of (batch_size,)

        X_batch_padded = [x + [PAD] * (max_len - len(x)) for x in X_batch]
        Y_batch_padded = [y + [PAD] * (max_len - len(y)) for y in Y_batch]

        X_tensor = torch.LongTensor(X_batch_padded)
        Y_tensor = torch.LongTensor(Y_batch_padded)

        mask_tensor = torch.ne(X_tensor, PAD)
    
        return X_tensor, Y_tensor, mask_tensor, length_tensor
    
    else:
        if sort:
            batch = sorted(batch, key=lambda x: len(x), reverse=True)
        lengths = [len(x) for x in batch]
        max_len = max(lengths)
        if sort:
            assert max(lengths) == max_len, ('Something goes wrong, please check carefully.')
        length_tensor = torch.LongTensor(lengths)

        batch_padded = [x + [PAD] * (max_len - len(x)) for x in batch]
        batch_tensor = torch.LongTensor(batch_padded)
        mask_tensor = torch.ne(batch_tensor, PAD)

        return batch_tensor, mask_tensor, length_tensor

# <========== The next function is used to index ==========>
def convert_to_index(batch, vocab, label_vocab=None, mode='pair'):
    '''Convert a batch into index.
    Make sure `batch` is a list of tuples, where each tuple consists of an `x_sample` and a `y_sample`, if mode is `train`.
    '''
    if mode == 'pair':
        assert label_vocab is not None, ('Under the `pair` mode, you should provide `label_vocab`.')
        batch_index = [([vocab.word_to_index(x_word) for x_word in x], [label_vocab[y_word] for y_word in y]) for x, y in batch]
    else:
        batch_index = [[vocab.word_to_index(x_word) for x_word in x] for x in batch]
    return batch_index

# <========== The next function is used to construct a vocabulary ==========>
def construct_vocab(corpus=None, corpus_file=None, save_path=None, vocab_name=None, max_size=50000, min_count=1, level='char'):
    '''Construct a vocab according to `corpus` or `corpus_file`
    '''
    if corpus is None and corpus_file is None:
        raise ValueError('You should provide a value to either `corpus` or `corpus_file`.')
    
    vocab = Vocab(vocab_name=vocab_name, max_size=max_size, min_count=min_count, level=level)
    
    if corpus:
        for sentence in corpus:
            vocab.add_sentence(sentence)
    else: 
        vocab.add_file(corpus_file)
    
    vocab.trim()

    if save_path:
        vocab.save_vocab(save_path)

    return vocab

# <========== The next function is used to construct a label vocabulary ==========>
def construct_label_vocab(labels):
    '''
    `labels` is a list. DON'T use `str` type.
    '''
    return {label: i for i, label in enumerate(labels)}

# <========== The next function is used to calculate CrossEntropy loss with mask ==========>
def CrossEntropyWithMask(logits, labels, mask, lengths):
    '''
        logits: 3D tensor, shape of (batch_size, seq_len, output_size)
        labels: 2D tensor, shape of (batch_size, seq_len)
        mask: 2D tensor, shape of (batch_size, seq_len)
        lengths: 1D tensor, shape of (batch_size, )
    '''
    total = torch.sum(lengths).item()
    probs = F.softmax(logits, dim=-1)
    logits_selected = torch.gather(probs, dim=-1, index=labels.unsqueeze(2)).squeeze(2) # shape of (batch_size, seq_len)
    loss = -torch.sum(torch.log(logits_selected).masked_select(mask))
    loss = 1. / total * loss
    return loss

# <========== The next function is used to load data once per batch ==========>
def LoadData(dataset, batch_size, shuffle=False):
    size = len(dataset)
    start = 0
    end = batch_size

    index_list = list(range(size))

    if shuffle:
        index_list = torch.randperm(size).tolist()

    while end <= size:
        current_index = index_list[start:end]
        batch = []
        for idx in current_index:
            batch.append(dataset[idx])
        yield batch
        start = end 
        end = end + batch_size
    
    if start < size:
        current_index = index_list[start:]
        batch = []
        for idx in current_index:
            batch.append(dataset[idx])
        yield batch

# <========== The next function is used to calculate the number of parameters of a model ==========>
def calculate_params(model, modules=None):
    '''
    If `module` is not given, this function will calculate all the number of params. in the model, otherwise, 
    it'll calculate the # params. of the given modules.
        modules: a list of string, each string is a module name.
    '''
    number = 0
    if modules is None:
        for param in model.parameters():
            number += torch.numel(param)
    else:
        for module in modules:
            for param in getattr(model, module).parameters():
                number += torch.numel(param)
    print('='*50)
    print('The number of parameters of ' + ('"model"' if modules is None else ' & '.join(modules)) + ' is: %d.' % number)
    print('='*50)
    return number