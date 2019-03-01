"""
This file implements some common metrics such as F1, BLEU etc.
@ author: Qinghong Han
@ date: Feb 22nd, 2019
@ contact: qinghong_han@shannonai.com
"""

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F  

import math
import re 
import numpy as np 
import itertools
import collections
from collections import Counter

def calculate_accuracy(y_true, y_pred, mask=None):
    '''This funtcion is used to calculate `accuracy`.
        y_true: a 2D tensor, shape of (batch_size, seq_len) or a 1D tensor, shape of (batch_size, )
        y_pred: shape same as `y_true`
        mask: a ByteTensor, shape same as `y_true`
    '''
    if mask is None:
        y_true = y_true.view(-1, 1).squeeze(1)
        y_pred = y_pred.view(-1, 1).squeeze(1)
    else:
        y_true = y_true.masked_select(mask)
        y_pred = y_pred.masked_select(mask)
    
    total = torch.numel(y_true)
    right = torch.sum(torch.eq(y_true, y_pred.long())).item()
    accuracy = 1.0 * right / total
    return right, total, accuracy


# <========== The next function is used to calculate F1 score ==========>
def calculate_f1(y_true, y_pred, labels, type='micro', mask=None):
    '''This function is used to calculate f1 score.
        y_true: a 2D tensor, shape of (batch_size, seq_len) or a 1D tensor, shape of (batch_size, )
        y_pred: same as `y_true`
        labels: a list of all possible labels
        type: `macro` or `micro`
        mask: a ByteTensor, shape same as `y_pred`
    '''
    labels = set(labels)
    true_size = list(y_true.size())
    pred_size = list(y_pred.size())
    
    if true_size != pred_size:
        raise ValueError('Input invalid!')
    
    if mask is None:
        y_true = y_true.view(-1, 1).squeeze(1).cpu().detach().numpy().tolist()
        y_pred = y_pred.view(-1, 1).squeeze(1).cpu().detach().numpy().tolist()   # (length, )
    else:
        y_true = y_true.masked_select(mask).cpu().detach().numpy().tolist()
        y_pred = y_pred.masked_select(mask).cpu().detach().numpy().tolist()

    data_dict = {}
    for i in labels:
        data_dict[i] = {'TP': 0, 'FP': 0, 'FN': 0}

    for i, pred in enumerate(y_pred):
        true = y_true[i]
        if true == pred:
            data_dict[pred]['TP'] += 1
        else:
            data_dict[true]['FN'] += 1
            if pred in labels:
                data_dict[pred]['FP'] += 1

    if type == 'micro':
        TP = FP = FN = 0
        for label in data_dict.keys():
            TP += data_dict[label]['TP']
            FP += data_dict[label]['FP']
            FN += data_dict[label]['FN']
        if TP == 0:
            precision = recall = f1 = 0
        else:
            precision = 1.0 * TP / (TP + FP)
            recall = 1.0 * TP / (TP + FN) 
            f1 = 2 * precision * recall / (precision + recall)
    else:
        labels_precision = []
        labels_recall = []
        for label in data_dict.keys():
            label_TP = data_dict[label]['TP']
            label_FP = data_dict[label]['FP']
            label_FN = data_dict[label]['FN']
            if label_TP == 0:
                label_precision = label_recall = 0.
            else:
                label_precision = 1.0 * label_TP / (label_TP + label_FP)
                label_recall = 1.0 * label_TP / (label_TP + label_FN)
            labels_precision.append(label_precision)
            labels_recall.append(label_recall)
        precision = sum(labels_precision) / len(labels_precision)
        recall = sum(labels_recall) / len(labels_recall)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)    
    return precision, recall, f1, data_dict

# <========== The next 5 functions are used to calculate BLEU score ==========>
# Borrowed from NLTK
def calculate_n_grams(sentence, n):
    '''This function gets all n-grams in `sentence`.
    '''
    if len(sentence) < n:
        n_grams = Counter()
    else:
        n_grams = []
        for i in range(len(sentence) - n + 1):
            n_gram = tuple(sentence[i: i + n])
            n_grams.append(n_gram)
        n_grams = Counter(n_grams)
    return n_grams

def calculate_modified_precision(candidate, references, n):
    '''This function calculates modified precision for BLEU.
        candidate: a candidate sentence, which is a list of integers.
        references: a list of reference sentences, where each sentence is a list of integers.
        n: BLEU's modified precision for the nth order.
    '''
    # extract all n-grams
    n_grams = calculate_n_grams(candidate, n)

    # for each unique ngram, we need to obtain it's max counts in the references
    max_counts = {}
    for n_gram in n_grams:
        max_counts[n_gram] = 0
    for ref in references:
        ref_n_grams = calculate_n_grams(ref, n)
        for n_gram in n_grams:
            max_counts[n_gram] = max(ref_n_grams[n_gram], max_counts[n_gram])
    
    # now clip
    clipped_ngrams = {n_gram: min(count, max_counts[n_gram]) for n_gram, count in n_grams.items()}

    numerator = sum(clipped_ngrams.values())
    denominator = max(1, sum(n_grams.values()))

    return (1.0 * numerator, 1.0 * denominator)

def closest_ref_length(references, candidate_len):
    '''This function finds the closest reference length given the candidate length.
    '''
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lengths, key=lambda ref_len: (abs(ref_len - candidate_len))
    )
    return closest_ref_len

def brevity_penalty(reference_lengths, candidate_lengths):
    '''This function calculates brevity penalty.
    '''
    if candidate_lengths > reference_lengths:
        return 1
    elif candidate_lengths == 0:
        return 0
    else:
        return math.exp(1 - reference_lengths / candidate_lengths)

def calculate_bleu(candidates, list_references, n=4):
    '''This function calculates n-gram BLEU score.
        candidates: a list of candidate sentences, where each sentence is a list of integers.
        list_references: a list of references sentences, where each element in `list_references` is a list of references corresponding to the candidate.
        n: BLEU-n score, default set to 4.
    '''
    assert len(candidates) == len(list_references), ('The number of candidates should be equal to the number of references.')

    weights = [1. / n for _ in range(n)]
    p_numerators = Counter()   # key = ngram order, value = No. of ngram matches
    p_denominators = Counter() # key = ngram order, value = No, of ngram matches
    candidate_lengths, reference_lengths = 0, 0

    # Iterate through each candidate and their corresponding references.
    for candidate, references in zip(candidates, list_references):
        # for each order of n-gram, calculate the numerator and 
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = calculate_modified_precision(candidate, references, i)
            p_numerators[i] += p_i[0]
            p_denominators[i] += p_i[1]
        
        # calculate the candidate length and the closest reference length.
        # adds them to the corpus-level candidate and reference counts
        candidate_lengths += len(candidate)
        reference_lengths += closest_ref_length(references, len(candidate))
    
    BP = brevity_penalty(reference_lengths, candidate_lengths)

    # collect the various precision values for the different ngram orders.
    p_n = [(p_numerators[i], p_denominators[i]) for i, _ in enumerate(weights, start=1)]

    # returns 0 if there's no matching n-grams
    if p_numerators[1] == 0:
        return 0

    BLEU_score = 0.
    for i in range(n):
        BLEU_score += weights[i] * math.log(p_n[i][0] / p_n[i][1])
    BLEU_score = BP * math.exp(BLEU_score)
    return BLEU_score


# <========== The next 2 functions are used to calculate Rouge score ==========>
def LCS(sentence1, sentence2):
    '''This function is used to calculate the LCS of sentence1 and sentence2, a naive implementation.
    '''
    m = len(sentence1)
    n = len(sentence2)
    if m == 0 or n == 0:
        return 0
    lcs = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0 or j == 0) and sentence1[i] == sentence2[j]:
                lcs[i][j] = 1
            elif i == 0 or j == 0:
                continue 
            else:
                if sentence1[i] == sentence2[j]:
                    lcs[i][j] = max(lcs[i-1][j-1] + 1, lcs[i][j-1], lcs[i-1][j])
                else:
                    lcs[i][j] = max(lcs[i][j-1], lcs[i-1][j])
    return lcs[m-1][n-1]

def calculate_rouge(reference, candidates, type='L', level='sentence'):
    '''This function is used to calculate ROUGE score.
        reference: a reference sentence.
        candidates: a list of candidate senteces.
        type: ROUGE score type.
        level: `sentence` or `summary`.
    '''
    if level == 'sentence':
        lcs = LCS(reference, candidates)
        R_lcs = 1.0 * lcs / len(reference)
        P_lcs = 1.0 * lcs / len(candidates)
        beta = P_lcs / R_lcs
        F_lcs = (1 + beta**2) * R_lcs * P_lcs / (R_lcs + beta**2 * P_lcs)
        return F_lcs
    
    

def calculate_meteor():
    pass


# <========== The next function is used to calculate PPL ==========>
def calculate_ppl(logits=None, labels=None, loss=None):
    '''This function is used to calculate `ppl`.
        logits: tensor, shape of (batch_size, seq_len, vocab_size)
        labels: tensor, shape of (batch_size, seq_len)
        loss: if you have `loss`, you can directly use it.
    '''
    if loss is None:
        N = torch.numel(labels)
        labels = labels.unsqueeze(2)
        probs = torch.gather(logits, dim=-1, index=labels).squeeze(2)
        log_probs = -1. / N * torch.sum(torch.log2(probs))
        ppl = torch.pow(2, log_probs)
        return ppl.item()
    else:
        return torch.pow(2, loss).item()
