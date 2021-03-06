import torch 
import torch.nn as nn 
import torch.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader

import numpy as np 
import os
import sys 

from utils import construct_vocab, construct_label_vocab, convert_to_index, convert_to_tensor
from utils import CrossEntropyWithMask, MSEWithMask, calculate_params, convert_to_word_lengths
from utils import CWSDataset, CTBDataset, LoadData

from metrics import calculate_f1, calculate_accuracy
from modules import CWSLstm

import argparse
import random
import math
import itertools
import collections

random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)

BASE_DIR = os.path.dirname(__file__)
if not os.path.exists(os.path.join(BASE_DIR, 'checkpoint')):
    os.mkdir(os.path.join(BASE_DIR, 'checkpoint'))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoint')

parser = argparse.ArgumentParser(description='Argparser for Chinese Word Segmentation.')

# you may need to change the next arguments.
parser.add_argument('--data_type', type=str, default='PKU')
parser.add_argument('--train_path', type=str, default='',help='Training data path.')
parser.add_argument('--dev_path', type=str, default='', help='Development data path.')
parser.add_argument('--eval_path', type=str, default='', help='Evaluation data path.')
parser.add_argument('--save_pred_path', type=str, default='', help='Save predictions path.')
parser.add_argument('--vocab_path', type=str, default='', help='Vocabulary path.')
parser.add_argument('--model', type=str, default='', help='Path of trained model.')
parser.add_argument('--layers', type=int, default=2, help='Lstm layers.')
parser.add_argument('--embed_dim', type=int, default=100, help='Embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=300, help='Lstm hiddem dimension for each direction.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
parser.add_argument('--decay_step', type=int, default=10, help='Steps to decay.')
parser.add_argument('--decay_rate', type=float, default=0.5, help='The rate of learning rate decay at end of each epoch.')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Whether to use GPU.')
parser.add_argument('--shuffle', action='store_true', default=True, help='Whether to shuffle for each epoch.')
parser.add_argument('--use_cnn', action='store_true', default=False, help='Whether to use cnn to extract ngram features.')
parser.add_argument('--predict_word', action='store_true', default=False, help='Whether to predict the length of a word.')
parser.add_argument('--loss_lambda', type=float, default=1e-5, help='The ratio of loss when using `predict_word`.')
parser.add_argument('--oov_window', type=int, default=0, help='OOV window size. If 0, we don"t use it')
parser.add_argument('--use_attention', action='store_true', default=False, help='Whether to use attention to predict word spans.')

# the next arguments can be frozen.
parser.add_argument('--interval_report', type=int, default=10, help='Interval to report.')
parser.add_argument('--interval_write', type=int, default=10, help='Record while training.')
parser.add_argument('--inplace_test', action='store_true', default=False, help='Whether to test in place.')

args = parser.parse_args()
print(args)

data_type = args.data_type
train_path = args.train_path
dev_path = args.dev_path
eval_path = args.eval_path
save_pred_path = args.save_pred_path
layers = args.layers
embed_dim = args.embed_dim
hidden_dim = args.hidden_dim
dropout = args.dropout
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
decay_step = args.decay_step
decay_rate = args.decay_rate
shuffle = args.shuffle
use_cnn = args.use_cnn
predict_word = args.predict_word 
loss_lambda = args.loss_lambda
oov_window = args.oov_window
use_attention = args.use_attention
use_cuda = args.use_cuda
interval_report = args.interval_report
interval_write = args.interval_write
model_path = args.model

if not os.path.exists(os.path.join(CHECKPOINT_DIR, data_type)):
    os.mkdir(os.path.join(CHECKPOINT_DIR, data_type))
SAVE_DIR = os.path.join(CHECKPOINT_DIR, data_type)

print("="*15, "Constructing vocabulary", '='*15)
labels = ['B', 'I', 'E', 'S']
label_vocab = construct_label_vocab(labels)
vocab = construct_vocab(corpus_file=train_path, min_count=1)
vocab_size = len(vocab)
print("="*15, "Finish constructing vocabulary", '='*15)

model = CWSLstm(layers=layers,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                vocab_size=vocab_size,
                dropout=dropout,
                use_cnn=use_cnn,
                oov_window=oov_window,
                predict_word=predict_word,
                use_attention=use_attention)

print('='*15, 'Model Information', '='*15)
print(model)
calculate_params(model)
print('='*50)

if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=decay_rate, step_size=decay_step)

if data_type not in ['PKU', 'MS', 'City', 'AS', 'UD', 'CTB']:
    assert ValueError('`data_type` wrong, please check again.')

if data_type in ['PKU', 'MS', 'City', 'AS']:
    if train_path != '':
        train_dataset = CWSDataset(train_path, type=data_type)
    if dev_path != '':
        dev_dataset = CWSDataset(dev_path, type=data_type)
    if eval_path != '':
        if args.inplace_test is False:
            test_dataset =CWSDataset(eval_path, type=data_type, mode='single', sort=False)
        else:
            test_dataset =CWSDataset(eval_path, type=data_type)
else: # for `CTB` and `UD``
    if train_path != '':
        train_dataset = CWSDataset(train_path)
    if dev_path != '':
        dev_dataset = CWSDataset(dev_path)
    if eval_path != '':
        if args.inplace_test is False:
            test_dataset =CWSDataset(eval_path, mode='single', sort=False)
        else:
            test_dataset =CWSDataset(eval_path)

# <========== if `model_path` is not provided, training procedure will be executed =========>
if model_path == '':

    train_size = len(train_dataset)
    n_iters = (train_size // batch_size) if train_size % batch_size == 0 else (train_size // batch_size + 1)
    early_stop = 0

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_acc = 0.0
    train_iter_f1 = []
    valid_epoch_f1 = []
    valid_epoch_acc = []

    print('='*80)
    print('Training data has %d samples, which need %d iterations to finish one epoch.' % (train_size, n_iters))
    print('*'*80)
    print('Vocabulary has %d words/chars.' % vocab_size)
    print('='*80)

    print('='*5 + ' Starting training! ' + '='*5)
    for epoch in range(epochs):
        model.train()
        total = 0
        right = 0
        data_dict = {}
        for i in label_vocab.values():
            data_dict[i] = {'TP': 0, 'FP': 0, 'FN': 0}
        current_epoch = epoch + 1
        print('='*30)
        print('Start training at epoch: %d' % current_epoch)
        print('='*30)
        for iters, current_batch in enumerate(LoadData(train_dataset, batch_size, shuffle=shuffle)):
            model.zero_grad()
            current_iter = iters + 1 

            batch_indexed = convert_to_index(current_batch, vocab, label_vocab=label_vocab)
            X_tensor, Y_tensor, mask_tensor, length_tensor = convert_to_tensor(batch_indexed)
            if use_cuda:
                X_tensor = X_tensor.cuda()
                Y_tensor = Y_tensor.cuda()
                mask_tensor = mask_tensor.cuda()
                length_tensor = length_tensor.cuda()

            if predict_word is False:
                logits = model(X_tensor, length_tensor, mask_tensor) # shape of (batch_size, seq_len, output_size)
            else:
                word_lengths_tensor = convert_to_word_lengths(Y_tensor, length_tensor, label_vocab) # shape of (batch_size, seq_len)\
                logits, outputs_word_lengths = model(X_tensor, length_tensor, mask_tensor) # outputs_word: shape of (batch_size, seq_len)
                aux_loss = MSEWithMask(word_lengths_tensor, outputs_word_lengths, mask_tensor)

            loss = CrossEntropyWithMask(logits, Y_tensor, mask=mask_tensor, lengths=length_tensor)
            main_loss_item = loss.item()
            aux_loss_item = aux_loss.item() if predict_word else 0

            if predict_word:
                loss = loss + loss_lambda * aux_loss
            current_loss = loss.item()

            _, Y_pred = torch.max(logits, dim=-1)
            batch_right, batch_total, _ = calculate_accuracy(Y_tensor, Y_pred, mask_tensor)
            _, _, _, batch_data_dict = calculate_f1(Y_tensor, Y_pred, labels=[0, 1, 2, 3], mask=mask_tensor)

            total += batch_total
            right += batch_right
            for label in data_dict.keys():
                data_dict[label]['TP'] += batch_data_dict[label]['TP']
                data_dict[label]['FP'] += batch_data_dict[label]['FP']
                data_dict[label]['FN'] += batch_data_dict[label]['FN']

            if current_iter % interval_report == 0:
                accuracy = 1.0 * batch_right / batch_total # calculate accuracy
                # calculate macro F1
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

                print('[ Epoch: %-4d, Iteration: %-5d / %-5d]  Total loss: %-6.2f, Main loss: %-6.2f, Aux loss: %-6.2f, Precision: %-6.4f, Recall: %-6.4f, F1: %-6.4f, Accuracy: %-6.4f.' %(
                    current_epoch, current_iter, n_iters, current_loss, main_loss_item, aux_loss_item, precision, recall, f1, accuracy))
                train_iter_f1.append(f1)
        
                total = 0
                right = 0
                for i in label_vocab.values():
                    data_dict[i] = {'TP': 0, 'FP': 0, 'FN': 0}

            loss.backward()
            optimizer.step()

        # validation procedure
        print('='*5 + ' Starting validating! ' + '='*5)
        model.eval()

        total_loss = 0.0
        total = 0
        right = 0
        for i in label_vocab.values():
            data_dict[i] = {'TP': 0, 'FP': 0, 'FN': 0}

        for current_batch in LoadData(dev_dataset, batch_size):
            
            batch_indexed = convert_to_index(current_batch, vocab, label_vocab=label_vocab)
            X_tensor, Y_tensor, mask_tensor, length_tensor = convert_to_tensor(batch_indexed)
            if use_cuda:
                X_tensor = X_tensor.cuda()
                Y_tensor = Y_tensor.cuda()
                mask_tensor = mask_tensor.cuda()
                length_tensor = length_tensor.cuda()
            
            if predict_word is False:
                logits = model(X_tensor, length_tensor, mask_tensor) # shape of (batch_size, seq_len, output_size)
            else:
                word_lengths_tensor = convert_to_word_lengths(Y_tensor, length_tensor, label_vocab) # shape of (batch_size, seq_len)
                logits, outputs_word_lengths = model(X_tensor, length_tensor, mask_tensor) # outputs_word: shape of (batch_size, seq_len)
                aux_loss = MSEWithMask(word_lengths_tensor, outputs_word_lengths, mask_tensor)

            loss = CrossEntropyWithMask(logits, Y_tensor, mask=mask_tensor, lengths=length_tensor)

            if predict_word:
                loss = loss + loss_lambda * aux_loss
            current_loss = loss.item()

            total_loss += current_loss

            _, Y_pred = torch.max(logits, dim=-1)   # shape of (batch_size, seq_len)

            batch_right, batch_total, _ = calculate_accuracy(Y_tensor, Y_pred, mask_tensor)
            _, _, _, batch_data_dict = calculate_f1(Y_tensor, Y_pred, labels=[0, 1, 2, 3], mask=mask_tensor)
            right += batch_right
            total += batch_total
            for label in data_dict.keys():
                data_dict[label]['TP'] += batch_data_dict[label]['TP']
                data_dict[label]['FP'] += batch_data_dict[label]['FP']
                data_dict[label]['FN'] += batch_data_dict[label]['FN']

        accuracy = 1.0 * right / total 
        if best_acc < accuracy:
            best_acc = accuracy
        valid_epoch_acc.append(accuracy)

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

        scheduler.step()

        valid_epoch_f1.append(f1)
        if best_f1 < f1:
            early_stop = 0
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model'))
        else:
            early_stop += 1
            if early_stop == 10:
                break
        print('='*80)
        print('[Epoch: %4d] Current loss is %-6.4f, Current accuracy is %-6.4f, best accuracy is %-6.4f. Current F1 is %-6.4f, best F1 is %-6.4f' % 
                (current_epoch, total_loss, accuracy, best_acc, f1, best_f1))
        print('='*80)

    print('='*5 + ' Training finished! The best precision is: %-6.4f, recall is: %-6.4f, F1 is: %-6.4f, accuracy is: %-6.4f' % (best_precision, best_recall, best_f1, best_acc) + '='*5)

# <========== if `model_path` is provided, evaluation procedure will be executed =========>
else:
    model.load_state_dict(torch.load(model_path))

    print('='*5 + ' Starting Evaluation! ' + '='*5)
    model.eval()
    print('='*15, 'Model Information', '='*15)
    print(model)
    calculate_params(model)
    print('='*50)

    if args.inplace_test is False:
        save_path = os.path.join(SAVE_DIR, data_type) + '_pred'
        print(save_path)
        for i, current_batch in enumerate(LoadData(test_dataset, batch_size=1)):
            
            batch_indexed = convert_to_index(current_batch, vocab, label_vocab=label_vocab, mode='single')
            X_tensor, mask_tensor, length_tensor = convert_to_tensor(batch_indexed, mode='single', sort=False)
            if use_cuda:
                X_tensor = X_tensor.cuda()
                mask_tensor = mask_tensor.cuda()
                length_tensor = length_tensor.cuda()
                
            if predict_word is False:
                logits = model(X_tensor, length_tensor, mask_tensor) # shape of (batch_size, seq_len, output_size)
            else:
                logits, _ = model(X_tensor, length_tensor, mask_tensor) # outputs_word: shape of (batch_size, seq_len)
                
            _, Y_pred = torch.max(logits, dim=-1)   # shape of (batch_size, seq_len)

            batch_label = CWSDataset.tensor_label_to_str(Y_pred, mask_tensor, label_vocab)
            
            CWSDataset.unsegmented_to_segmented(current_batch, batch_label, save_seg_text_file=save_path, rewrite=(False if i != 0 else True), type=data_type)
        print('='*80)
        print('Finish Prediction, save to %s.' % (save_path))
        print('='*80)

    else:
        print('===== Start testing in place =====')
        total = 0
        right = 0
        data_dict = {}
        for i in label_vocab.values():
            data_dict[i] = {'TP': 0, 'FP': 0, 'FN': 0}
        for i, current_batch in enumerate(LoadData(test_dataset, batch_size=batch_size)):

            batch_indexed = convert_to_index(current_batch, vocab, label_vocab=label_vocab)
            X_tensor, Y_tensor, mask_tensor, length_tensor = convert_to_tensor(batch_indexed)
            if use_cuda:
                X_tensor = X_tensor.cuda()
                Y_tensor = Y_tensor.cuda()
                mask_tensor = mask_tensor.cuda()
                length_tensor = length_tensor.cuda()

            if predict_word is False:
                logits = model(X_tensor, length_tensor, mask_tensor) # shape of (batch_size, seq_len, output_size)
            else:
                logits, _ = model(X_tensor, length_tensor, mask_tensor) # outputs_word: shape of (batch_size, seq_len)

            _, Y_pred = torch.max(logits, dim=-1)   # shape of (batch_size, seq_len)

            batch_right, batch_total, _ = calculate_accuracy(Y_tensor, Y_pred, mask_tensor)
            _, _, _, batch_data_dict = calculate_f1(Y_tensor, Y_pred, labels=[0, 1, 2, 3], mask=mask_tensor)
            right += batch_right
            total += batch_total
            for label in data_dict.keys():
                data_dict[label]['TP'] += batch_data_dict[label]['TP']
                data_dict[label]['FP'] += batch_data_dict[label]['FP']
                data_dict[label]['FN'] += batch_data_dict[label]['FN']

        accuracy = 1.0 * right / total

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
        print('Validation accuracy is %-6.4f, precision is %-6.4f, recall is %-6.4f, F1 is %-6.4f.' % 
                (accuracy, precision, recall, f1))

