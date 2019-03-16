import os 
import sys 
import csv
import itertools
import collections
import math
import numpy as np  
import pandas as pd 

from utils import *

def merge_files(BASE_DIR, files, save_file_path):
    for file in files:
        with open(os.path.join(BASE_DIR, file), 'r', encoding='utf-8') as fin:
            with open(save_file_path, 'a', encoding='utf-8') as fout:
                fout.writelines(fin.readlines())

def convert_seg_to_unseg(file_path, save_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        with open(save_path, 'w', encoding='utf-8') as fout:
            seg_lines = fin.readlines()
            unseg_lines = [''.join(line.split(' ')) for line in seg_lines]
            fout.writelines(unseg_lines)

# ================== split Bakeoff 2005 dataset ==========================================================
# 
#file_path = '/home/sulley/桌面/CWS/datasets/training/msr' #['msr', 'as', 'cityu', 'pku']
#save_dir = '/home/sulley/桌面/CWS/datasets/training'
#save_file_name = 'msr' #['msr', 'as', 'cityu', 'pku']
#split_dataset(file_path, save_dir, save_file_name)

# Bakeoff 2005 dataset statistics
#file_names = ['as', 'pku', 'msr', 'cityu']
#post_fix = ['_train', '_dev']
#base_dir = './datasets/training'

#for file_name in file_names:
#    for fix in post_fix:
#        file_path = os.path.join(base_dir, file_name+fix)
#        statisctics(file_path)

#file_names = ['as_test_gold', 'pku_test_gold', 'msr_test_gold', 'cityu_test_gold']
#base_dir = './datasets/gold'
#for file_name in file_names:
#    file_path = os.path.join(base_dir, file_name)
#    statisctics(file_path)


# ================ merge CTB6.0 dataset and split into train/dev/test ============================================================
# 
#test_data_files = ['chtb_000' + str(i) + '.seg' for i in range(1, 10)] + ['chtb_00' + str(i) + '.seg' for i in range(10, 41)] + \
#                    ['chtb_0' + str(i) + '.seg' for i in range(901, 932)] + ['chtb_1018.seg', 'chtb_1020.seg', 'chtb_1036.seg'] + \
#                    ['chtb_1044.seg', 'chtb_1060.seg', 'chtb_1061.seg', 'chtb_1072.seg', 'chtb_1118.seg', 'chtb_1119.seg', 'chtb_1132.seg'] + \
#                    ['chtb_1141.seg','chtb_1142.seg', 'chtb_1148.seg'] + ['chtb_' + str(i) + '.seg' for i in range(2165, 2181)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2296, 2311)] + ['chtb_' + str(i) + '.seg' for i in range(2570, 2603)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2800, 2820)] + ['chtb_' + str(i) + '.seg' for i in range(3110, 3146)]


#valid_data_files = ['chtb_00' + str(i) + '.seg' for i in range(41, 80)] + ['chtb_' + str(i) + '.seg' for i in range(1120, 1130)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2140, 2160)] + ['chtb_' + str(i) + '.seg' for i in range(2280, 2295)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2550, 2570)] + ['chtb_' + str(i) + '.seg' for i in range(2775, 2800)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(3080, 3110)]


#train_data_files = ['chtb_00' + str(i) + '.seg' for i in range(81, 100)] + ['chtb_0' + str(i) + '.seg' for i in range(100, 326)] + ['chtb_0' + str(i) + '.seg' for i in range(400, 455)] + \
#                    ['chtb_0' + str(i) + '.seg' for i in range(600, 886)] + ['chtb_0900.seg'] + \
#                    ['chtb_0' + str(i) + '.seg' for i in range(500, 555)] + ['chtb_0' + str(i) + '.seg' for i in range(590, 597)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(1001, 1018)] + ['chtb_1019.seg'] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(1021, 1036)] + ['chtb_' + str(i) + '.seg' for i in range(1037, 1044)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(1045, 1060)] + ['chtb_' + str(i) + '.seg' for i in range(1062, 1072)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(1073, 1079)] + ['chtb_' + str(i) + '.seg' for i in range(1100, 1118)] + \
#                    ['chtb_1130.seg', 'chtb_1131.seg'] + ['chtb_' + str(i) + '.seg' for i in range(1133, 1141)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(1143, 1148)] + ['chtb_' + str(i) + '.seg' for i in range(1149, 1152)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2000, 2140)] + ['chtb_' + str(i) + '.seg' for i in range(2160, 2165)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2181, 2280)] + ['chtb_' + str(i) + '.seg' for i in range(2311, 2550)] + \
#                    ['chtb_' + str(i) + '.seg' for i in range(2603, 2775)] + ['chtb_' + str(i) + '.seg' for i in range(2820, 3080)]

#BASE_DIR = '/home/sulley/桌面/datasets/ctb6/data/segmented'
#merge_files(BASE_DIR, train_data_files, './datasets/ctb_train')
#merge_files(BASE_DIR, valid_data_files, './datasets/ctb_dev')
#merge_files(BASE_DIR, test_data_files, './datasets/ctb_test')

# CTB6.0 dataset statistics
#file_names = ['ctb_train', 'ctb_dev', 'ctb_test']
#base_dir = './datasets'

#for file_name in file_names:
#    file_path = os.path.join(base_dir, file_name)
#    statisctics(file_path)


# ================ create UD data ============================================================
# 
#train_file = '/home/sulley/桌面/datasets/UD2017/train/UD_Chinese/zh-ud-train.conllu'
#dev_file = '/home/sulley/桌面/datasets/UD2017/train/UD_Chinese/zh-ud-dev.conllu'
#test_file = '/home/sulley/桌面/datasets/UD2017/test/gold/zh.conllu'

#convert_conll_to_cws(train_file, './datasets/ud_train')
#convert_conll_to_cws(dev_file, './datasets/ud_dev')
#convert_conll_to_cws(test_file, './datasets/ud_test')

# UD dataset statistics
#file_names = ['ud_train', 'ud_dev', 'ud_test']
#base_dir = './datasets'
#for file in file_names:
#    file_path = os.path.join(base_dir, file)
#    statisctics(file_path)


# convert to unsegmented sentences
#convert_seg_to_unseg('./datasets/gold/ctb_test_gold', './datasets/testing/ctb_test')
#convert_seg_to_unseg('./datasets/gold/ud_test_gold', './datasets/testing/ud_test')

#================== convert data format (for PKU) ==============================================
#
#half_to_full_width('./datasets/testing/pku_test', './datasets/testing/pku_test_full')

#================== build word vocab for UD and CTB training sets =============================
#
#ud_vocab = Vocab(min_count=1)
#ud_vocab.add_file('./datasets/training/ud_train')
#ud_vocab.trim()
#ud_vocab.save_vocab('./datasets/gold/ud_training_words')

#ud_vocab = Vocab(min_count=1)
#ud_vocab.add_file('./datasets/training/ctb_train')
#ud_vocab.trim()
#ud_vocab.save_vocab('./datasets/gold/ctb_training_words')

ud_train_vocab = Vocab(min_count=1)
ud_train_vocab.add_file('./datasets/training/ud_train')
ud_train_vocab.trim()
ud_dev_vocab = Vocab(min_count=1)
ud_dev_vocab.add_file('./datasets/training/ud_dev')
ud_dev_vocab.trim()

ud_train_vocab = ud_train_vocab.as_list()
ud_dev_vocab = ud_dev_vocab.as_list()

vocab_overlap_and_difference_ratio(ud_train_vocab, ud_dev_vocab)