import os 
import sys 
import csv
import itertools
import collections
import math
import numpy as np  
import pandas as pd 

from utils import split_dataset, Vocab

# all we need to do is to split the raw data into `train` and `valid` parts.
file_path = '/home/sulley/桌面/CWS/datasets/training/msr' #['msr', 'as', 'cityu', 'pku']
save_dir = '/home/sulley/桌面/CWS/datasets/training'
save_file_name = 'msr' #['msr', 'as', 'cityu', 'pku']
split_dataset(file_path, save_dir, save_file_name)