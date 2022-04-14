import torch
import os
from math import log10

folder_name = 'models'
# folder_name = 'best_model_storage'

file_list = os.listdir(folder_name)
for filename in file_list:
    weight = torch.load(folder_name + '/' + filename)
    print(filename, weight['best'], 10 * log10(1 / weight['best'] ** 2))
