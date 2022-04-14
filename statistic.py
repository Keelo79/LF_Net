import torch
import os
from math import log10

file_list = os.listdir('models')
for filename in file_list:
    weight = torch.load('models/' + filename)
    print(filename, weight['best'], 10 * log10(1 / weight['best'] ** 2))
