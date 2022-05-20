# from __future__ import unicode_literals, print_function, division
# from io import open
# import unicodedata
# import string
# import re
# import random

# import torch
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearEncoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x