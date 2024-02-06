#Collector function to collate the data from the data source

# Path: src/data/datasets.py

from torch.utils.data import Dataset
import numpy as np
import os

class LipReadingDataset(Dataset):
    