import EnsembledAttentionAndNNv2.data_create as data_create
from torch.utils.data import Dataset
import numpy as np
import os
import utilities as utils
import torch
from example_indices import sorted_indices

DATA_PATH = os.path.join(utils.INTERMEDIATE_DATA_DIR, "AttentionAndNN")


class AllAgentsNormalizedDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datafile = self.loadData()
        data = self.datafile['data']
        valid_indices = self.getAllValidIndices(data)
        self.indices = [ind for ind in self.config["INDICES"] if ind in valid_indices]
        self.data = data[self.indices]
        self.X = self.datafile['X'][self.indices]
        
        # breakpoint()
        if config.get('INFERENCE', False) == False:
            self.Y = self.datafile['Y'][self.indices]
        else:
            self.Y = np.zeros_like(self.X)

    def getAllValidIndices(self, data):
        dv = np.mean(np.abs(data[:, 0, 40:50, 2:4] - data[:, 0, 39:49, 2:4]), axis=(1, 2))
        valid_inds = np.nonzero((dv >= self.config['MIN_METRIC']) & (dv < self.config['MAX_METRIC']))[0].tolist()
        breakpoint()
        return set(valid_inds)

    def loadData(self):
        filename = self.config["DATA_FILENAME"]
        datafile = np.load(os.path.join(DATA_PATH, filename))

        return datafile

    def getOriginalSpacePredictions(self, normY, indices, indicator="none"):
        if indicator == 'true':
            return self.data[indices, 0, 50:, :2]
        else:
            origY = data_create.createOriginalSpacePredictions(
                normY, self.data[indices]
            )
            
            return origY

    def computeOriginalSpaceMetric(self, true, pred):
        return np.mean((true - pred) ** 2)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y, index

    def __len__(self):
        return len(self.data)




class AllAgentsNormalizedRouterDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datafile = self.loadData()
        self.indices = self.config["INDICES"]
        self.data = self.datafile["data"][self.indices]
        self.X = self.datafile['X'][self.indices]
        
        if config.get('INFERENCE', False) == False:
            self.Y = self.loadY()
        else:
            self.Y = np.zeros_like(self.X)

    def loadY(self):
        percentiles = [sorted_indices.index(ind) / len(sorted_indices) for ind in self.indices]
        classes = []

        for ind, perc in zip(self.indices, percentiles):
            for i, regime in enumerate(self.config['REGIMES']):
                if perc >= regime[0] and perc < regime[1]:
                    classes.append(i)

        # breakpoint()
        return classes
                

    def loadData(self):
        filename = self.config["DATA_FILENAME"]
        datafile = np.load(os.path.join(DATA_PATH, filename))

        return datafile

    def getOriginalSpacePredictions(self, normY, indices, indicator="none"):
        if indicator == 'true':
            return self.data[indices, 0, 50:, :2]
        else:
            origY = data_create.createOriginalSpacePredictions(
                normY, self.data[indices]
            )
            
            return origY

    def computeOriginalSpaceMetric(self, true, pred):
        return np.mean((true - pred) ** 2)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)

        # breakpoint()

        return x, y, index

    def __len__(self):
        return len(self.data)
