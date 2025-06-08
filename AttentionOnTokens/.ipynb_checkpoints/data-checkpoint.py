import AttentionOnTokens.data_create as data_create
from torch.utils.data import Dataset
import numpy as np
import os
import utilities as utils
import torch

DATA_PATH = os.path.join(utils.INTERMEDIATE_DATA_DIR, "AttentionOnTokens")


class AllAgentsNormalizedDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datafile = self.loadData()
        self.indices = self.config["INDICES"]
        self.data = self.datafile["data"][self.indices]
        self.X = self.datafile['X'][self.indices]
        if config.get('INFERENCE', False) == False:
            self.Y = self.datafile['Y'][self.indices]
        else:
            self.Y = np.zeros_like(self.X)

    def loadData(self):
        filename = self.config["DATA_FILENAME"]
        datafile = np.load(os.path.join(DATA_PATH, filename))

        return datafile

    def getOriginalSpacePredictions(self, normY, indices, indicator="none"):
        # breakpoint()
        corresp_data = self.data[indices//3, 0, (indices % 3 ) * 15 + 14, :2]
        orgY = data_create.createOriginalSpacePredictions(
                normY, corresp_data
            )
        # for ind in indices:
        #     source_index = index // 3
        #     source_slide_ind = (index % 3 ) * 15

        #     org_pos = self.data[source_index][0, source_slide_ind+14]
            
        # if indicator == 'true':
        #     return self.data[indices, 0, 50:, :2]
        # else:
        #     origY = data_create.createOriginalSpacePredictions(
        #         normY, self.data[indices]
        #     )
            
        return orgY

    def computeOriginalSpaceMetric(self, true, pred):
        return np.mean((true - pred) ** 2)

    def __getitem__(self, index):
        source_index = index // 3
        source_slide_ind = (index % 3 ) * 15
        
        x = self.X[source_index][:, source_slide_ind : source_slide_ind + 15]
        y = self.Y[source_index][source_slide_ind + 15 - 1 : source_slide_ind + 15 - 1 + 60]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y, index

    def __len__(self):
        return len(self.data) * 3
