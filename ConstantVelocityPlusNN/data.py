import ConstantVelocityPlusNN.data_create as data_create
from torch.utils.data import Dataset
import numpy as np
import os
import utilities as utils
import torch

DATA_PATH = os.path.join(utils.INTERMEDIATE_DATA_DIR, "ConstantVelocityPlusNN")


# class EgoAgentNormalizedDataset(Dataset):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.datafile = self.loadData()
#         self.indices = self.config["INDICES"]
#         self.data = self.datafile["data"][self.indices]
#         self.X = self.datafile["X"][self.indices]
#         if self.config.get("INFERENCE", False) == False:
#             self.Y = self.datafile["Y"][self.indices]

#         # breakpoint()

#     def loadData(self):
#         filename = self.config["DATA_FILENAME"]
#         datafile = np.load(os.path.join(DATA_PATH, filename))

#         return datafile

#     def getOriginalSpacePredictions(self, normY, indices):
#         # breakpoint()
#         Y = data_create.createUnnormalizedY(normY)
#         origY = data_create.createOriginalSpacePredictions(Y, self.data[indices])

#         return origY

#     def computeOriginalSpaceMetric(self, true, pred):
#         # breakpoint()
#         # print(f"computeUnnormalizedMetric: true={true[0, :, :]}, pred={pred[0, :, :]}")
#         return np.mean((true[:, :, :2] - pred[:, :, :2]) ** 2)

#     def __getitem__(self, index):
#         x = self.X[index]
#         y = self.Y[index]

#         x = torch.tensor(x, dtype=torch.float32)
#         y = torch.tensor(y, dtype=torch.float32)

#         return x, y, index

#     def __len__(self):
#         return len(self.data)


class EgoAgentNormalizedDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inference = self.config.get("INFERENCE", False)

        self.datafile = self.loadData()
        self.indices = self.config["INDICES"]

        self.data = self.datafile["data"][self.indices]
        self.X = self.datafile["X"][self.indices]

        if not self.inference:
            self.Y = self.datafile["Y"][self.indices]

    def loadData(self):
        filename = self.config["DATA_FILENAME"]
        return np.load(os.path.join(DATA_PATH, filename))

    def getOriginalSpacePredictions(self, normY, indices):
        Y = data_create.createUnnormalizedY(normY)
        #return data_create.createOriginalSpacePredictions(Y, self.data[indices])
        local_indices = [self.indices.tolist().index(i) for i in indices]
        return data_create.createOriginalSpacePredictions(Y, self.data[local_indices])

    def computeOriginalSpaceMetric(self, true, pred):
        return np.mean((true[:, :, :2] - pred[:, :, :2]) ** 2)

    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype=torch.float32)

        if self.inference:
            # create dummy tensor with same shape as Y (used during training)
            y = torch.zeros((60, 5), dtype=torch.float32)  # or match your Y shape
        else:
            y = torch.tensor(self.Y[index], dtype=torch.float32)

        return x, y, self.indices[index]

    def __len__(self):
        return len(self.X)
