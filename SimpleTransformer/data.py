import SimpleTransformer.data_create as data_create
from torch.utils.data import Dataset
import numpy as np
import os
import utilities as utils


class EgoAgentNormalizedDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datafile = self.loadData()
        self.indices = self.config["INDICES"]
        self.data = self.datafile["data"][self.indices]
        # breakpoint()

    def loadData(self):
        filename = self.config["DATA_FILENAME"]
        datafile = np.load(os.path.join(utils.INTERMEDIATE_DATA_DIR, filename))

        return datafile

    def unnormalizeData(self, norm_data, indices):
        params = {
            # "pos_max": self.datafile["pos_max"][self.indices][indices],
            # "pos_min": self.datafile["pos_min"][self.indices][indices],
            "pos_initial": self.datafile["pos_initial"][self.indices][indices],
            "vel_multiplier": 1 / 15,  # self.datafile["vel_multipier"],
            "head_multiplier": 1 / 3.14159,  # self.datafile["head_multiplier"],
        }
        data_unnorm = data_create.unnormalizePredictions(norm_data, params)
        # breakpoint()

        return data_unnorm

    def computeUnnormalizedMetric(self, true, pred):
        # breakpoint()
        # print(f"computeUnnormalizedMetric: true={true[0, :, :]}, pred={pred[0, :, :]}")
        return np.mean((true[:, :, :2] - pred[:, :, :2]) ** 2)

    def __getitem__(self, index):
        x = self.data[index][:49]
        # if self.config.get("INFERENCE", False) == False:
        y = self.data[index][49:]

        return x, y, index

    def __len__(self):
        return len(self.data)
