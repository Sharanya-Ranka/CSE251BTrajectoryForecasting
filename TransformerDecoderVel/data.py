import TransformerDecoderVel.data_create as data_create
from torch.utils.data import Dataset
import numpy as np
import os
import utilities as utils
import torch

DATA_PATH = os.path.join(utils.INTERMEDIATE_DATA_DIR, "TransformerDecoderVel")


class AllAgentsNormalizedDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datafile = self.loadData()
        self.indices = self.config["INDICES"]
        self.data = self.datafile["data"][self.indices]
        self.norm_data = self.datafile["normalized"][self.indices]

        # breakpoint()

    def loadData(self):
        filename = self.config["DATA_FILENAME"]
        datafile = np.load(os.path.join(DATA_PATH, filename))

        return datafile

    def getOriginalSpacePredictions(self, normY, indices, indicator="none"):
        if indicator == 'true':
            return self.data[indices, 0:1, :, :2]
        else:
            Y = normY[:, 0:1, :, :4]
            origY = data_create.createOriginalSpacePredictions(
                Y, self.data[indices, 0:1, :, :2]
            )

            prev = self.data[indices, 0:1, 49:50, :2]
            preds = [self.data[indices, 0:1, :50, :2]]

            for step in range(49, 109):
                cur_pred = prev + 0.1 * origY[:, 0:1, step:step+1, 2:4]
                preds.append(cur_pred)
                prev = cur_pred
                # breakpoint()

            pos_preds = np.concatenate(preds, axis=2)
            # breakpoint()
            return pos_preds
            
                
        # breakpoint()

        return origY

    def computeOriginalSpaceMetric(self, true, pred):
        # breakpoint()
        # print(f"computeUnnormalizedMetric: true={true[0, :, :]}, pred={pred[0, :, :]}")
        # true_pos = true[:, 0:1, 1:, :2]
        # pred_vel_based_pos = (true[:, 0:1, :, :2] + pred[:, 0:1, :, 2:4] * 0.1)[:, 0:1, :-1, :2]
        
        return np.mean((true - pred) ** 2)

    def __getitem__(self, index):
        x = self.norm_data[index]

        x = torch.tensor(x, dtype=torch.float32)

        return x, index

    def __len__(self):
        return len(self.data)
