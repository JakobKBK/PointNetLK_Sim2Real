import torch
from pyntcloud import PyntCloud
import pandas as pd

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data.loc['path']
        self.labels = data.loc['labels']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        label = self.labels.iloc[idx]
        path_dp = self.data.iloc[idx]
        cloud_temp = PyntCloud.from_file(path_dp)
        dp = cloud_temp.points
        dp = torch.tensor(dp.values)
        label = torch.tensor(label)

        return dp, label