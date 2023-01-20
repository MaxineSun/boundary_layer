import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class data_load:
    def __init__(self, filedir, first_n_col):  # read the excel file, and pick the first 5 columns
        self.dir = filedir
        self.n_col = first_n_col

    def data_prepare(self):
        rawdata = pd.read_excel(self.dir, engine='openpyxl').iloc[:, :self.n_col]  # load the excel file
        nplist = rawdata.T.to_numpy()
        # features
        data = nplist[0:-1].T
        data[:, 2:] = np.log(data[:, 2:])  # since the range of the last 2 columns is too large, take the log of them
        data[:, 2] = normalization(data[:, 2])  # do the normalization since the range become too small
        data[:, 3] = normalization(data[:, 3])
        data = torch.from_numpy(np.float32(data))
        # targets
        target = torch.from_numpy(nplist[-1])
        target = torch.from_numpy(np.float32(target))
        # make the dataloader fot pytorch
        dataset = Data.TensorDataset(data, target)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [145, 40])  # split the dataset
        train_dataset = Data.DataLoader(
            dataset=train_dataset,
            batch_size=29,
            shuffle=True,
        )
        return train_dataset, test_dataset
