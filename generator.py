##
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import config.config as cfg


class MovieDataSet(Dataset):

    def __init__(self):
        #do some data loading
        data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

        xy = np.loadtxt(data_dir + "ratings.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.X = torch.from_numpy(xy[:, [0, 1]])
        self.y = torch.from_numpy(xy[:,[2]])
        self.n_samples = xy.shape[0]

        pass

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    batch_size = 8
    num_workers = 3
    num_epochs = 10
    initialTime = time.time()

    dataset = MovieDataSet()

    # num_workers denotes the amount of subprocesses you want.
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #dummy training loop

    total_samples = len(dataset)
    num_iterations = math.ceil(total_samples/ batch_size)
    a = 0

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            a+= 1
            # you'd do some training here
            # print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_iterations}, labels:  {labels}')

    print(a)
    print(f'Elapsed Time: {time.time() - initialTime}')

