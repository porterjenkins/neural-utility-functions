import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from generator.generator import CoocurrenceGenerator, Generator
from torch import nn
import numpy as np

class Predictor(object):

    def __init__(self, model, batch_size, users, items, y, n_items, use_cuda=False):
        self.model = model
        self.batch_size = batch_size
        self.users = users
        self.items = items
        self.y = y
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.n_items = n_items
        self.generator = self.get_generator(users, items, y)
        self.n_gpu = torch.cuda.device_count()

        print(self.device)
        #if self.use_cuda and self.n_gpu > 1:
        #    self.model = nn.DataParallel(model)  # enabling data parallelism
        #else:
        #    self.model = model

        self.model.to(self.device)


    def get_generator(self, users, items, y_train):

            return Generator(users, items, y_train, batch_size=self.batch_size, n_item=self.n_items, shuffle=False)

    def predict(self):

        print("Getting predictions on device: {} - batch size: {}".format(self.device, self.batch_size))

        n = self.users.shape[0]
        preds = list()

        cntr = 0
        while self.generator.epoch_cntr < 1:


            test = self.generator.get_batch(as_tensor=True)

            test['users'] = test['users'].to(self.device)
            test['items'] = test['items'].to(self.device)

            preds_batch = self.model.forward(test['users'], test['items']).detach().data.cpu().numpy()
            preds.append(preds_batch)

            progress = 100*(cntr / n)
            print("inference progress: {:.2f}".format(progress), end='\r')

            cntr += self.batch_size

        preds = np.concatenate(preds, axis=0)


        return preds
