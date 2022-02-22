import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from generator.generator import CoocurrenceNUFDataset, NUFDataset
from model._loss import utility_loss, mrs_loss
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from baselines.vae_cf import MultiVAE


class NeuralUtilityTrainer(object):
    def __init__(self, users, items, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps,
                 use_cuda=False,
                 user_item_rating_map=None, item_rating_map=None, c_size=None, s_size=None, n_items=None,
                 checkpoint=False, model_path=None, model_name=None, X_val=None, y_val=None, lmbda=.1, parallel=True,
                 max_iter=None):
        self.users = users
        self.items = items
        self.y_train = y_train
        self.loss = loss
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_step = loss_step_print
        self.eps = eps
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.user_item_rating_map = user_item_rating_map
        self.item_rating_map = item_rating_map
        self.c_size = c_size
        self.s_size = s_size
        self.n_items = n_items
        self.checkpoint = checkpoint
        self.model_path = model_path
        self.X_val = X_val
        self.y_val = y_val
        self.use_cuda = use_cuda
        self.n_gpu = torch.cuda.device_count()
        self.lmbda = lmbda
        self.max_iter = max_iter

        print(self.device)

        if self.use_cuda and self.n_gpu > 1 and parallel:
            self.model = nn.DataParallel(model)  # enabling data parallelism
            print("Parallel processing enabled")
        else:
            self.model = model

        self.model.to(self.device)

        if model_name is None:
            self.model_name = 'model'
        else:
            self.model_name = model_name

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_item_user_indices(self, batch):

        ## user ids in first column, item id's in remaining
        user_ids = batch[:, 0]
        item_ids = batch[:, 1:]

        return user_ids, item_ids

    def get_dataset(self, users, items, y_train, use_utility_loss):
        if use_utility_loss:
            return CoocurrenceNUFDataset(users, items, y_train, batch_size=self.batch_size,
                                         user_item_rating_map=self.user_item_rating_map,
                                         item_rating_map=self.item_rating_map,
                                         c_size=self.c_size, s_size=self.s_size, n_item=self.n_items)
        else:
            return NUFDataset(users, items, y_train, n_item=self.n_items)

    def checkpoint_model(self, suffix):

        if self.checkpoint:

            if self.model_path is None:
                fname = "{}_{}.pt".format(self.model_name, suffix)
            else:
                fname = "{}/{}_{}.pt".format(self.model_path, self.model_name, suffix)

            with open(fname, 'wb') as f:
                torch.save(self.model, f)

    def get_validation_loss(self, X_val, y_val):

        y_hat = self.model.forward(X_val)
        val_loss = self.loss(y_true=y_val, y_hat=y_hat)
        print("---> Validation Error: {:.4f}".format(val_loss.data.numpy()))
        return val_loss

    def print_device_specs(self):

        if self.use_cuda:
            print("Training on GPU: {} devices".format(self.n_gpu))
        else:
            print("Training on CPU")

    def _get_input_grad(self, loss, x):

        x_grad_all = torch.autograd.grad(loss, x, retain_graph=True)[0]
        x_grad = torch.sum(torch.mul(x_grad_all, x), dim=-1)

        return x_grad

    """def forward_prop(self, batch):
        y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)
        loss = self.loss(y_true=batch['y'], y_hat=y_hat)
        return loss, y_hat

    def forward_prop_vae(self, batch):
        recon_batch, mu, logvar = self.model.forward(batch['users'], batch['items'])
        loss = self.loss(recon_batch, batch["y"], mu, logvar)
        y_hat = torch.mean(mu, dim=1)
        return loss, y_hat"""

    @classmethod
    def get_gradient(cls, model, loss, users, items, y_true):
        items = items.requires_grad_(True)
        y_hat = model.forward(users, items)
        loss_val = loss(y_true=y_true, y_hat=y_hat)

        x_grad_all = torch.autograd.grad(loss_val, items, retain_graph=True)[0]
        x_grad = torch.sum(torch.mul(x_grad_all, items), dim=-1)

        return x_grad.data.numpy()

    def _check_max_iter(self, i):
        if self.max_iter is None:
            return False
        if i >= self.max_iter:
            print("Stopping. Max iterations reached: {}".format(i))
            return True

    def fit(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.dataset = self.get_dataset(self.users, self.items, self.y_train, False)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        num_batches = len(dataloader)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(dataloader):

                batch['users'] = batch['users'].to(self.device)
                batch['items'] = batch['items'].squeeze().to(self.device)
                batch['y'] = batch['y'].to(self.device)

                # zero gradient
                self.optimizer.zero_grad()

                y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)
                loss = self.loss(y_true=batch['y'], y_hat=y_hat)

                if self.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                loss = loss.detach()
                cum_loss += loss

                if iter % self.loss_step == 0:
                    if iter == 0:
                        avg_loss = cum_loss
                    else:
                        avg_loss = cum_loss / self.loss_step
                    #todo: uncomment this
                    print("iteration: {} - loss: {:.5f}".format(iter, avg_loss))
                    # print("{:.5f}".format(avg_loss.item()))
                    cum_loss = 0

                    loss_arr.append(avg_loss)

                    if abs(prev_loss - loss) < self.eps:
                        print('early stopping criterion met. Finishing training')
                        print("{:.4f} --> {:.5f}".format(prev_loss, loss))
                        break
                    else:
                        prev_loss = loss

                if i == (num_batches - 1):
                    # Check if epoch is ending. Checkpoint and get evaluation metrics
                    self.checkpoint_model(suffix=iter)
                    if self.X_val is not None:
                        _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

                iter += 1

                stop = self._check_max_iter(iter)
                if stop:
                    break
            if stop:
                break

        self.checkpoint_model(suffix='done')
        return loss_arr

    def user_item_batch(self, input):
        x_user_batch = input[:, 0]
        x_batch = input[:, 1]
        return x_user_batch, x_batch

    def fit_utility_loss(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.dataset = self.get_dataset(self.users, self.items, self.y_train, True)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

        num_batches = len(dataloader)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(dataloader):

                batch['y'] = batch['y'].to(self.device)
                batch['y_c'] = batch['y_c'].to(self.device)
                batch['y_s'] = batch['y_s'].to(self.device)

                batch['items'] = batch['items'].requires_grad_(True).to(self.device)
                batch['x_c'] = batch['x_c'].requires_grad_(True).to(self.device)
                batch['x_s'] = batch['x_s'].requires_grad_(True).to(self.device)
                batch['users'] = batch['users'].to(self.device)

                y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)
                y_hat_c = self.model.forward(batch['users'], batch['x_c']).to(self.device)
                y_hat_s = self.model.forward(batch['users'], batch['x_s']).to(self.device)

                # TODO: Make this function flexible in the loss type (e.g., MSE, binary CE)
                loss_u = utility_loss(y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s),
                                      batch['y'], batch['y_c'], batch['y_s'], self.loss)

                if self.n_gpu > 1:
                    loss_u = loss_u.mean()

                x_grad = self._get_input_grad(loss_u, batch['items'])
                x_c_grad = self._get_input_grad(loss_u, batch['x_c'])
                x_s_grad = self._get_input_grad(loss_u, batch['x_s'])

                loss = mrs_loss(loss_u, x_grad.reshape(-1, 1), x_c_grad, x_s_grad, lmbda=self.lmbda)

                if self.n_gpu > 1:
                    loss = loss.mean()

                # zero gradient
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                loss = loss.detach()
                cum_loss += loss

                if iter % self.loss_step == 0:
                    if iter == 0:
                        avg_loss = cum_loss
                    else:
                        avg_loss = cum_loss / self.loss_step
                    print("iteration: {} - loss: {:.5f}".format(iter, avg_loss))
                    cum_loss = 0

                    loss_arr.append(avg_loss)

                    if abs(prev_loss - loss) < self.eps:
                        print('early stopping criterion met. Finishing training')
                        print("{:.4f} --> {:.5f}".format(prev_loss, loss))
                        break
                    else:
                        prev_loss = loss

                if i == (num_batches - 1):
                    # Check if epoch is ending. Checkpoint and get evaluation metrics
                    self.checkpoint_model(suffix=iter)
                    if self.X_val is not None:
                        _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

                iter += 1

                stop = self._check_max_iter(iter)
                if stop:
                    break

            if stop:
                break

        self.checkpoint_model(suffix='done')
        return loss_arr

    def fit_pairwise_ranking_loss(self):

        """see Rendle et al UAI'2009"""

        self.print_device_specs()
        assert (self.s_size == 1)

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.dataset = self.get_dataset(self.users, self.items, self.y_train, True)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(dataloader):

                batch['y'] = batch['y'].to(self.device)
                batch['y_s'] = batch['y_s'].to(self.device)

                batch['items'] = batch['items'].requires_grad_(True).to(self.device)
                batch['x_s'] = batch['x_s'].requires_grad_(True).to(self.device)
                batch['users'] = batch['users'].to(self.device)

                y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)

                y_hat_s = self.model.forward(batch['users'], batch['x_s']).to(self.device)

                if y_hat_s.ndim == 3:
                    y_hat_s = y_hat_s.squeeze(-1)

                if y_hat.ndim == 3:
                    y_hat = y_hat.squeeze(-1)

                y_hat_diff = y_hat - y_hat_s

                # zero gradient
                self.optimizer.zero_grad()

                loss = self.loss(y_true=batch['y'], y_hat=torch.sigmoid(y_hat_diff))

                if self.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                loss = loss.detach()
                cum_loss += loss

                if iter % self.loss_step == 0:
                    if iter == 0:
                        avg_loss = cum_loss
                    else:
                        avg_loss = cum_loss / self.loss_step
                    print("iteration: {} - loss: {:.5f}".format(iter, avg_loss))
                    cum_loss = 0

                    loss_arr.append(avg_loss)

                    if abs(prev_loss - loss) < self.eps:
                        print('early stopping criterion met. Finishing training')
                        print("{:.4f} --> {:.5f}".format(prev_loss, loss))
                        break
                    else:
                        prev_loss = loss

                if i == (num_batches - 1):
                    # Check if epoch is ending. Checkpoint and get evaluation metrics
                    self.checkpoint_model(suffix=iter)
                    if self.X_val is not None:
                        _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

                iter += 1

                stop = self._check_max_iter(iter)
                if stop:
                    break

            if stop:
                break

        self.checkpoint_model(suffix='done')
        return loss_arr

    def fit_pairwise_utility_loss(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.dataset = self.get_dataset(self.users, self.items, self.y_train, True)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(dataloader):

                batch['y'] = batch['y'].to(self.device)
                batch['y_c'] = batch['y_c'].to(self.device)
                batch['y_s'] = batch['y_s'].to(self.device)

                batch['items'] = batch['items'].requires_grad_(True).to(self.device)
                batch['x_c'] = batch['x_c'].requires_grad_(True).to(self.device)
                batch['x_s'] = batch['x_s'].requires_grad_(True).to(self.device)
                batch['users'] = batch['users'].to(self.device)

                y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)

                if y_hat.ndim == 3:
                    y_hat = y_hat.squeeze(-1)

                y_hat_c = torch.sigmoid(self.model.forward(batch['users'], batch['x_c']).to(self.device))
                y_hat_s = torch.sigmoid(self.model.forward(batch['users'], batch['x_s']).to(self.device))

                # classify difference of each x_ui to the first sample x_ij
                y_hat_diff = torch.sigmoid(y_hat - y_hat_s[:, 0, :])

                # TODO: Make this function flexible in the loss type (e.g., MSE, binary CE)
                loss_u = utility_loss(y_hat_diff, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s),
                                      batch['y'], batch['y_c'], batch['y_s'], self.loss)

                if self.n_gpu > 1:
                    loss_u = loss_u.mean()

                x_grad = self._get_input_grad(loss_u, batch['items'])
                x_c_grad = self._get_input_grad(loss_u, batch['x_c'])
                x_s_grad = self._get_input_grad(loss_u, batch['x_s'])

                loss = mrs_loss(loss_u, x_grad.reshape(-1, 1), x_c_grad, x_s_grad, lmbda=self.lmbda)

                if self.n_gpu > 1:
                    loss = loss.mean()

                # zero gradient
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                loss = loss.detach()
                cum_loss += loss

                if iter % self.loss_step == 0:
                    if iter == 0:
                        avg_loss = cum_loss
                    else:
                        avg_loss = cum_loss / self.loss_step
                    print("iteration: {} - loss: {:.5f}".format(iter, avg_loss))
                    cum_loss = 0

                    loss_arr.append(avg_loss)

                    if abs(prev_loss - loss) < self.eps:
                        print('early stopping criterion met. Finishing training')
                        print("{:.4f} --> {:.5f}".format(prev_loss, loss))
                        break
                    else:
                        prev_loss = loss

                if i == (num_batches - 1):
                    # Check if epoch is ending. Checkpoint and get evaluation metrics
                    self.checkpoint_model(suffix=iter)
                    if self.X_val is not None:
                        _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

                iter += 1

                stop = self._check_max_iter(iter)
                if stop:
                    break

            if stop:
                break

        self.checkpoint_model(suffix='done')
        return loss_arr

    def predict(self, users, items, y=None, batch_size=32):

        #todo: uncomment this
        # print("Getting predictions on device: {} - batch size: {}".format(self.device, batch_size))

        self.dataset.update_data(users=users, items=items,
                                   y=y)

        predict_dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)


        n = users.shape[0]
        preds = list()
        cntr = 0

        for epoch in range(1):
            for i, test in enumerate(predict_dataloader):

                test['users'] = test['users'].to(self.device)
                test['items'] = test['items'].to(self.device)

                preds_batch = self.model.forward(test['users'], test['items']).detach().data.cpu().numpy()
                preds.append(preds_batch)

                progress = 100 * (cntr / n)
                #todo: uncomment this
                # print("inference progress: {:.2f}".format(progress), end='\r')

                cntr += batch_size

        preds = np.concatenate(preds, axis=0)

        return preds


class SequenceTrainer(NeuralUtilityTrainer):
    def __init__(self, users, items, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps,
                 use_cuda=False,
                 user_item_rating_map=None, item_rating_map=None, c_size=None, s_size=None, n_items=None,
                 checkpoint=False, model_path=None, model_name=None, X_val=None, y_val=None, lmbda=.1, seq_len=5,
                 parallel=False, max_iter=None, grad_clip=None):

        super().__init__(users, items, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps, use_cuda,
                         user_item_rating_map, item_rating_map, c_size, s_size, n_items,
                         checkpoint, model_path, model_name, X_val, y_val, lmbda, parallel)
        self.seq_len = seq_len
        self.max_iter = max_iter
        self.grad_clip = grad_clip

    def get_generator(self, users, items, y_train, use_utility_loss):

        #AGAGHGHGHAG Not sure what to do here

        return SeqCoocurrenceGenerator(users, items, y_train, batch_size=self.batch_size,
                                       user_item_rating_map=self.user_item_rating_map,
                                       item_rating_map=self.item_rating_map, shuffle=True,
                                       c_size=self.c_size, s_size=self.s_size, n_item=self.n_items,
                                       seq_len=self.seq_len)

    def init_hidden(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        return torch.zeros(1, batch_size, self.h_dim_size)

    def fit_utility_loss(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.dataset = self.get_dataset(self.users, self.items, self.y_train, True)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(dataloader):
                batch = self.generator.get_batch(as_tensor=True)

                batch['y'] = batch['y'].to(self.device)
                batch['y_c'] = batch['y_c'].to(self.device)
                batch['y_s'] = batch['y_s'].to(self.device)

                batch['items'] = batch['items'].requires_grad_(True).to(self.device)
                batch['x_c'] = batch['x_c'].requires_grad_(True).to(self.device)
                batch['x_s'] = batch['x_s'].requires_grad_(True).to(self.device)
                batch['users'] = batch['users'].to(self.device)

                y_hat = self.model.forward(batch['users'], batch['items']).to(self.device)

                batch["x_c"] = batch["x_c"].view(self.batch_size, self.c_size * self.seq_len, -1)
                y_hat_c = self.model.forward(batch['users'], batch['x_c']).to(self.device)
                y_hat_c = y_hat_c.view(self.batch_size, self.seq_len, self.c_size)

                batch["x_s"] = batch["x_s"].view(self.batch_size, self.s_size * self.seq_len, -1)
                y_hat_s = self.model.forward(batch['users'], batch['x_s']).to(self.device)
                y_hat_s = y_hat_s.view(self.batch_size, self.seq_len, self.s_size)

                # TODO: Make this function flexible in the loss type (e.g., MSE, binary CE)
                loss_u = utility_loss(y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s),
                                      batch['y'], batch['y_c'], batch['y_s'])

                if self.n_gpu > 1:
                    loss_u = loss_u.mean()

                x_grad = self._get_input_grad(loss_u, batch['items'])
                x_c_grad = self._get_input_grad(loss_u, batch['x_c'])
                x_s_grad = self._get_input_grad(loss_u, batch['x_s'])

                # x_grad = x_grad.view(self.batch_size, self.seq_len)
                x_c_grad = x_c_grad.view(self.batch_size, self.seq_len, self.c_size)
                x_s_grad = x_s_grad.view(self.batch_size, self.seq_len, self.s_size)

                loss = mrs_loss(loss_u, x_grad.unsqueeze(-1), x_c_grad, x_s_grad, lmbda=self.lmbda)

                if self.n_gpu > 1:
                    loss = loss.mean()

                # zero gradient
                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                loss = loss.detach()
                cum_loss += loss

                if iter % self.loss_step == 0:
                    if iter == 0:
                        avg_loss = cum_loss
                    else:
                        avg_loss = cum_loss / self.loss_step
                    print("iteration: {} - loss: {:.5f}".format(iter, avg_loss))
                    cum_loss = 0

                    loss_arr.append(avg_loss)

                    if abs(prev_loss - loss) < self.eps:
                        print('early stopping criterion met. Finishing training')
                        print("{:.4f} --> {:.5f}".format(prev_loss, loss))
                        break
                    else:
                        prev_loss = loss

                    if i == (num_batches - 1):
                        # Check if epoch is ending. Checkpoint and get evaluation metrics
                        self.checkpoint_model(suffix=iter)
                        if self.X_val is not None:
                            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

                    iter += 1

                    stop = self._check_max_iter(iter)
                    if stop:
                        break

                if stop:
                    break

        self.checkpoint_model(suffix='done')
        return loss_arr

    def predict(self, users, items, y=None, batch_size=32):

        print("Getting predictions on device: {} - batch size: {}".format(self.device, batch_size))

        # self.generator.update_data(users=users, items=items,
        #                            y=y, shuffle=False,
        #                            batch_size=batch_size)
        n = users.shape[0]
        preds = list()

        cntr = 0

        while self.generator.epoch_cntr < 1:
            test = self.generator.get_batch(as_tensor=True)

            test['users'] = test['users'].to(self.device)
            test['items'] = test['items'].to(self.device)

            preds_batch = self.model.forward(test['users'], test['items'])
            preds_batch = preds_batch.detach().data.cpu().numpy()
            preds.append(preds_batch)

            progress = 100 * (cntr / n)
            print("inference progress: {:.2f}".format(progress), end='\r')

            cntr += batch_size

        preds = np.concatenate(preds, axis=0)

        return preds


class VariationalTrainer(NeuralUtilityTrainer):
    def __init__(self, users, items, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps,
                 use_cuda=False,
                 user_item_rating_map=None, item_rating_map=None, c_size=None, s_size=None, n_items=None,
                 checkpoint=False, model_path=None, model_name=None, X_val=None, y_val=None, lmbda=.1,
                 parallel=False):

        super().__init__(users, items, y_train, model, loss, n_epochs, batch_size, lr, loss_step_print, eps, use_cuda,
                         user_item_rating_map, item_rating_map, c_size, s_size, n_items,
                         checkpoint, model_path, model_name, X_val, y_val, lmbda, parallel)

    def fit(self):

        self.print_device_specs()

        if self.X_val is not None:
            _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

        loss_arr = []

        iter = 0
        cum_loss = 0
        prev_loss = -1

        self.dataset = self.get_dataset(self.users, self.items, self.y_train, False)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(dataloader):

                batch['users'] = batch['users'].to(self.device)
                batch['items'] = batch['items'].to(self.device)
                batch['y'] = batch['y'].to(self.device)

                # zero gradient
                self.optimizer.zero_grad()

                recon_batch, mu, logvar = self.model.forward(batch['users'], batch['items'])
                loss = self.loss(recon_batch, batch["y"], mu, logvar)

                if self.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()
                loss = loss.detach()
                cum_loss += loss

                if iter % self.loss_step == 0:
                    if iter == 0:
                        avg_loss = cum_loss
                    else:
                        avg_loss = cum_loss / self.loss_step
                    print("iteration: {} - loss: {:.5f}".format(iter, avg_loss))
                    cum_loss = 0

                    loss_arr.append(avg_loss)

                    if abs(prev_loss - loss) < self.eps:
                        print('early stopping criterion met. Finishing training')
                        print("{:.4f} --> {:.5f}".format(prev_loss, loss))
                        break
                    else:
                        prev_loss = loss

            if i == (num_batches - 1):
                # Check if epoch is ending. Checkpoint and get evaluation metrics
                self.checkpoint_model(suffix=iter)
                if self.X_val is not None:
                    _ = self.get_validation_loss(self.X_val[:, 1:], self.y_val)

            iter += 1

            stop = self._check_max_iter(iter)
            if stop:
                break

            if stop:
                break

        self.checkpoint_model(suffix='done')
        return loss_arr

    def predict(self, users, items, y=None, batch_size=32):

        print("Getting predictions on device: {} - batch size: {}".format(self.device, batch_size))

        self.generator.update_data(users=users, items=items,
                                   y=y, shuffle=False,
                                   batch_size=batch_size)
        n = users.shape[0]
        preds = list()

        cntr = 0
        while self.generator.epoch_cntr < 1:
            test = self.generator.get_batch(as_tensor=True)

            test['users'] = test['users'].to(self.device)
            test['items'] = test['items'].to(self.device)

            recon_batch, mu, logvar = self.model.forward(test['users'], test['items'])
            preds_batch = torch.mean(mu, dim=1).detach().data.cpu().numpy()

            preds.append(preds_batch)

            progress = 100 * (cntr / n)
            print("inference progress: {:.2f}".format(progress), end='\r')

            cntr += batch_size

        preds = np.concatenate(preds, axis=0)

        return preds
