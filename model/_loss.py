import torch
from torch.nn import BCELoss


def loss_mse(y_true, y_hat):
    err = y_true - y_hat
    mse = torch.mean(torch.pow(err, 2))

    return mse


def utility_loss(y_hat, y_hat_c, y_hat_s, y_true, y_true_c, y_true_s, loss_func):

    y_true_all = torch.cat((y_true.flatten(), y_true_c.flatten(), y_true_s.flatten()))
    y_hat_all = torch.cat((y_hat.flatten(), y_hat_c.flatten(), y_hat_s.flatten()))

    return loss_func(y_true_all, y_hat_all)


def mrs_loss(utility_loss, x_grad, x_c_grad, x_s_grad, lmbda=1):
    mrs_c = -(x_grad / x_c_grad)
    mrs_s = -(x_grad / x_s_grad)

    c_norm = torch.norm(mrs_c, dim=1)
    s_norm = torch.log(torch.norm(mrs_s, dim=1))

    mrs_loss = torch.mean(c_norm - s_norm)

    loss = lmbda*mrs_loss + utility_loss

    return loss

def loss_logit(y_true, y_hat):
    loss_calc = BCELoss()

    loss = loss_calc(y_hat, y_true)
    return loss
