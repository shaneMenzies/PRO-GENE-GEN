import torch
from torch import nn

__all__ = ["one_hot_embedding", "weights_init"]


def one_hot_embedding(y, num_classes=10, dtype=torch.FloatTensor, device="cpu"):
    """
    apply one hot encoding on labels
    :param y:
    :param num_classes:
    :param dtype:
    :return:
    """
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.LongTensor).view(*y.size(), -1).to(device)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype).to(device)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)