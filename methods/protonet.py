# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        # collapsing all the feature dims and take the average over the support set
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1) # (n_way, feat_dims), 1 vector per class of n_way
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1) # (n_way * n_query, feat_dims)

        dists = euclidean_dist(z_query, z_proto) # (n_query*n_way, n_way)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        # y_query shape: (n_query * n_way,), e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] if n_way = 3, n_query = 4
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query )).long()
        y_query = Variable(y_query.to(self.device))

        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # note this function computes squared euclidean distance (no sqrt)
    # x: N x D
    # y: M x D
    n = x.size(0) # n_way * n_query
    m = y.size(0) # n_way
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2) # sum of squared differences, sum over feature dimensions
