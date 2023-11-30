import torch
from torch import nn as nn
import math

from backbones.blocks import full_block, full_block_fw

class FCNet(nn.Module):
    fast_weight = False  # Default

    def __init__(self, x_dim, layer_dim=[64, 64], dropout=0.2, fast_weight=False):
        super(FCNet, self).__init__()
        self.fast_weight = fast_weight

        layers = []
        in_dim = x_dim
        for dim in layer_dim:
            if self.fast_weight:
                layers.append(full_block_fw(in_dim, dim, dropout))
            else:
                layers.append(full_block(in_dim, dim, dropout))
            in_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.final_feat_dim = layer_dim[-1]

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def generate_simple_go_mask(x_dim, num_GOs=20):
    # Generate a simple mask by dividing the genes into num_GOs groups
    go_mask = []
    assert x_dim % num_GOs == 0, f"x_dim must be divisible by num_GOs: {x_dim}, {num_GOs}"
    num_genes_per_go = x_dim // num_GOs
    for i in range(num_GOs):
        go_mask.append(list(range(i * num_genes_per_go, (i + 1) * num_genes_per_go)))
    return go_mask


class EnFCNet(nn.Module):

    def __init__(self, x_dim, layer_dim=None, go_mask=None, mask_method="index", dropout=0.2):
        super(EnFCNet, self).__init__()

        if go_mask is None:
            self.go_mask = generate_simple_go_mask(x_dim=x_dim, num_GOs=20) # for testing
        else:
            self.go_mask = go_mask

        self.num_GOs = len(self.go_mask)
        self.masks = None
        self.n_concepts = self.num_GOs# + 1
        self.mask_method = mask_method

        if mask_method == "index":
            self.input_dim = self.compute_input_dim(x_dim)
        elif mask_method == "multiply":
            self.input_dim = x_dim
        else:
            raise ValueError("Unsupported masking method: {}".format(mask_method))

        self.conv1 = nn.Conv1d(self.input_dim, layer_dim[0], 1, bias=True)
        self.conv2 = nn.Conv1d(layer_dim[0], layer_dim[1], 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(self.n_concepts)
        self.bn2 = nn.BatchNorm1d(self.n_concepts)
        self.final_feat_dim = layer_dim[1]

    def compute_input_dim(self, x_dim):
        if self.n_concepts == self.num_GOs + 1:
            return x_dim # the final concept is the original input
        num_genes_per_go = max([len(genes) for genes in self.go_mask]) # all GOs have different number of genes
        num_genes_per_go = int(math.ceil(num_genes_per_go / 8) * 8) # make it divisible by 8
        return num_genes_per_go

    def generate_masks(self, x):
        batch, num_genes = x.shape
        self.masks = torch.zeros(self.n_concepts, batch, num_genes, device=x.device)
        for i, genes in enumerate(self.go_mask):
            self.masks[i, :, genes] = 1
        if self.n_concepts == self.num_GOs + 1:
            self.masks[-1, :, :] = 1

    def get_masked_inputs(self, x):
        assert self.mask_method in ["multiply", "index"], "Unsupported masking method: {}".format(self.mask_method)
        batch, num_genes = x.shape
        if self.masks is None or self.masks.shape[1] != batch:
            self.generate_masks(x)
            self.masks = self.masks.to(x.device) # (n_concepts, batch, num_genes)
        if self.mask_method == "multiply":
            x = x.view(1, batch, -1)
            x = self.masks * x # (n_concepts, batch, numGenes)
        else: # we index the original x according to each concept in self.masks (bool tensor)
            # note input_dim is << num_genes (2866 genes in Tabula Muris)
            x_new = torch.zeros(self.num_GOs, batch, self.input_dim, device=x.device)
            masks = self.masks.bool()
            for i in range(self.num_GOs):
                mask_i = masks[i].view(batch, num_genes)
                x_i = torch.masked_select(x, mask_i).view(batch, -1)
                x_new[i][:, :x_i.shape[1]] = x_i # pad with zeros
            x = x_new
        return x

    def forward(self, x):
        # need to generate masks if the batch size change or
        x = self.get_masked_inputs(x) # (n_concepts, batch, numGenes)
        x = x.permute(1, 2, 0) # (batch, numGenes, n_concepts)
        x = self.conv1(x) # 1x1 conv operating on the n_concepts dimension independently
        x = x.permute(0, 2, 1) # (batch, n_concepts, numGenes)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1) # (batch, numGenes, n_concepts)
        x = self.conv2(x)
        x = x.permute(0, 2, 1) # (batch, n_concepts, numGenes)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
