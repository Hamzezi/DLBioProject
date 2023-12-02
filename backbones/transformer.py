import torch
from torch import nn as nn
import math
from backbones.fcnet import ConceptNetMixin


class TransformerEncoderNet(nn.Module, ConceptNetMixin):
    #TODO: this seems to work FAR better than TransformerDecoderNet

    def __init__(self, x_dim, layer_dim=None, ffw_dim=64, go_mask=None, dropout=0.2):
        super(TransformerEncoderNet, self).__init__()
        ConceptNetMixin.__init__(self, x_dim, go_mask=None, mask_method="index", num_GOs=20) # more patches => fewer parameters
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim, nhead=1, dim_feedforward=ffw_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.final_feat_dim = self.input_dim
    
    def forward(self, x):
        x = self.get_masked_inputs(x) # (batch, n_concepts, numGenes)
        x = self.encoder(x)
        return x


class TransformerDecoderNet(nn.Module, ConceptNetMixin):
    #TODO: this has subpar performance compared to EnFCNet

    def __init__(self, x_dim, layer_dim=None, ffw_dim=64, go_mask=None, dropout=0.2):
        super(TransformerDecoderNet, self).__init__()
        ConceptNetMixin.__init__(self, x_dim, go_mask=None, mask_method="index", num_GOs=20)

        self.n_decoder_concepts = 10
        self.concept_embeds = nn.Embedding(self.n_decoder_concepts, self.input_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.input_dim, nhead=1, dim_feedforward=ffw_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.final_feat_dim = self.input_dim
    
    def forward(self, x, mask=True):
        if mask:
            x = self.get_masked_inputs(x) # (batch, n_concepts, numGenes)
        concept_indices = torch.arange(self.n_decoder_concepts, device=x.device)
        concepts = self.concept_embeds(concept_indices).unsqueeze(0).repeat(x.shape[0], 1, 1)
        concepts = self.decoder(concepts, x)
        return concepts


class TransformerNet(nn.Module, ConceptNetMixin):

    def __init__(self, x_dim, layer_dim=None, ffw_dim=64, go_mask=None, dropout=0.2):
        super(TransformerNet, self).__init__()
        ConceptNetMixin.__init__(self, x_dim, go_mask=None, mask_method="index", num_GOs=20)

        self.encoder = TransformerEncoderNet(x_dim, layer_dim, ffw_dim, go_mask, dropout)
        self.decoder = TransformerDecoderNet(x_dim, layer_dim, ffw_dim, go_mask, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x, mask=False)
        return x
