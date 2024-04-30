import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import torch.nn.functional as F
import numpy as np

from data_utils import * 
from models import XRD_ConvEmb

params_list = []
indx = 0
for indx in range(2):
    param = torch.load(f"param_{indx}.pt", map_location=torch.device('cpu'))
    params_list.append(param)
    print(param.size())
    indx += 1

elem2vec = params_list[1]

class ElemFormer(nn.Module):
    def __init__(self):
        super(ElemFormer, self).__init__()
        self.encoder_layers = TransformerEncoderLayer(40, 10, 40, 0)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, 5)
        self.flatten = nn.Flatten()
        self.composition_net = nn.Sequential(
            nn.Linear(40, 40),
            nn.ReLU(), 
            nn.BatchNorm1d(40),
            nn.Linear(40, 40)
        )
        self.batchnorm = nn.BatchNorm1d(40)
        self.elem2vec = elem2vec

    def forward(self, c): 
        c = torch.matmul(c, self.elem2vec)
        c = self.transformer_encoder(c)
        c = torch.mean(c, dim = 1)
        c = self.composition_net(c)
        return c

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.seq_len = int(8500 / d_model)
        self.ConvEmb = nn.Sequential(
            nn.BatchNorm1d(self.seq_len),
            nn.Conv1d(in_channels = self.seq_len, out_channels = self.seq_len, kernel_size = 21, stride=1, padding = 10),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.BatchNorm1d(self.seq_len),
            # nn.Conv1d(in_channels = self.seq_len, out_channels = self.seq_len, kernel_size = 21, stride=1, padding = 10),
            # nn.Dropout(0.3),
            # nn.BatchNorm1d(self.seq_len),
            # nn.ReLU(),
            # nn.Conv1d(in_channels = self.seq_len, out_channels = self.seq_len, kernel_size = 21, stride=1, padding = 10),
        )

        self.ConvEmb = XRD_ConvEmb(in_channels=1, output_dim = 8500)

        # 2 layer MLP 
        self.mlp = nn.Sequential(
            nn.Linear(12160, 2300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2300, 1150),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1150, 230)
        )

        # self.conv_layers = nn.Sequential(
        #     nn.Conv1d(17, 80, kernel_size = 25, stride=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.BatchNorm1d(80),
        #     nn.Conv1d(80, 80, 10, stride=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.BatchNorm1d(80),
        #     nn.Conv1d(80, 80, 5, stride=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.BatchNorm1d(80),
        # )

         # Calculate flattened_size dynamically
        #self.flattened_size = self._get_flattened_size(input_shape=(1, 17, 500))
        # self.MLP = nn.Sequential(
        #     nn.Linear(self.flattened_size, 2300),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     #nn.BatchNorm1d(2300),
        #     nn.Linear(2300, 1150),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     #nn.BatchNorm1d(1150),
        #     nn.Linear(1150, 230)
        # )

        self.pooler = nn.AvgPool1d(kernel_size = self.seq_len)

        self.init_weights()
        self.flatten = nn.Flatten()

        self.elem_former = ElemFormer()
        self.merge_net = nn.Sequential(
            nn.Linear(270, 230),
            # nn.ReLU(),
            # nn.BatchNorm1d(150),
            # nn.Dropout(0.5),
            # nn.Linear(150,230)
        )

    def init_weights(self) -> None:
        initrange = 0.1
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.shape))
    

    def forward(self, src: Tensor, composition: Tensor = None, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.ConvEmb(src)
        src = tokenize_xrd(src, token_size=10, seq_len=12160)
        # src = F.normalize(src, p=2, dim=1)
    
        #src = src * math.sqrt(self.d_model)
        #src = self.ConvEmb(src)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        #output = src
        #reshape the output
        # n, seq_len, d_model = output.size()
        # output = output.reshape(n, d_model, seq_len)
        # #output = self.pooler(output)
        #output = output[:, :, 0]  
        #output = self.conv_layers(output)
        #output = self.flatten(output)
        output = untokenize_xrd(src, token_size=10, seq_len=12160)
        #output = self.MLP(output)
        output = self.mlp(output.squeeze())

        if composition is not None: 
            composition = self.elem_former(composition)
            merged = torch.cat([output, composition], dim = 1)
            output = self.merge_net(merged)

        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + 0.1*self.pe[:x.size(0)]
        return x