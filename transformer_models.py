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

class ElemFormer(nn.Module):
    """
    A PyTorch module that implements the ElemFormer model.

    The ElemFormer model consists of a TransformerEncoder layer followed by a composition network.
    It takes an input tensor, applies element-wise multiplication with a learnable parameter, passes it through the TransformerEncoder,
    takes the mean along the sequence dimension, and then passes it through the composition network.

    Args:
        None

    Attributes:
        encoder_layers (nn.TransformerEncoderLayer): The TransformerEncoderLayer module.
        transformer_encoder (nn.TransformerEncoder): The TransformerEncoder module.
        flatten (nn.Flatten): The Flatten module.
        composition_net (nn.Sequential): The composition network module.
        batchnorm (nn.BatchNorm1d): The BatchNorm1d module.
        elem2vec (torch.Tensor): The learnable parameter for element-wise multiplication.

    Methods:
        forward(c): Performs a forward pass of the ElemFormer model.

    """

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
        
        params_list = []
        indx = 0
        for indx in range(2):
            param = torch.load(f"param_{indx}.pt", map_location=torch.device('cpu'))
            params_list.append(param)
            print(param.size())
            indx += 1
        
        elem2vec = params_list[1]
        
        self.elem2vec = elem2vec

    def forward(self, c): 
        """
        Performs a forward pass of the ElemFormer model.

        Args:
            c (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        c = torch.matmul(c, self.elem2vec)
        c = self.transformer_encoder(c)
        c = torch.mean(c, dim = 1)
        c = self.composition_net(c)
        return c

class XRD_ConvEmb(nn.Module):
    """
    Convolutional Embedding module for XRD data.

    Args:
        in_channels (int): Number of input channels.
        output_dim (int): Dimension of the output.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
    """

    def __init__(self, in_channels, output_dim, dropout=0.5, batch_norm=True):
        super(XRD_ConvEmb, self).__init__()
        self.flatten = nn.Flatten()
        if batch_norm:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels, 80, kernel_size=100, stride=5),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(80),
                nn.Conv1d(80, 80, 50, stride=5),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(80),
                nn.Conv1d(80, 80, 25, stride=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(80),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels, 80, kernel_size=100, stride=5),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(80, 80, 50, stride=5),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(80, 80, 25, stride=2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Calculate flattened_size dynamically
        self.flattened_size = self._get_flattened_size(input_shape=(1, in_channels, 8500))
        print(self.flattened_size)

    def _get_flattened_size(self, input_shape):
        """
        Calculate the size of the flattened output tensor.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            int: Size of the flattened output tensor.
        """
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.shape))

    def forward(self, x):
        """
        Forward pass of the XRD_ConvEmb module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x.unsqueeze(1)

class FullModel(nn.Module):
    """
    FullModel is a class that implements a Transformer-based model for predicting spacegroups from XRD and composition data.

    Args:
        ntoken (int): The number of tokens in the input vocabulary.
        d_model (int): The dimensionality of the input and output embeddings.
        nhead (int): The number of attention heads in the multi-head attention mechanism.
        d_hid (int): The dimensionality of the feedforward network hidden layer.
        nlayers (int): The number of transformer encoder layers.
        dropout (float, optional): The dropout probability. Default is 0.5.
        batch_norm (bool, optional): Whether to apply batch normalization. Default is True.
    """

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, batch_norm=True):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.seq_len = int(8500 / d_model)
        self.ConvEmb = XRD_ConvEmb(in_channels=1, output_dim=8500, dropout=dropout, batch_norm=batch_norm)

        # 2 layer MLP 
        self.mlp = nn.Sequential(
            nn.Linear(12160, 2300),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2300, 1150),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1150, 230)
        )

        self.pooler = nn.AvgPool1d(kernel_size = self.seq_len)

        self.flatten = nn.Flatten()

        self.elem_former = ElemFormer()

        self.merge_net = nn.Sequential(
            nn.Linear(270, 230),
            nn.ReLU(),
            nn.BatchNorm1d(150),
            nn.Dropout(0.5),
            nn.Linear(150,230)
        )

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.shape))
    

    def forward(self, src: Tensor, composition: Tensor = None, src_mask: Tensor = None) -> Tensor:
        """
        Forward pass of the FullModel.

        Args:
            src (Tensor): The input tensor of shape [seq_len, batch_size].
            composition (Tensor, optional): The composition tensor. Default is None.
            src_mask (Tensor, optional): The source mask tensor of shape [seq_len, seq_len]. Default is None.

        Returns:
            output (Tensor): The output tensor of shape [seq_len, batch_size, ntoken].
        """
        src = self.ConvEmb(src)
        src = tokenize_xrd(src, token_size=10, seq_len=12160)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        output = untokenize_xrd(src, token_size=10, seq_len=12160)
        output = self.mlp(output.squeeze())

        if composition is not None: 
            composition = self.elem_former(composition)
            merged = torch.cat([output, composition], dim = 1)
            output = self.merge_net(merged)

        return output
    
class PositionalEncoding(nn.Module):
    """
    Positional encoding module for Transformer models.

    Args:
        d_model (int): The dimension of the input embeddings.
        dropout (float, optional): The dropout probability. Default: 0.1.
        max_len (int, optional): The maximum length of the input sequence. Default: 5000.
    """

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
        Forward pass of the positional encoding module.

        Args:
            x (Tensor): The input tensor of shape [seq_len, batch_size, embedding_dim].

        Returns:
            Tensor: The output tensor after applying positional encoding.
        """
        x = x + 0.1*self.pe[:x.size(0)]
        return x
    

