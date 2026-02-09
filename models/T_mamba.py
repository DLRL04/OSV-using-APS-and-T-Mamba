#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTW-Mamba Model for Signature Recognition
A deep learning model combining Mamba state space models with Dynamic Time Warping
for few-shot signature verification tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import Mamba

from soft_dtw_cuda import SoftDTW

class TangoMamba(nn.Module):
    """
    Bidirectional Mamba module with residual connections.
    
    Processes sequences in both forward and backward directions and combines the results.
    
    Args:
        d_model (int): Model dimension
        d_state (int): State dimension for SSM
        d_conv (int): Convolution kernel size
        expand (int): Expansion factor for hidden dimension
        num_layers (int): Number of Mamba layers
        flip_dim (int): Dimension to flip for backward pass (typically 1 for time dimension)
    """
    def __init__(self, d_model, d_state, d_conv, expand, num_layers, flip_dim):
        super().__init__()
        self.mambas = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.flip_dim = flip_dim

    def forward(self, x):
        """
        Forward pass with bidirectional processing.
        
        Args:
            x (Tensor): Input tensor of shape [B, L, D] or [B, D, L]
            
        Returns:
            Tensor: Combined forward and backward features
        """
        # Forward direction
        out_fwd = x.clone()
        for layer in self.mambas:
            out_fwd = layer(out_fwd) + out_fwd

        # Backward direction
        x_flipped = torch.flip(x, dims=[self.flip_dim])
        out_bwd = x_flipped.clone()
        for layer in self.mambas:
            out_bwd = layer(out_bwd) + out_bwd
        
        # Flip back if necessary
        if self.flip_dim == 1:
            out_bwd = torch.flip(out_bwd, dims=[self.flip_dim])

        return out_fwd + out_bwd
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class T_Mamba(nn.Module):
    """
    DTW-Mamba model for few-shot signature recognition.
    
    Combines Temporal Convolutional Networks, Mamba state space models, and 
    Soft-DTW for robust signature verification in few-shot learning scenarios.
    
    Args:
        n_in (int): Number of input features
        n_hidden (int): Hidden dimension size (default: 128)
        d_state (int): State dimension for Mamba (default: 128)
        d_conv (int): Convolution kernel size for Mamba (default: 4)
        expand (int): Expansion factor for Mamba (default: 2)
        n_out (int): Output feature dimension (default: 64)
        n_shot_g (int): Number of genuine samples per task (default: 5)
        n_shot_f (int): Number of forged samples per task (default: 5)
        n_task (int): Number of tasks in a batch (default: 1)
        batchsize (int, optional): Batch size
        alpha (float, optional): Not currently used
        initial_gamma (float): Initial gamma value for Soft-DTW (default: 5.0)
    """
    
    def __init__(
        self,
        n_in,
        n_hidden=128,
        d_state=256,
        d_conv=4,
        expand=2,
        n_out=64,
        n_shot_g=5,
        n_shot_f=5,
        n_task=1,
        batchsize=None,
    ):
        super().__init__()
        
        # Task configuration
        self.n_shot_g = n_shot_g
        self.n_shot_f = n_shot_f
        self.n_task = n_task
        
        # Initialize smooth CE loss mask
        self.smoothCElossMask = torch.zeros(n_task * (1 + n_shot_g + n_shot_f)).cuda()
        for i in range(n_task):
            start_idx = i * (1 + n_shot_g + n_shot_f)
            end_idx = start_idx + 1 + n_shot_g
            self.smoothCElossMask[start_idx:end_idx] = (1.0 + n_shot_g + n_shot_f) / (1.0 + n_shot_g)
        
        if batchsize is None:
            batchsize = (n_shot_g + n_shot_f + 1) * n_task
        
        # Feature extraction layers
        self.conv4 = TemporalConvNet(num_inputs=n_in, num_channels=[256, 128])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)        
        # Mamba layers
        self.TangoMamba = TangoMamba(
            d_model=128,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=1,
            flip_dim=1
        )
        
        self.norm = nn.LayerNorm(128)
        # Soft-DTW for sequence alignment
        self.dtw = SoftDTW(use_cuda=True, gamma=5.0, normalize=False, bandwidth=0.1)

    def getOutputMask(self, lens):
        """
        Generate output mask based on sequence lengths.
        
        Args:
            lens (array-like): Array of sequence lengths
            
        Returns:
            np.ndarray: Binary mask indicating valid positions
        """
        lens = np.array(lens, dtype=np.int32)
        lens = (lens + 1) // 2
        N = len(lens)
        D = np.max(lens)
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask

    def forward(self, x, mask):
        length = torch.sum(mask, dim=1)
        length, indices = torch.sort(length, descending=True)
        x = torch.index_select(x, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        # CNN feature extraction
        output = x.transpose(1, 2)  # (N, D, T)
        output = self.conv4(output)
        output = self.pool(output)
        output = output.transpose(1, 2)  # (N, T, D)
        output = self.norm(output)
        output = output * mask.unsqueeze(2)
        
        # Mamba processing
        output = self.TangoMamba(output)

        # Recover original order
        _, indices = torch.sort(indices, descending=False)
        output = torch.index_select(output, 0, indices)
        length = torch.index_select(length, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        # Adjust for pooling
        if self.training:
            length = (length // 2).float()
            output = F.max_pool1d(output.permute(0, 2, 1), 2, 2, ceil_mode=False).permute(0, 2, 1)
        else:
            length = length.float()
            output = output * mask.unsqueeze(2)

        return output, length

    def EuclideanDistances(self, a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
        bt = b.t()
        return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))

    def tripletLoss(self, x, length, margin=3.0, ga=0.001):

        Ng = self.n_shot_g
        Nf = self.n_shot_f
        Nt = self.n_task
        step = 1 + Ng + Nf
        
        anchor_list = []
        var = triLoss_std = triLoss_hard = 0
        
        for i in range(Nt):
            # Extract anchor, positive, and negative samples
            anchor = x[i * step]
            anchor_list.append(anchor)
            
            pos = x[i * step + 1:i * step + 1 + Ng]
            neg = x[i * step + 1 + Ng:(i + 1) * step]
            
            len_a = length[i * step]
            len_p = length[i * step + 1:i * step + 1 + Ng]
            len_n = length[i * step + 1 + Ng:(i + 1) * step]
            
            # Compute DTW distances
            dist_g = torch.zeros(len(pos), dtype=x.dtype, device=x.device)
            dist_n = torch.zeros(len(neg), dtype=x.dtype, device=x.device)
            
            # Positive distances (anchor to genuine)
            for j in range(len(pos)):
                dist_g[j] = self.dtw(
                    anchor[None, :int(len_a)],
                    pos[j:j+1, :int(len_p[j])]
                ) / (len_a + len_p[j])
            
            # Negative distances (anchor to forged)
            for j in range(len(neg)):
                dist_n[j] = self.dtw(
                    anchor[None, :int(len_a)],
                    neg[j:j+1, :int(len_n[j])]
                ) / (len_a + len_n[j])
            
            var += torch.sum(dist_g) / Ng
            triLoss = F.relu(dist_g.unsqueeze(1) - dist_n.unsqueeze(0) + margin)  # (Ng, Nf)
            
            triLoss_std += torch.mean(triLoss)
            
            num_active = triLoss.data.nonzero(as_tuple=False).size(0)
            triLoss_hard += torch.sum(triLoss) / (num_active + 1)
        var = var / Nt
        triLoss_std = triLoss_std / Nt
        triLoss_hard = triLoss_hard / Nt
        
        return [triLoss_std, triLoss_hard, var]