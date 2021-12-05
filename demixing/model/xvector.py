import math
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn import functional as F
from collections import OrderedDict


import sys
sys.path.insert(0, '..')
from demixing.engine.trainer import Trainer


class Classifier(Trainer):

    def __init__(self, n_mfcc=20, embed_dim=512, dropout_p=0.1, num_classes=630):
        super(Classifier, self).__init__()
        self.xvector = XVector(n_mfcc, dropout_p)
        self.classifier = nn.Sequential(OrderedDict([
            ('hidden', nn.Linear(embed_dim, 512)), 
            ('bn', nn.BatchNorm1d(512)), 
            ('dropout', nn.Dropout(p=dropout_p)), 
            ('nonlinearity', nn.ReLU()), 
            ('output', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        embed = self.xvector(x)
        logits = self.classifier(embed)
        return logits

    def freeze_xvector(self):
        for param in self.xvector.parameters():
            param.requires_grad = False

    def unfreeze_xvector(self):
        for param in self.xvector.parameters():
            param.requires_grad = True

    def extract_embedding(self, x):
        x = self.xvector(x)
        return x

    def encode_batch(self, dataloader, gpu=True):
        device = "cuda" if gpu else "cpu"
        self.eval()
        self.to(device)
        embeddings = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, leave=False), 0):
                inputs, targets = batch
                inputs = inputs.to(device)
                embed = self.extract_embedding(inputs)
                embeddings.extend(embed.detach().cpu())
        return np.vstack(embeddings)


class XVector(nn.Module):
    
    def __init__(
        self, 
        n_mfcc, 
        dropout_p=0.0
    ):
        super(XVector, self).__init__()
        self.extractor1 = nn.Sequential(OrderedDict([
            ('tdnn1', TDNN(n_mfcc, 512, 3, padding=math.floor(3/2))), 
            ('tdnn2', TDNN(512, 512, 1, padding=math.floor(1/2))), 
            ('tdnnres1', ResBlockTDNN(512, 512, 5, padding=math.floor(5/2))), 
            ('tdnnres2', ResBlockTDNN(512, 512, 5, padding=math.floor(5/2))), 
            ('tdnnres3', ResBlockTDNN(512, 512, 5, padding=math.floor(5/2))), 
            ('tdnn3', TDNN(512, 1500, 1, padding=math.floor(1/2)))
        ]))
        self.pool = StatsPool()
        self.extractor2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(3000, 512)), 
            ('bn1', nn.BatchNorm1d(512)), 
            ('dropout1', nn.Dropout(p=dropout_p)), 
            ('nonlinearity1', nn.ReLU()), 
            ('fc2', nn.Linear(512, 512)), 
            ('bn2', nn.BatchNorm1d(512)), 
            ('dropout2', nn.Dropout(p=dropout_p)), 
            # ('nonlinearity2', nn.ReLU())
        ]))

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Shape of [batch, step, mfcc].
        """
        # Residual TDNN based Frame-level Feature Extractor
        x = self.extractor1(x)
        # Statistics Pooling
        x = self.pool(x)
        # DNN based Segment level Feature Extractor
        x = self.extractor2(x)
        # Frobenius norm
        # x = x / torch.norm(x)
        return x


class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x


class TDNN(nn.Module):

    def __init__(
        self,
        input_dim=23,
        output_dim=512,
        context_size=5,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout_p=0.0,
        padding=0
    ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding

        self.kernel = nn.Conv1d(self.input_dim,
                                self.output_dim,
                                self.context_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)

        self.nonlinearity = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input = [batch, seq_len, input_features]
        output = [batch, new_seq_len, output_features]
        '''
        x = self.kernel(x.transpose(1, 2))
        x = self.nonlinearity(x)
        x = self.drop(x)

        if self.batch_norm:
            x = self.bn(x)
        return x.transpose(1, 2)
    

class ResBlockTDNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ResBlockTDNN, self).__init__()
        self.tdnn = TDNN(*args, **kwargs)

    def forward(self, x):
        return self.tdnn(x) + x