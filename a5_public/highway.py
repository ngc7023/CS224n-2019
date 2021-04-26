#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, eword_size):
        super(Highway, self).__init__()
        self.eword_size = eword_size
        self.x_projection = nn.Linear(self.eword_size, self.eword_size, bias=True)
        self.x_highwaygate = nn.Linear(self.eword_size, self.eword_size, bias=True)

    def forward(self, conv_out: torch.Tensor) -> torch.Tensor:
        # conv_out: batch * eword
        x_proj = F.relu(self.x_projection(conv_out))
        x_gate = torch.sigmoid(self.x_highwaygate(conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1-x_gate), conv_out)
        return x_highway # batch * eword

### END YOUR CODE

if __name__ == '__main__':
    high = Highway(5)
    inpt = torch.randn(4,5)
    pred = high(inpt)
    assert(pred.shape == inpt.shape)