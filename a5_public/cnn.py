#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, input_channel, output_channel, m_word=21, kernel_size = 5):

        """initial cnn
        @param input_channel(int): the dimension of echar 
        @param output_channel(int): the dimension of eword
        @param kernel_size(int): the filters' length
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(input_channel, output_channel, kernel_size)
        self.maxpool = nn.MaxPool1d(m_word - kernel_size + 1)

    def forward(self, reshaped: torch.Tensor) -> torch.Tensor: #reshaped: batch * echar * mword
        conv_out = F.relu(self.conv1d(reshaped))
        conv_out = self.maxpool(conv_out)
        return conv_out.squeeze(-1) # maxpool后最后维度为1

### END YOUR CODE
if __name__ == '__main__':
    echar = 50
    eword = 4
    batch_size = 3
    mword = 21
    cnn = CNN(echar, eword)
    inpt = torch.randn(batch_size, echar, mword) # batch * echar * mword
    pred = cnn(inpt) # batch * eword
    assert(pred.shape == (batch_size, eword))
