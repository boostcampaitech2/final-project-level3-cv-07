import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, inputs):
        recurrent, _ = self.rnn(inputs)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
    
class ConvBNMish(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(nOut)
        self.mish = nn.Mish(True)
    def forward(self, inputs):
        conv = self.conv(inputs)
        bn = self.bn(conv)
        return self.mish(bn)
    
class ResidueBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.residue = nn.Sequential(
            ConvBNMish(channel, channel),
            ConvBNMish(channel, channel)
        )
    def forward(self, inputs):
        residue = self.residue(inputs)
        return residue + inputs
    
class Unit(nn.Module):
    def __init__(self, nIn, nOut, unit_num):
        super().__init__()
        self.unit = nn.Sequential(
            ConvBNMish(nIn, nOut),
            ResidueBlock(nOut)
        )
        if unit_num == 1:
            self.unit.add_module('ResidueBlock2',
                               ResidueBlock(nOut))
        else:
            self.unit.add_module('Conv4',
                               ConvBNMish(nOut, nOut))
    def forward(self, inputs):
        return self.unit(inputs)

class RNetMish(nn.Module):
    def __init__(self, nc, nclass, nh):
        super().__init__()

        self.cnn = nn.Sequential(
            Unit(nc, 64, 1),
            nn.MaxPool2d(2, 2),
            Unit(64, 128, 2),
            nn.MaxPool2d(2, 2),
            Unit(128, 256, 3),
            nn.MaxPool2d((2, 1), (2, 1)),
            Unit(256, 512, 4),
            nn.MaxPool2d((2, 1), (2, 1)),
            # ConvBNReLU(512, 512)
            ConvBNMish(512, 512, 2, 0)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nclass)
        )

    def forward(self, inputs):
        # conv features
        conv = self.cnn(inputs)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output
    
if __name__ == '__main__':
    inputs = torch.rand((2, 1, 16, 128))
    output = RNet(1, 29, 256)(inputs)
    print(output)