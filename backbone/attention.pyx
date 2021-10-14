import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, in_channel):
        super(Attention, self).__init__()
        self.avePool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_channel, in_channel)
        self.linear2 = nn.Linear(in_channel, in_channel)
        # self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        fm = self.avePool(feature_map)
        fm = fm.view(B, -1)
        fm1 = self.linear1(fm)
        fm1 = self.relu(fm1)

        fm2 = self.linear2(fm1)
        fm3 = self.sigmoid(fm2)

        fm = fm3.view(B, -1, 1, 1)
        fm = fm.expand(B, C, H, W)
        return fm * feature_map

