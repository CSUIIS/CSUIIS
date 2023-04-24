import torch
import torch.nn as nn
import torch.nn.functional as F


class MTTSNLNet(nn.Module):
    def __init__(self, layers, dropout=None):
        torch.manual_seed(1024)
        super(MTTSNLNet, self).__init__()
        self.fc0 = nn.Linear(layers[0], layers[1])
        self.fc1 = nn.Linear(layers[1], layers[2])
        self.fc2 = nn.Linear(layers[2], layers[3])
        self.fc_final = nn.Linear(layers[3], 1)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()

        self.dropout = dropout
        if self.dropout is not None:
            print(f'Using dropout: {dropout}')
            self.dropout0 = nn.Dropout(p=dropout)
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)
        else:
            self.dropout0 = nn.Identity()
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
        self._initialize_weights()

    def forward(self, x):
        x = self.dropout0(self.relu(self.fc0(x)))
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.fc2(x)
        hidden = x
        x = self.dropout2(self.relu(x))
        x = self.act(self.fc_final(x))

        return x, hidden

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


from torchstat import stat

if __name__ == '__main__':
    model = MTTSNLNet([25, 23, 18, 14])
    print(count_param(model))
