import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_hidden = 200):
        super(MLP, self).__init__()
        self.input = nn.Linear(in_features=700, out_features=n_hidden, bias=True)
        self.hidden1 = nn.Linear(n_hidden, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.hidden4 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, 256)

        nn.init.constant_(self.input.bias, 0)
        nn.init.constant_(self.hidden1.bias, 0)
        nn.init.constant_(self.hidden2.bias, 0)
        nn.init.constant_(self.hidden3.bias, 0)
        nn.init.constant_(self.hidden4.bias, 0)
        nn.init.constant_(self.predict.bias, 0)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        output = self.predict(x)
        # output = F.softmax(output, dim=1)

        return output

#  todo 参数初始化