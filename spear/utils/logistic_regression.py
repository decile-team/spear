import torch.nn as nn

class LogisticReg(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticReg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
