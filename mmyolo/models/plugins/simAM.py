import torch
import torch.nn as nn
from mmyolo.registry import MODELS


@MODELS.register_module()
class SimAM(torch.nn.Module):
    def __init__(self, in_channels, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)


if __name__ == '__main__':
    input = torch.randn(3, 64, 7, 7)
    model = SimAM()
    outputs = model(input)
    print(outputs.shape)
