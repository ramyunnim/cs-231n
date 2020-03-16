# 자동으로 그래디언트 해주는 것(autograd)을 내 맘대로 바꿀 수 있음. 그 코드임(ReLU 방법). 이렇게 할 필요는 없음 사실..
import torch
import numpy as np


class ReLU(torch.autograd.Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.clamp(min=0)

    def backward(self, grad_y):
        x, = self.saved_tensors
        grad_input = grad_y.clone()
        grad_input[x < 0] = 0
        return grad_input