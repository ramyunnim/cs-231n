import torch
from torch.autograd import Variable


N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)
# 뒷부분 grad 부분은 variables에 대한 grad를 계산할 것인지 결정

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)           # 텐서랑 같은 api 방법이기 때문에 같음
    loss = (y_pred - y).pow(2).sum()

    if w1.grad: w1.grad.data.zero_()
    if w2.grad: w2.grad.data.zero_()
    loss.backward()                     # 알아서 grad계산

    w1.data -= learning_rate * grad_w1.data                 # 가중치 업데이트
    w2.data -= learning_rate * grad_w2.data


"""
X는 variable이고 x.data는 tensor임. x.grad도 variable인데 loss에 대한 gradient를 담고 있음 그래서 x.grad.data가 실제 tensor를 담고 있음
tensor


"""