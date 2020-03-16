# nn패키지는 high level wrappers 제공

import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad = False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out))
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    y_pred = model(x)               # prediction 결과
    loss = loss_fn(y_pred, y)

    #model.zero_grad()
    optimizer.zero_grad()                           # optimizer 썼을 때 zero_grad()
    loss.backward()

    optimizer.step()                                # optimizer 썼을 때 업데이트

    # for param in model.parameters():                            # 모델 업데이트 부분
    #     param.data -= learning_rate * param.grad.data