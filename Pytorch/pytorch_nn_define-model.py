# 많이 사용하게 될 것인 pytorch에서 새로운 모듈을 만들어 주는 방법 pytorch사용하는 가장 일반적인 방법
# 데이터 로더부분 추가하면서 for문 달라짐. 중요한 부분 minibatch관리. 학습도중 disk에서 minibatches를 가져오는 일련의 작업들을 multi-threading을 통해 알아서 관리해줌


import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        # linear1,2 이 2개의 module object를 클래스 안에 저장장


    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)       # x는 variable이며, linear1을 통과함.  autograd op를 사용하여 relu를 계산
        y_pred = self.linear2(h_relu)               # 그 출력값은 linear2를 통과후 출력
        return y_pred
        # 네트워크 출력 계산하기 전에 정의한 모듈을 사용할 수 있고 다양한 autograd도 사용할 수 있음



N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

loader = DataLoader(TensorDataset(x, y), batch_size=0)              # dataloader는 dataset을 wrapping하는 일종의 추상화 객체를 제공. 실제로 데이터를 이용하고자 할 때 데이터를 어떤 방식으로 읽은것인지 명시하는 dataset class만 작성해주면 이 클래스를 래핑시켜서 학습시킬 수 있음.

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


# for t in range(500):
for epoch in range(10):
    for x_batch, y_batch in loader:
        x_var, y_var = Variable(x), Variable(y)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()