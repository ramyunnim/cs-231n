# pytorch는 pretrained model이 가장 잘 되어있음 아래 있는 코드만 적어주면 됨. 가중치 처음사용할 때 자동으로 pretrained model을 다운로드 할 수 있게 함.
# pytorch는 visdom도 해줌. loss가 얼마나 나오는지 시각화 해주는 것.


import torch
import torchvision

alexnet - torchvision.models.alexnet(pretrained=True)
vgg16 = torchvision.models.vgg16(pretrained=True)
resnet101 = torchvision.models.resnet101(pretrained=True)