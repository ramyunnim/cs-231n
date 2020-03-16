import numpy as np
from keras.models import Sequntial
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD


# 여기에는 CPU/GPU 설정 없음 알아서 찾으셈.

N, D, H = 64, 1000, 1000

model = Sequential()
model.add(Dense(input_dim=d, ouput_dim = H))
model.add(Activation('relu'))
model.add(Dense(input_dim=H, ouput_dim=D))

optimizer = SGD(lr=1e0)
model.compile(loss='mean_squared_error', optimizer=optimizer)

x = np.random.randn(N, D)
y = np.random.randn(N, D)
history = model.fit(x, y, nb_epoch=50, batch_size=N, verbose=0)     # 전체 학습과정이 알아서 진행