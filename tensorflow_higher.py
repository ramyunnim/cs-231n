import tensorflow as tf
import numpy as np


# define
N, D, H = 64, 1000, 100
# 그래프 구성 부분
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
# w1 = tf.Variable(tf.random_normal((D, H)))                      # 이 명령어 자체가 초기화 시켜주는 것은 아님 어떻게 초기화 시키는건지 tf에게 알려주는 것
# w2 = tf.Variable(tf.random_normal((H, D)))
# w2 = tf.placeholder(tf.float32, shape=(D, H))                  GPU에서 문제 생김
# w1 = tf.Variable(tf.float32, shape=(D, H))

# 연산부분 - 내가 하고 싶은 연산을 넣어주면 됨
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.losses.mean_squared_error(y_pred, y)                      # tf에서 제공. L2를 따로 설정 안해줘도 됨.
# loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
# gradient 계산
# grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# gradient랑 learning_rate까지 한번에 해결
optimizer = tf.train.GradientDescentOpimizer(1e-5)
updates = optimizer.minimize(loss)
# learning_rate = 1e-5
# new_w1 = w1.assign(w1 - learning_rate * grad_w1)
# new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # w1, w2 를 초기화 시켜줌.
    # 할당
    values = {x: np.random.rand(N, D),
              y: np.random.rand(N, D),}
    losses = []
    # 실제 실행되는 부분      loss, grad1, grad2 값을 알고 싶고 feed_dict로 실제 값을 알고 싶음. 출력값들도 numpy array임. 반복 실행 시키고 싶으면 for문 사용하면 됨.
    # out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
    for t in range(50):
        loss_val, = sess.run([loss, updates], feed_dict=values)                  # loss 계산 해줭