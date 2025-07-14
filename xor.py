import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

w1 = np.random.randn(2, 8)
b1 = np.random.randn(1, 8)
w2 = np.random.randn(8, 1)
b2 = np.random.randn(1, 1)

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

lr = 0.1
for i in range(1000):
    z1 = np.dot(x, w1) + b1 # 4, 2
    h1 = relu(z1) # 4, 2
    z2 = np.dot(h1, w2) + b2 # 4, 1
    h2 = sigmoid(z2) # 4, 1

    loss = (1/2 * (y - h2)**2).mean()

    dh2 = -(y - h2) # 4, 1
    dz2 = dh2 * h2 * (1-h2) # 4, 1
    dw2 = np.dot(h1.T, dz2) # 2, 4 @ 4, 1
    db2 = np.sum(dz2, axis=0)

    dh1 = np.dot(dz2, w2.T)
    dz1 = dh1 * (z1 > 0)
    dw1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    
    if i % 100 == 99:
        print(f'epoch: {i}, loss: {loss}')
        
    if loss < 0.01:
        print(f'epoch: {i}, loss: {loss} \n 학습이 잘됐습니다.')
        break
else:
    print('학습이 제대로 되지 않았습니다.')


z1 = np.dot(x, w1) + b1
h1 = relu(z1)
z2 = np.dot(h1, w2) + b2
h2 = sigmoid(z2)

print(f'예측값: \n{np.round(h2)}')