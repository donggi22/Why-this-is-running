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
    z1 = np.dot(x, w1) + b1
    h1 = relu(z1)
    z2 = np.dot(h1, w2) + b2
    h2 = sigmoid(z2)

    loss = (1/2 * (y - h2)**2).mean()

    dh2 = -(y - h2)
    dz2 = dh2 * h2 * (1-h2)
    dw2 = np.dot(h1.T, dz2)
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



#--- class ver

import numpy as np

class Affine():
    def __init__(self, m, n):
        self.w = np.random.randn(m, n)
        self.b = np.random.randn(1, n)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dx):
        self.dw = np.dot(self.x.T, dx)
        self.db = np.sum(dx, axis=0, keepdims=True)
        return np.dot(dx, self.w.T)

class Relu():
    def forward(self, x):
        self.x = x
        self.out = np.maximum(0, x)
        return self.out

    def backward(self, dx):
        return dx * (self.x > 0)

class Sigmoid():
    def forward(self, x):
        self.out = 1/(np.exp(-x)+1)
        return self.out

    def backward(self, dx):
        return dx*self.out*(1-self.out)

class MSE():
    def forward(self, h, y):
        self.h = h
        self.y = y
        return (1/2 * (h - y) ** 2).mean()

    def backward(self):
        return (self.h - self.y)

class XOR():
    def __init__(self):
        self.affine_1 = Affine(2, 8)
        self.affine_2 = Affine(8, 1)
        self.relu = Relu()
        self.sigmoid = Sigmoid()
        self.mse = MSE()
        
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

    def learn(self):
        epochs = 1000
        lr = 0.1

        for epoch in range(epochs):
            o = self.affine_1.forward(self.x)
            z = self.relu.forward(o)
            a = self.affine_2.forward(z)
            h = self.sigmoid.forward(a)
            cost = self.mse.forward(h, self.y)

            dh = self.mse.backward()
            da = self.sigmoid.backward(dh)
            dz = self.affine_2.backward(da)

            dw2 = self.affine_2.dw
            db2 = self.affine_2.db

            do = self.relu.backward(dz)
            dx = self.affine_1.backward(do) # 실제로 활용은 안 하는 코드지만 dw, db 사용하기 위해 backward 호출.

            dw1 = self.affine_1.dw
            db1 = self.affine_1.db

            self.affine_2.w -= lr*dw2
            self.affine_2.b -= lr*db2
            self.affine_1.w -= lr*dw1
            self.affine_1.b -= lr*db1

            if epoch % 100 == 99:
                print(cost)
                if cost < 0.01:
                    print(f'최종 cost: {cost}, 학습이 잘됐습니다.')
                    break

        else:
            if cost > 0.01:
                print(f'최종 cost: {cost}, 학습이 제대로 되지 않았습니다.')
        print(f'결과: \n {h}')

    def predict(self, x):
        o = self.affine_1.forward(x)
        z = self.relu.forward(o)
        a = self.affine_2.forward(z)
        h = self.sigmoid.forward(a)

        if h > 0.5:
            return 1
        else:
            return 0

