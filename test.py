import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

d = 1
n = 200
X = torch.rand(n,d)  #200*1, batch * feature_dim
#y = 3*torch.sin(X) + 5* torch.cos(X**2)
y = 4 * torch.sin(np.pi * X) * torch.cos(6*np.pi*X**2)

#注意这里hid_dim 设置是超参数(如果太小，效果就不好)，使用tanh还是relu效果也不同，优化器自选
hid_dim_1 = 128
hid_dim_2 = 32
d_out = 1

model = nn.Sequential(nn.Linear(d,hid_dim_1),
                     nn.Tanh(),
                     nn.Linear(hid_dim_1, hid_dim_2),
                     nn.Tanh(),
                     nn.Linear(hid_dim_2, d_out)
                     )
loss_func = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), 0.05)

epochs = 6000
print("epoch\t loss\t")
for i in range(epochs):
    y_hat = model(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if((i+1)%100 == 0):
        print("{}\t {:.5f}".format(i+1,loss.item()))

#这个地方容易出错，测试时不要用原来的x，因为原来的x不是从小到达排序，导致x在连线时会混乱，所以要用np.linspace重新来构造
test_x  = torch.tensor(np.linspace(0,1,50), dtype = torch.float32).reshape(-1,1)
final_y = model(test_x)
plt.scatter(X,y)
plt.plot(test_x.detach(),final_y.detach(),"r")  #不使用detach会报错
print("over")