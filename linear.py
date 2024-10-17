import torch
import random
from d2l import torch as d2l
def create_Data(w , b , num_e):
    "生成y = wx + b + 噪声"
    x = torch.normal(0 , 1 ,(num_e,len(w)))
    y = torch.matmul(x , w) + b
    #两个张量对应的元素相乘（element-wise），在PyTorch中可以通过torch.mul函数（或者∗ *∗运算符）,实现两个张量矩阵相乘（Matrix product），在PyTorch中可以通过torch.matmul函数实现
    y += torch.normal(0 , 0.01 , y.shape)
    return x,y.reshape((-1,1))

true_w = torch.Tensor([2,-1,3])
true_b = 4.2
features , label = create_Data(true_w,true_b,30)
print('features:',features[0],'\nlabel:',label[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),
                label.detach().numpy(), 1)