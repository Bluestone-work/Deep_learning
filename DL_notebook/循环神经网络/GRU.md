## 门控循环单元
## 概念
1.什么是门控，gated
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930121452.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930121458.png)
把信息尽量的放到隐藏状态内。

## 门
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930121749.png)

## 候选隐状态
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930122210.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930122218.png)
圈点是按元素乘法，假设Rt中的元素靠近0的话，乘出来的结果就会靠近0，等于就是把上一个时刻的隐藏状态变为0。
如果Rt里面全是1，就相当于直接把前面的隐藏状态拿过来。算是一个控制单元。

## 隐状态
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930122549.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930122614.png)
假如说Zt接近于0的时候基本就回到了RNN的情况，就不去拿过去的状态了，只看现在的状态。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930123257.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240930123311.png)
